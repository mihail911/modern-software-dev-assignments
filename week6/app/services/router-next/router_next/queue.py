"""The delivery queue, with a visibility timeout per delivery instead of leases.

A worker `receive`s one delivery, which is then hidden from other workers for
VISIBILITY seconds, and gets a receipt. Each `heartbeat` hides it for another
VISIBILITY seconds. The worker finishes with `ack` (an outcome) or `nack` (give
it back now, nothing counted). A timeout that fires with no ack is an attempt
with an unrecorded result: `unknown`, counted and retried.

Held in memory, with the time passed in everywhere. `storage.py` persists it.
"""

import hashlib
from typing import Dict, Iterable, List, Optional, Set

from .models import DEAD, DELIVERED, INVISIBLE, OUTCOMES, QUEUED, Delivery, Receipt

VISIBILITY = 30
MAX_HOLD = 300  # heartbeats cannot hide a delivery for longer than this in total
MAX_ATTEMPTS = 6
RETRY_AFTER = [30, 120, 600, 3600, 21600]


class VisibilityQueue:
    def __init__(self, visibility: int = VISIBILITY, max_hold: int = MAX_HOLD):
        self.visibility = visibility
        self.max_hold = max_hold
        self.deliveries: Dict[str, Delivery] = {}
        self.order: List[str] = []
        self.paused: Set[str] = set()  # endpoint ids that are disabled
        self._received_at: Dict[str, int] = {}
        self._receipts = 0

    def put(self, delivery_id: str, event_id: str, endpoint_id: str, now: int) -> bool:
        """Add a delivery. False, and no change, if the id is already known."""
        if delivery_id in self.deliveries:
            return False
        self.deliveries[delivery_id] = Delivery(
            id=delivery_id, event_id=event_id, endpoint_id=endpoint_id, visible_at=now, created_at=now
        )
        self.order.append(delivery_id)
        return True

    def restore(self, deliveries: Iterable[Delivery]) -> None:
        for delivery in deliveries:
            self.deliveries[delivery.id] = delivery
            self.order.append(delivery.id)

    def receive(self, worker_id: str, now: int) -> Optional[Receipt]:
        """Hand out the oldest due delivery, or None when nothing is due."""
        self.sweep(now)
        due = [
            d
            for d in (self.deliveries[i] for i in self.order)
            if d.state == QUEUED and d.visible_at <= now and d.endpoint_id not in self.paused
        ]
        if not due:
            return None
        delivery = min(due, key=lambda d: d.visible_at)  # min is stable: ties go to the older
        delivery.state = INVISIBLE
        delivery.visible_at = now + self.visibility
        delivery.worker_id = worker_id
        delivery.receipt = self._new_receipt(delivery.id)
        self._received_at[delivery.id] = now
        return Receipt(delivery.receipt, delivery.id, delivery.attempts + 1, delivery.visible_at)

    def heartbeat(self, receipt: str, now: int) -> Optional[int]:
        """Extend the timeout. Returns the new deadline, or None for a stale receipt."""
        self.sweep(now)
        delivery = self._held(receipt)
        if delivery is None:
            return None
        ceiling = self._received_at[delivery.id] + self.max_hold
        delivery.visible_at = min(now + self.visibility, ceiling)
        return delivery.visible_at

    def ack(self, receipt: str, outcome: str, now: int, status: Optional[int] = None) -> bool:
        """Record the outcome of the attempt. False for a stale receipt."""
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        self.sweep(now)
        delivery = self._held(receipt)
        if delivery is None:
            return False
        delivery.last_status = status
        if outcome == "delivered":
            self._finish(delivery, DELIVERED, now)
        else:
            self._count(delivery, now)
        return True

    def nack(self, receipt: str, now: int) -> bool:
        """Give the delivery back at once. No attempt is counted."""
        self.sweep(now)
        delivery = self._held(receipt)
        if delivery is None:
            return False
        self._release(delivery)
        delivery.state = QUEUED
        delivery.visible_at = now
        return True

    def sweep(self, now: int) -> int:
        """Fire every timeout that is due. Returns how many fired."""
        fired = 0
        for delivery_id in self.order:
            delivery = self.deliveries[delivery_id]
            if delivery.state == INVISIBLE and delivery.visible_at <= now:
                self._count(delivery, delivery.visible_at)
                fired += 1
        return fired

    def next_wake(self) -> Optional[int]:
        times = [
            d.visible_at
            for d in self.deliveries.values()
            if d.state == INVISIBLE or (d.state == QUEUED and d.endpoint_id not in self.paused)
        ]
        return min(times) if times else None

    def _new_receipt(self, delivery_id: str) -> str:
        self._receipts += 1
        digest = hashlib.sha256(f"switchyard:rcp:{delivery_id}:{self._receipts}".encode()).hexdigest()
        return f"rcp_{digest[:14]}"

    def _held(self, receipt: str) -> Optional[Delivery]:
        for delivery in self.deliveries.values():
            if delivery.state == INVISIBLE and delivery.receipt == receipt:
                return delivery
        return None

    def _release(self, delivery: Delivery) -> None:
        delivery.receipt = None
        delivery.worker_id = None
        self._received_at.pop(delivery.id, None)

    def _finish(self, delivery: Delivery, state: str, now: int) -> None:
        self._release(delivery)
        delivery.state = state
        delivery.finished_at = now

    def _count(self, delivery: Delivery, now: int) -> None:
        """One more counted attempt: wait out the schedule, or dead-letter."""
        delivery.attempts += 1
        if delivery.attempts >= MAX_ATTEMPTS:
            self._finish(delivery, DEAD, now)
            return
        self._release(delivery)
        delivery.state = QUEUED
        delivery.visible_at = now + RETRY_AFTER[delivery.attempts - 1]
