"""Plain records shared by the queue, the storage layer and the HTTP app."""

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

QUEUED = "queued"
INVISIBLE = "invisible"  # received by a worker; hidden until `visible_at`
DELIVERED = "delivered"
DEAD = "dead"

OUTCOMES = ("delivered", "failed", "unknown")


def delivery_id(event_id: str, endpoint_id: str) -> str:
    """One delivery per event and endpoint, and the id says so: routing the
    same event twice cannot produce a second delivery."""
    digest = hashlib.sha256(f"{event_id}\x1f{endpoint_id}".encode()).hexdigest()
    return f"dlv_{digest[:14]}"


@dataclass
class Endpoint:
    id: str
    merchant_id: str
    url: str
    secret: str
    event_types: List[str] = field(default_factory=lambda: ["*"])
    enabled: bool = True

    def wants(self, event_type: str) -> bool:
        return "*" in self.event_types or event_type in self.event_types


@dataclass
class Event:
    id: str
    merchant_id: str
    type: str
    created: int
    data: dict


@dataclass
class Delivery:
    id: str
    event_id: str
    endpoint_id: str
    state: str = QUEUED
    attempts: int = 0  # counted attempts: failed or unknown
    visible_at: int = 0  # QUEUED: due time. INVISIBLE: when the timeout fires.
    receipt: Optional[str] = None
    worker_id: Optional[str] = None
    created_at: int = 0
    finished_at: Optional[int] = None
    last_status: Optional[int] = None


@dataclass
class Receipt:
    """What a worker gets from `receive` and quotes on every later call."""

    receipt: str
    delivery_id: str
    attempt_no: int
    visible_until: int
