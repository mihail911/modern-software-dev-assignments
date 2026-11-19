"""The visibility-timeout queue. Time is passed in; nothing here sleeps."""

import unittest

from router_next.models import DEAD, DELIVERED, QUEUED, delivery_id
from router_next.queue import MAX_ATTEMPTS, RETRY_AFTER, VisibilityQueue

T0 = 1_780_000_000


def queue_with(*names):
    queue = VisibilityQueue()
    for offset, name in enumerate(names):
        queue.put(delivery_id(f"evt_{name}", "ep_1"), f"evt_{name}", "ep_1", T0 + offset)
    return queue


class ReceiveTest(unittest.TestCase):
    def test_oldest_due_first_and_hidden_once_received(self):
        queue = queue_with("a", "b")
        first = queue.receive("wrk_1", T0 + 5)
        second = queue.receive("wrk_2", T0 + 5)
        self.assertEqual(first.delivery_id, delivery_id("evt_a", "ep_1"))
        self.assertEqual(second.delivery_id, delivery_id("evt_b", "ep_1"))
        self.assertEqual(first.attempt_no, 1)
        self.assertEqual(first.visible_until, T0 + 35)
        self.assertIsNone(queue.receive("wrk_3", T0 + 5))

    def test_same_event_and_endpoint_is_one_delivery(self):
        queue = queue_with("a")
        again = queue.put(delivery_id("evt_a", "ep_1"), "evt_a", "ep_1", T0 + 9)
        self.assertFalse(again)
        self.assertEqual(len(queue.deliveries), 1)
        self.assertNotEqual(delivery_id("evt_a", "ep_1"), delivery_id("evt_a", "ep_2"))


class TimeoutTest(unittest.TestCase):
    def test_timeout_counts_an_unknown_attempt_and_retries(self):
        queue = queue_with("a")
        receipt = queue.receive("wrk_1", T0)
        self.assertEqual(queue.sweep(T0 + 29), 0)
        self.assertEqual(queue.sweep(T0 + 30), 1)
        delivery = queue.deliveries[receipt.delivery_id]
        self.assertEqual((delivery.state, delivery.attempts), (QUEUED, 1))
        self.assertEqual(delivery.visible_at, T0 + 30 + RETRY_AFTER[0])
        self.assertIsNone(queue.receive("wrk_2", T0 + 31))
        retry = queue.receive("wrk_2", T0 + 60)
        self.assertEqual(retry.attempt_no, 2)
        self.assertNotEqual(retry.receipt, receipt.receipt)

    def test_heartbeat_extends_the_timeout_up_to_the_ceiling(self):
        queue = queue_with("a")
        receipt = queue.receive("wrk_1", T0)
        self.assertEqual(queue.heartbeat(receipt.receipt, T0 + 20), T0 + 50)
        self.assertEqual(queue.sweep(T0 + 40), 0)
        deadlines = [queue.heartbeat(receipt.receipt, now) for now in range(T0 + 45, T0 + 400, 25)]
        self.assertEqual(max(d for d in deadlines if d), T0 + queue.max_hold)
        self.assertIsNone(deadlines[-1])
        self.assertEqual(queue.deliveries[receipt.delivery_id].attempts, 1)

    def test_stale_receipt_is_refused(self):
        queue = queue_with("a")
        receipt = queue.receive("wrk_1", T0)
        queue.sweep(T0 + 30)
        self.assertIsNone(queue.heartbeat(receipt.receipt, T0 + 31))
        self.assertFalse(queue.ack(receipt.receipt, "delivered", T0 + 31, 200))
        self.assertFalse(queue.nack(receipt.receipt, T0 + 31))
        self.assertEqual(queue.deliveries[receipt.delivery_id].state, QUEUED)


class AckTest(unittest.TestCase):
    def test_nack_returns_at_once_and_counts_nothing(self):
        queue = queue_with("a")
        receipt = queue.receive("wrk_1", T0)
        self.assertTrue(queue.nack(receipt.receipt, T0 + 3))
        again = queue.receive("wrk_2", T0 + 3)
        self.assertEqual(again.delivery_id, receipt.delivery_id)
        self.assertEqual(again.attempt_no, 1)

    def test_delivered_finishes_the_delivery(self):
        queue = queue_with("a")
        receipt = queue.receive("wrk_1", T0)
        self.assertTrue(queue.ack(receipt.receipt, "delivered", T0 + 2, 204))
        delivery = queue.deliveries[receipt.delivery_id]
        self.assertEqual((delivery.state, delivery.attempts, delivery.finished_at), (DELIVERED, 0, T0 + 2))
        self.assertEqual(queue.sweep(T0 + 3600), 0)
        self.assertIsNone(queue.next_wake())

    def test_failures_follow_the_schedule_then_dead_letter(self):
        queue = queue_with("a")
        now = T0
        for attempt_no in range(1, MAX_ATTEMPTS + 1):
            receipt = queue.receive("wrk_1", now)
            self.assertEqual(receipt.attempt_no, attempt_no)
            queue.ack(receipt.receipt, "failed", now, 503)
            delivery = queue.deliveries[receipt.delivery_id]
            if attempt_no < MAX_ATTEMPTS:
                self.assertEqual(delivery.visible_at, now + RETRY_AFTER[attempt_no - 1])
                self.assertIsNone(queue.receive("wrk_1", delivery.visible_at - 1))
                now = delivery.visible_at
        self.assertEqual((delivery.state, delivery.attempts), (DEAD, MAX_ATTEMPTS))
        self.assertIsNone(queue.receive("wrk_1", now + 86400))

