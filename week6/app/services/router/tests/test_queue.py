import unittest

from helpers import Harness


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.r = Harness()
        self.addCleanup(self.r.close)
        self.r.endpoint("ep_a")

    def test_a_lease_holds_the_oldest_due_delivery(self):
        for n in range(6):
            self.r.at(n)
            self.r.event(f"evt_{n}")
        got = self.r.lease()
        self.assertEqual(got["delivery"]["event"]["id"], "evt_0")
        self.assertEqual(got["expires_at"], self.r.now() + 120)

    def test_nothing_due_is_an_empty_lease(self):
        self.assertEqual(self.r.lease(), {"lease_id": None, "delivery": None})

    def test_a_delivered_report_finishes_the_delivery(self):
        self.r.event("evt_1")
        got = self.r.lease()
        self.r.report(got["lease_id"], got["delivery"]["delivery_id"], "att_1", "delivered", 200)
        self.assertEqual(self.r.deliveries()[0]["state"], "delivered")
        self.assertEqual(self.r.deliveries()[0]["attempts"], 0)

    def test_failed_reports_follow_the_retry_schedule(self):
        self.r.event("evt_1")
        waits = []
        for n in range(5):
            got = self.r.lease()
            delivery = got["delivery"]
            self.assertEqual(delivery["attempt_no"], n + 1)
            self.r.report(got["lease_id"], delivery["delivery_id"], f"att_{n}", "failed", 500)
            self.r.close_lease(got["lease_id"])
            row = self.r.deliveries()[0]
            waits.append(row["available_at"] - self.r.now())
            self.r.clock.set(row["available_at"])
        self.assertEqual(waits, [30, 120, 600, 3600, 21600])

    def test_a_delivery_is_dead_after_six_failed_attempts(self):
        self.r.event("evt_1")
        for n in range(6):
            got = self.r.lease()
            self.r.report(got["lease_id"], got["delivery"]["delivery_id"], f"att_{n}", "failed", 503)
            self.r.close_lease(got["lease_id"])
            self.r.clock.set(self.r.deliveries()[0]["available_at"])
        self.assertEqual(self.r.deliveries()[0]["state"], "dead")
        self.assertEqual(self.r.deliveries()[0]["attempts"], 6)

    def test_closing_a_lease_returns_what_was_not_reported(self):
        self.r.event("evt_1")
        got = self.r.lease()
        self.r.close_lease(got["lease_id"])
        self.assertEqual(self.r.deliveries()[0]["state"], "queued")
        self.assertEqual(self.r.deliveries()[0]["attempts"], 0)

    def test_the_same_attempt_reported_twice_counts_once(self):
        self.r.event("evt_1")
        got = self.r.lease()
        delivery = got["delivery"]["delivery_id"]
        self.r.report(got["lease_id"], delivery, "att_1", "failed", 500)
        self.r.report(got["lease_id"], delivery, "att_1", "failed", 500)
        self.assertEqual(self.r.deliveries()[0]["attempts"], 1)

    def test_a_report_under_the_wrong_lease_is_ignored(self):
        self.r.event("evt_1")
        got = self.r.lease()
        reply = self.r.report("lse_other", got["delivery"]["delivery_id"], "att_1", "delivered", 200)
        self.assertEqual(reply["ignored"], [got["delivery"]["delivery_id"]])
        self.assertEqual(self.r.deliveries()[0]["state"], "leased")

    def test_a_long_queue_drains_in_order(self):
        for n in range(120):
            self.r.at(n)
            self.r.event(f"evt_{n:04d}")
        seen = []
        while True:
            got = self.r.lease()
            if got["delivery"] is None:
                break
            delivery = got["delivery"]
            seen.append(delivery["event"]["id"])
            self.r.report(got["lease_id"], delivery["delivery_id"], "att_" + delivery["delivery_id"], "delivered", 200)
            self.r.close_lease(got["lease_id"])
        self.assertEqual(seen, [f"evt_{n:04d}" for n in range(120)])


if __name__ == "__main__":
    unittest.main()
