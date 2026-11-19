import unittest

from helpers import Harness


class RoutingTest(unittest.TestCase):
    def setUp(self):
        self.r = Harness()
        self.addCleanup(self.r.close)

    def test_an_event_gets_one_delivery_per_subscribed_endpoint(self):
        self.r.endpoint("ep_a")
        self.r.endpoint("ep_b")
        self.r.endpoint("ep_other", merchant="m_bolt")
        self.r.event("evt_1")
        self.assertEqual([d["endpoint_id"] for d in self.r.deliveries()], ["ep_a", "ep_b"])

    def test_a_subscription_filters_by_type(self):
        self.r.endpoint("ep_a", types=["refund.created"])
        self.r.event("evt_1")
        self.r.event("evt_2", type_="refund.created")
        self.assertEqual([d["event_id"] for d in self.r.deliveries()], ["evt_2"])

    def test_an_event_that_arrives_twice_is_routed_once(self):
        self.r.endpoint("ep_a")
        self.r.event("evt_1")
        self.r.event("evt_1")
        self.assertEqual(len(self.r.deliveries()), 1)

    def test_a_disabled_endpoint_gets_no_delivery(self):
        self.r.endpoint("ep_a", enabled=False)
        self.r.event("evt_1")
        self.assertEqual(self.r.deliveries(), [])

    def test_an_older_endpoint_version_is_ignored(self):
        self.r.endpoint("ep_a", version=3, url="http://127.0.0.1:9/new")
        self.r.endpoint("ep_a", version=2, url="http://127.0.0.1:9/old")
        row = self.r.db.execute("select url, version from endpoints where id = 'ep_a'").fetchone()
        self.assertEqual((row["url"], row["version"]), ("http://127.0.0.1:9/new", 3))

    def test_many_events_each_get_their_own_delivery(self):
        self.r.endpoint("ep_a")
        for n in range(150):
            self.r.at(n)
            self.r.event(f"evt_{n:04d}", amount=100 + n)
        ids = [d["id"] for d in self.r.deliveries()]
        self.assertEqual(len(ids), 150)
        self.assertEqual(len(set(ids)), 150)


if __name__ == "__main__":
    unittest.main()
