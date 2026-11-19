import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dispatcher import signing  # noqa: E402
from dispatcher.dispatch import Dispatcher, Invalid  # noqa: E402
from dispatcher.server import make_server  # noqa: E402

START = 1741082400  # 2025-03-04T10:00:00Z


class FakeClock:
    def __init__(self, now=START):
        self.value = now

    def now(self):
        return self.value

    def set(self, now):
        self.value = now

    def advance(self, seconds):
        self.value += seconds


class Merchant:
    """A merchant's server. Answers with the scripted responses in turn, then 200."""

    def __init__(self):
        self.requests = []
        self.responses = []
        merchant = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                merchant.requests.append({"path": self.path, "headers": self.headers, "body": body})
                response = merchant.responses.pop(0) if merchant.responses else 200
                if response == "slow":
                    time.sleep(0.1)
                    response = 200
                try:
                    self.send_response(response)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                except OSError:
                    pass  # the dispatcher gave up waiting

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def event(event_id="evt_1001", **changes):
    body = {"id": event_id, "merchant": "m_acme", "event_type": "payment.captured", "amount": "12.50",
            "currency": "USD", "payment": "pay_77", "ts": "2025-03-04T10:00:00Z"}
    body.update(changes)
    return body


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.clock = FakeClock()
        self.merchant = Merchant()
        self.addCleanup(self.merchant.stop)
        self.dispatcher = self.open()

    def open(self, timeout=1.0):
        dispatcher = Dispatcher(os.path.join(self.dir.name, "switchyard.db"), clock=self.clock,
                                log_path=os.path.join(self.dir.name, "dispatcher.log"), timeout=timeout)
        self.addCleanup(dispatcher.close)
        return dispatcher

    def endpoint(self, endpoint_id, path="/hooks", merchant="m_acme", event_types=("*",), url=None):
        self.dispatcher.add_endpoint({"id": endpoint_id, "merchant": merchant, "url": url or self.merchant.url + path,
                                      "secret": "secret-" + endpoint_id, "event_types": list(event_types)})

    def delivery(self, delivery_id):
        return self.dispatcher.store.get_delivery(delivery_id)

    def test_delivers_to_each_subscribed_endpoint(self):
        self.endpoint("ep_all", "/all")
        self.endpoint("ep_refunds", "/refunds", event_types=["refund.created"])
        self.endpoint("ep_captured", "/captured", event_types=["payment.captured", "payment.failed"])
        self.endpoint("ep_other", "/other", merchant="m_other")
        ids = self.dispatcher.ingest(event())
        self.assertEqual(self.dispatcher.tick(), 2)
        self.assertEqual(sorted(r["path"] for r in self.merchant.requests), ["/all", "/captured"])
        for delivery_id in ids:
            row = self.delivery(delivery_id)
            self.assertEqual((row["status"], row["confirmed"], row["attempts"]), ("delivered", 1, 1))
        self.assertEqual(self.dispatcher.tick(), 0)

    def test_delivery_ids_are_assigned_at_ingest(self):
        self.endpoint("ep_a", "/a")
        self.endpoint("ep_b", "/b")
        ids = self.dispatcher.ingest(event("evt_9f3c"))
        self.assertEqual(ids, ["dlv_9f3c_1", "dlv_9f3c_2"])
        self.assertEqual(self.merchant.requests, [])
        self.assertEqual([self.delivery(i)["endpoint_id"] for i in ids], ["ep_a", "ep_b"])
        self.assertEqual([self.delivery(i)["status"] for i in ids], ["pending", "pending"])

    def test_repeated_event_id_creates_nothing_new(self):
        self.endpoint("ep_a")
        first = self.dispatcher.ingest(event())
        self.dispatcher.tick()
        self.endpoint("ep_b")
        self.assertEqual(self.dispatcher.ingest(event()), first)
        self.assertEqual(self.dispatcher.tick(), 0)
        self.assertEqual(len(self.merchant.requests), 1)

    def test_retry_schedule(self):
        self.endpoint("ep_a")
        self.merchant.responses = [500, 503, 404, 302]
        (delivery_id,) = self.dispatcher.ingest(event())
        self.assertEqual(self.dispatcher.tick(), 1)
        for attempts, wait in enumerate([10, 60, 300, 1800], 1):
            row = self.delivery(delivery_id)
            self.assertEqual((row["status"], row["attempts"]), ("pending", attempts))
            self.assertEqual(row["next_attempt_at"], self.clock.now() + wait)
            self.clock.advance(wait - 1)
            self.assertEqual(self.dispatcher.tick(), 0)
            self.clock.advance(1)
            self.assertEqual(self.dispatcher.tick(), 1)
        row = self.delivery(delivery_id)
        self.assertEqual((row["status"], row["confirmed"], row["attempts"]), ("delivered", 1, 5))
        self.assertEqual(len(self.merchant.requests), 5)

    def test_failed_after_five_attempts(self):
        self.endpoint("ep_a")
        self.merchant.responses = [500] * 5
        (delivery_id,) = self.dispatcher.ingest(event())
        for _ in range(8):
            self.dispatcher.tick()
            self.clock.advance(3600)
        row = self.delivery(delivery_id)
        self.assertEqual((row["status"], row["attempts"], row["last_status"]), ("failed", 5, 500))
        self.assertIsNone(row["confirmed"])
        self.assertEqual(len(self.merchant.requests), 5)

    def test_connection_refused_is_retried(self):
        closed = socket.socket()
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
        closed.close()
        self.endpoint("ep_a", url="http://127.0.0.1:%d/hooks" % port)
        (delivery_id,) = self.dispatcher.ingest(event())
        self.dispatcher.tick()
        row = self.delivery(delivery_id)
        self.assertEqual((row["status"], row["attempts"]), ("pending", 1))
        self.assertEqual(row["next_attempt_at"], self.clock.now() + 10)
        self.assertIsNone(row["last_status"])

    def test_timeout_is_retried(self):
        self.dispatcher = self.open(timeout=0.03)
        self.endpoint("ep_a")
        self.merchant.responses = ["slow"]
        (delivery_id,) = self.dispatcher.ingest(event())
        self.dispatcher.tick()
        self.assertEqual(self.delivery(delivery_id)["status"], "pending")
        self.dispatcher.timeout = 1.0
        self.clock.advance(10)
        self.dispatcher.tick()
        self.assertEqual(self.delivery(delivery_id)["status"], "delivered")
        self.assertEqual(len(self.merchant.requests), 2)

    def test_sending_at_stop_is_closed_unconfirmed_and_not_sent_again(self):
        self.endpoint("ep_a")
        sending, waiting = self.dispatcher.ingest(event("evt_1")) + self.dispatcher.ingest(event("evt_2"))
        self.dispatcher.store.mark_sending(sending, 1)

        self.dispatcher = self.open()
        row = self.delivery(sending)
        self.assertEqual((row["status"], row["confirmed"], row["attempts"]), ("delivered", 0, 1))
        self.assertEqual(row["finished_at"], self.clock.now())
        for _ in range(6):
            self.dispatcher.tick()
            self.clock.advance(1800)
        self.assertEqual([r["headers"]["X-Switchyard-Delivery"] for r in self.merchant.requests], [waiting])
        self.assertEqual(self.delivery(sending)["confirmed"], 0)
        with open(os.path.join(self.dir.name, "dispatcher.log"), encoding="utf-8") as log:
            self.assertIn("2025-03-04 10:00:00 WARN closed %s unconfirmed" % sending, log.read())

    def test_amounts_are_forwarded_unchanged(self):
        self.endpoint("ep_a")
        amounts = [("USD", "12.50"), ("EUR", "0.07"), ("JPY", "1200"), ("KWD", "4.250"), ("GBP", "1000000.00")]
        for n, (currency, amount) in enumerate(amounts):
            self.dispatcher.ingest(event("evt_%d" % n, currency=currency, amount=amount))
        self.dispatcher.tick()
        bodies = [json.loads(r["body"]) for r in self.merchant.requests]
        self.assertEqual([(b["currency"], b["amount"]) for b in bodies], amounts)
        self.assertEqual(bodies[0], event("evt_0"))
        self.assertIn(b'"amount":"12.50"', self.merchant.requests[0]["body"])

    def test_bad_amounts_are_refused(self):
        self.endpoint("ep_a")
        bad = [("USD", 12.5), ("USD", 1250), ("USD", "12.5"), ("USD", "12"), ("USD", "-12.50"),
               ("JPY", "1200.00"), ("KWD", "4.25"), ("CHF", "12.50")]
        for currency, amount in bad:
            with self.assertRaises(Invalid):
                self.dispatcher.ingest(event(currency=currency, amount=amount))
        self.assertEqual(self.dispatcher.tick(), 0)

    def test_requests_are_signed(self):
        self.endpoint("ep_a")
        (delivery_id,) = self.dispatcher.ingest(event())
        self.dispatcher.tick()
        (request,) = self.merchant.requests
        self.assertEqual(request["headers"]["X-Switchyard-Delivery"], delivery_id)
        header = request["headers"]["X-Switchyard-Signature"]
        self.assertRegex(header, "^sha256=[0-9a-f]{64}$")
        self.assertTrue(signing.verify("secret-ep_a", request["body"], header))
        self.assertFalse(signing.verify("secret-ep_b", request["body"], header))

    def test_http_interface(self):
        server = make_server(self.dispatcher, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        def post(path, body):
            request = urllib.request.Request("http://127.0.0.1:%d%s" % (server.server_port, path),
                                             data=json.dumps(body).encode("utf-8"))
            try:
                with urllib.request.urlopen(request, timeout=2) as response:
                    return response.status, json.loads(response.read())
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read())

        self.merchant.responses = [500]
        endpoint = {"id": "ep_a", "merchant": "m_acme", "url": self.merchant.url + "/hooks",
                    "secret": "s3cret", "event_types": ["*"]}
        self.assertEqual(post("/endpoints", endpoint), (201, {"id": "ep_a"}))
        self.assertEqual(post("/events", event()), (202, {"id": "evt_1001", "deliveries": ["dlv_1001_1"]}))
        self.assertEqual(post("/events", event("evt_1002", amount=12.5))[0], 400)
        self.assertEqual(post("/__admin/tick", {}), (200, {"sent": 1}))
        self.assertEqual(post("/__admin/tick", {}), (200, {"sent": 0}))
        self.assertEqual(post("/__admin/clock", {"now": "2025-03-04T10:00:10Z"}), (200, {"now": START + 10}))
        self.assertEqual(post("/__admin/tick", {}), (200, {"sent": 1}))
        self.assertEqual(self.delivery("dlv_1001_1")["status"], "delivered")


if __name__ == "__main__":
    unittest.main()
