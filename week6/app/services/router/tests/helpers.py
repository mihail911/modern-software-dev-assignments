"""A router database and log in a temporary directory, for the tests."""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import leases  # noqa: E402
import routing  # noqa: E402
from store import Clock, Log, open_db  # noqa: E402

START = 1793577600


class Harness:
    """What a test needs of a router, without the HTTP server."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="router-test-")
        self.clock = Clock()
        self.clock.set(START)
        self.db = open_db(self.dir)
        self.log = Log(self.dir, self.clock)
        # Statements are printed as they run, so a failing test shows what led up to it.
        self.db.set_trace_callback(lambda sql: print("    sql>", " ".join(sql.split())))

    def close(self):
        self.db.close()
        self.log.file.close()
        with open(self.log.path, encoding="utf-8") as f:
            for line in f:
                print("    log>", line.rstrip())
        shutil.rmtree(self.dir, ignore_errors=True)

    def at(self, seconds):
        self.clock.set(START + seconds)
        return self.clock.now()

    def now(self):
        return self.clock.now()

    def endpoint(self, id_, merchant="m_acme", types=("*",), enabled=True, version=1, url=None):
        routing.apply(self.db, self.log, self.now(), "endpoint", {
            "id": id_, "merchant_id": merchant, "url": url or "http://127.0.0.1:9/hooks/" + id_,
            "secret": "whsec_dGVzdA==", "event_types": list(types), "enabled": enabled, "version": version,
        })  # fmt: skip
        self.db.commit()

    def event(self, id_, merchant="m_acme", type_="payment.captured", amount=1250, currency="USD"):
        routing.apply(self.db, self.log, self.now(), "event", {
            "id": id_, "merchant_id": merchant, "type": type_, "created": self.now(),
            "data": {"object": "payment", "payment_id": "pay_" + id_[4:], "amount_minor": amount, "currency": currency},
        })  # fmt: skip
        self.db.commit()

    def deliveries(self, event=None):
        rows = self.db.execute("select * from deliveries order by ord").fetchall()
        return [dict(r) for r in rows if event is None or r["event_id"] == event]

    def lease(self, worker="wrk_test"):
        got = leases.lease(self.db, self.log, self.now(), worker)
        self.db.commit()
        return got

    def report(self, lease_id, delivery_id, attempt_id, outcome, status=None):
        got = leases.report(self.db, self.log, self.now(), lease_id, [
            {"delivery_id": delivery_id, "attempt_id": attempt_id, "outcome": outcome, "status": status},
        ])  # fmt: skip
        self.db.commit()
        return got

    def close_lease(self, lease_id):
        leases.close(self.db, self.log, self.now(), lease_id)
        self.db.commit()
