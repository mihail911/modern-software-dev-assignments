#!/usr/bin/env python3
"""router: applies bus messages and serves the delivery queue."""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bus  # noqa: E402
import leases  # noqa: E402
import routing  # noqa: E402
from store import Clock, Log, open_db  # noqa: E402

RELEASE = "2.3.1"


class Router:
    def __init__(self, state_dir):
        self.clock = Clock()
        self.db = open_db(state_dir)
        self.log = Log(state_dir, self.clock)
        self.lock = threading.Lock()
        self.crash_point = None
        self.crash_hits = 0
        self.log.write("router.started", release=RELEASE)

    def crash(self, point):
        """Crash points are listed in contracts/harness.md."""
        if self.crash_point != point:
            return
        self.crash_hits -= 1
        if self.crash_hits <= 0:
            self.log.file.flush()
            os._exit(86)

    # --- bus -------------------------------------------------------------

    def on_bus(self, msg):
        try:
            producer, ident, kind, body = bus.parse(msg)
        except bus.Malformed as problem:
            self.log.write("bus.malformed", level="warn", reason=str(problem))
            return 422, {"error": str(problem)}
        seen = self.db.execute(
            "select 1 from bus_seen where producer = ? and ident = ?", (producer, ident)
        ).fetchone()
        if seen:
            return 200, {"applied": False}
        now = self.clock.now()
        self.db.execute("insert into bus_seen (producer, ident) values (?, ?)", (producer, ident))
        routing.apply(self.db, self.log, now, kind, body)
        self.db.commit()
        self.crash(1)
        return 200, {"applied": True}

    # --- queue -----------------------------------------------------------

    def on_lease(self, body):
        worker_id = body.get("worker_id")
        if not isinstance(worker_id, str):
            return 400, {"error": "worker_id is required"}
        granted = leases.lease(self.db, self.log, self.clock.now(), worker_id)
        self.db.commit()
        if granted["lease_id"] is not None:
            self.crash(2)
        return 200, granted

    def on_report(self, body):
        lease_id = body.get("lease_id")
        results = body.get("results")
        if not isinstance(lease_id, str) or not isinstance(results, list):
            return 400, {"error": "lease_id and results are required"}
        reply = leases.report(self.db, self.log, self.clock.now(), lease_id, results)
        self.db.commit()
        self.crash(3)
        return 200, reply

    def on_close(self, body):
        lease_id = body.get("lease_id")
        if not isinstance(lease_id, str):
            return 400, {"error": "lease_id is required"}
        closed = leases.close(self.db, self.log, self.clock.now(), lease_id)
        self.db.commit()
        if closed:
            self.crash(4)
        return 200, {"closed": bool(closed)}

    # --- operations ------------------------------------------------------

    def on_tick(self):
        did = leases.expire(self.db, self.log, self.clock.now())
        self.db.commit()
        return 200, {"did": did}

    def state(self):
        def rows(sql):
            return [dict(r) for r in self.db.execute(sql).fetchall()]

        return {
            "service": "router",
            "now": self.clock.now(),
            "next_wake": leases.next_wake(self.db),
            "endpoints": rows("select id, merchant_id, url, enabled, version, disabled_at from endpoints order by id"),
            "deliveries": rows(
                "select id, event_id, endpoint_id, endpoint_version, resend_of, state, attempts, available_at, lease_id, "
                "created_at, finished_at, last_status from deliveries order by ord"
            ),
            "leases": rows("select id, worker_id, granted_at, expires_at, closed_at from leases order by granted_at, id"),
            "attempts": rows(
                "select attempt_id, delivery_id, lease_id, outcome, status, reported_at from attempts "
                "order by reported_at, rowid"
            ),
        }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True  # replies are two small writes
    router = None

    def log_message(self, *args):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw) if raw else {}
        except ValueError:
            return None

    def do_GET(self):
        router = self.router
        with router.lock:
            if self.path == "/__admin/health":
                return self.reply(200, {"service": "router", "release": RELEASE})
            if self.path == "/__admin/state":
                return self.reply(200, router.state())
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        router = self.router
        body = self.body()
        if body is None:
            return self.reply(400, {"error": "body is not JSON"})
        with router.lock:
            if self.path == "/bus":
                return self.reply(*router.on_bus(body))
            if not isinstance(body, dict):
                return self.reply(400, {"error": "body must be an object"})
            if self.path == "/queue/lease":
                return self.reply(*router.on_lease(body))
            if self.path == "/queue/report":
                return self.reply(*router.on_report(body))
            if self.path == "/queue/close":
                return self.reply(*router.on_close(body))
            if self.path == "/__admin/clock":
                router.clock.set(body.get("now", 0))
                return self.reply(200, {"now": router.clock.now()})
            if self.path == "/__admin/tick":
                return self.reply(*router.on_tick())
            if self.path == "/__admin/crash_at":
                router.crash_point = body.get("point")
                router.crash_hits = int(body.get("hits", 1))
                return self.reply(200, {"armed": router.crash_point})
        self.reply(404, {"error": "not found"})


def main():
    state_dir = os.environ.get("SWITCHYARD_STATE", "./state")
    port = int(os.environ.get("SWITCHYARD_PORT_BASE", "7100")) + 2
    Handler.router = Router(state_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
