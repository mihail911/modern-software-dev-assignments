#!/usr/bin/env python3
"""receiver: a stand-in for a merchant's server.

It answers webhook requests the way it has been scripted to and records what
it processed. It is a test instrument, not part of the product.

Script a path with POST /__admin/script:
    {"path": "/hooks/m1", "responses": [500, "drop", 200], "default": 200}
Each request to the path takes the next response; when the list is used up the
default applies. A response is an HTTP status, or:
    "drop"    process the request, then close the connection without answering
    "refuse"  close the connection without processing the request
"""

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEPT_HEADERS = ("webhook-id", "webhook-timestamp", "webhook-signature", "x-switchyard-attempt", "x-switchyard-event")


class Receiver:
    def __init__(self, state_dir):
        self.lock = threading.Lock()
        start = os.environ.get("SWITCHYARD_CLOCK")
        self.now = int(start) if start else None
        self.scripts = {}
        self.received = []
        os.makedirs(os.path.join(state_dir, "receiver"), exist_ok=True)
        self.path = os.path.join(state_dir, "receiver", "received.jsonl")
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.received = [json.loads(line) for line in f if line.strip()]
        self.file = open(self.path, "a", encoding="utf-8")

    def clock(self):
        return int(time.time()) if self.now is None else self.now

    def next_response(self, path):
        script = self.scripts.get(path)
        if script is None:
            return 200
        if script["responses"]:
            return script["responses"].pop(0)
        return script["default"]

    def record(self, path, headers, body, response):
        entry = {
            "n": len(self.received) + 1,
            "now": self.clock(),
            "path": path,
            "headers": {name: headers.get(name) for name in KEPT_HEADERS},
            "body": body,
            "response": response if isinstance(response, int) else None,
            "processed": response != "refuse",
            "accepted": response == "drop" or (isinstance(response, int) and 200 <= response < 300),
        }
        self.received.append(entry)
        self.file.write(json.dumps(entry) + "\n")
        self.file.flush()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True  # replies are two small writes
    receiver = None

    def log_message(self, *args):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def hang_up(self):
        self.close_connection = True
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def do_GET(self):
        receiver = self.receiver
        with receiver.lock:
            if self.path == "/__admin/health":
                return self.reply(200, {"service": "receiver", "release": "test"})
            if self.path in ("/__admin/state", "/__admin/received"):
                return self.reply(200, {"service": "receiver", "received": receiver.received})
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        receiver = self.receiver
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        with receiver.lock:
            if self.path.startswith("/__admin/"):
                body = json.loads(raw) if raw else {}
                if self.path == "/__admin/clock":
                    receiver.now = int(body.get("now", 0))
                    return self.reply(200, {"now": receiver.now})
                if self.path == "/__admin/script":
                    receiver.scripts[body["path"]] = {
                        "responses": list(body.get("responses", [])),
                        "default": body.get("default", 200),
                    }
                    return self.reply(200, {"scripted": body["path"]})
                if self.path == "/__admin/tick":
                    return self.reply(200, {"did": 0})
                return self.reply(404, {"error": "not found"})
            response = receiver.next_response(self.path)
            receiver.record(self.path, self.headers, raw.decode("utf-8", "replace"), response)
        if isinstance(response, int):
            return self.reply(response, {"received": response < 300})
        self.hang_up()


def main():
    state_dir = os.environ.get("SWITCHYARD_STATE", "./state")
    port = int(os.environ.get("SWITCHYARD_PORT_BASE", "7100")) + 4
    Handler.receiver = Receiver(state_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
