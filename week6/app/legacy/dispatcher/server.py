"""HTTP interface: POST /events, /endpoints, /__admin/clock and /__admin/tick."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from . import __version__
from .dispatch import Invalid, parse_time


class Handler(BaseHTTPRequestHandler):
    server_version = "switchyard-dispatcher/" + __version__

    def log_message(self, format, *args):
        pass

    def reply(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            raise Invalid("body is not JSON")
        if not isinstance(body, dict):
            raise Invalid("body must be a JSON object")
        return body

    def do_POST(self):
        dispatcher = self.server.dispatcher
        try:
            body = self.read_json()
            if self.path == "/events":
                deliveries = dispatcher.ingest(body)
                return self.reply(202, {"id": body["id"], "deliveries": deliveries})
            if self.path == "/endpoints":
                dispatcher.add_endpoint(body)
                return self.reply(201, {"id": body["id"]})
            if self.path == "/__admin/clock":
                dispatcher.clock.set(parse_time(body.get("now")))
                return self.reply(200, {"now": dispatcher.clock.now()})
            if self.path == "/__admin/tick":
                return self.reply(200, {"sent": dispatcher.tick()})
        except Invalid as problem:
            return self.reply(400, {"error": str(problem)})
        return self.reply(404, {"error": "not found"})


def make_server(dispatcher, host, port):
    """A single-threaded server. One request is handled at a time, so is one tick."""
    server = HTTPServer((host, port), Handler)
    server.dispatcher = dispatcher
    return server
