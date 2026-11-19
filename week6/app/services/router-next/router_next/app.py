"""router-next HTTP app: asyncio streams, one request per connection.

    python3 -m router_next.app [--port 7202] [--db ./state/router-next/queue.db]

Single event loop, no threads, so no lock around the queue.
"""

import argparse
import asyncio
import json
import time
from typing import Callable, Tuple

from . import RELEASE
from .models import Endpoint, Event, delivery_id
from .queue import VisibilityQueue
from .storage import Storage

Reply = Tuple[int, dict]


class App:
    def __init__(self, storage: Storage, clock: Callable[[], int] = lambda: int(time.time())):
        self.storage = storage
        self.clock = clock
        self.queue = VisibilityQueue()
        self.queue.restore(storage.load_deliveries())
        self.queue.paused = {e.id for e in storage.endpoints() if not e.enabled}
        storage.recover_inflight(self.clock())
        self.routes = {
            "GET /__admin/health": self.health,
            "POST /__admin/tick": self.tick,
            "POST /bus": self.bus,
            "POST /queue/receive": self.receive,
            "POST /queue/heartbeat": self.heartbeat,
            "POST /queue/ack": self.ack,
            "POST /queue/nack": self.nack,
        }

    async def health(self, body: dict) -> Reply:
        return 200, {"service": "router-next", "release": RELEASE}

    async def tick(self, body: dict) -> Reply:
        fired = self.queue.sweep(self.clock())
        # TODO(jun): persist only what the sweep touched, not everything.
        for delivery in self.queue.deliveries.values():
            self.storage.save_delivery(delivery)
        return 200, {"did": fired}

    # --- bus (version 3 only) --------------------------------------------

    async def bus(self, msg: dict) -> Reply:
        if msg.get("v") != 3:
            # TODO(jun): versions 1 and 2. Port the parsers or put a translator in front.
            return 422, {"error": "only version 3 messages are accepted"}
        kind, data = msg.get("type"), msg.get("data") or {}
        if kind == "event.accepted":
            return await self.on_event(data.get("event") or {})
        if kind == "endpoint.changed":
            return await self.on_endpoint(data.get("endpoint") or {})
        if kind == "resend.requested":
            # Needs a second id for the same event and endpoint. Open question in the README.
            raise NotImplementedError("resend.requested")
        return 422, {"error": "unknown message type"}

    async def on_event(self, raw: dict) -> Reply:
        # TODO(jun): validate against bus.v3.schema.json; today a bad message is a bare 400, not a 422.
        # TODO(jun): de-duplicate on (producer, seq) like the live router, not only on event id.
        event = Event(raw["id"], raw["merchant_id"], raw["type"], raw["created"], raw["data"])
        now = self.clock()
        if not self.storage.save_event(event, now):
            return 200, {"applied": False}
        for endpoint in self.storage.endpoints(event.merchant_id):
            if endpoint.enabled and endpoint.wants(event.type):
                new_id = delivery_id(event.id, endpoint.id)
                if self.queue.put(new_id, event.id, endpoint.id, now):
                    self.storage.save_delivery(self.queue.deliveries[new_id])
        return 200, {"applied": True}

    async def on_endpoint(self, raw: dict) -> Reply:
        endpoint = Endpoint(raw["id"], raw["merchant_id"], raw["url"], raw["secret"],
                            list(raw.get("event_types") or ["*"]), bool(raw.get("enabled", True)))
        was_paused = endpoint.id in self.queue.paused
        self.storage.save_endpoint(endpoint)
        if not endpoint.enabled:
            self.queue.paused.add(endpoint.id)
            return 200, {"applied": True}
        self.queue.paused.discard(endpoint.id)
        if was_paused:
            # Catch-up of the last 24 hours. The events table has the columns; nothing reads them.
            raise NotImplementedError("catch-up on re-enable")
        return 200, {"applied": True}

    # --- queue -----------------------------------------------------------

    async def receive(self, body: dict) -> Reply:
        worker_id = body.get("worker_id")
        if not isinstance(worker_id, str):
            return 400, {"error": "worker_id is required"}
        receipt = self.queue.receive(worker_id, self.clock())
        if receipt is None:
            return 200, {"receipt": None}
        self.storage.save_delivery(self.queue.deliveries[receipt.delivery_id])
        # TODO(jun): include the event and the endpoint's url and secret, as /queue/lease does.
        return 200, dict(receipt.__dict__)

    async def heartbeat(self, body: dict) -> Reply:
        deadline = self.queue.heartbeat(str(body.get("receipt")), self.clock())
        return (200, {"visible_until": deadline}) if deadline else (409, {"error": "receipt is not current"})

    async def ack(self, body: dict) -> Reply:
        ok = self.queue.ack(str(body.get("receipt")), body.get("outcome"), self.clock(), body.get("status"))
        # TODO(jun): persist. Needs receipt -> delivery id out of the queue; ack returns a bool.
        return (200, {"acked": True}) if ok else (409, {"error": "receipt is not current"})

    async def nack(self, body: dict) -> Reply:
        ok = self.queue.nack(str(body.get("receipt")), self.clock())
        return (200, {"returned": True}) if ok else (409, {"error": "receipt is not current"})

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status, reply = 400, {"error": "bad request"}
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1").split("\r\n")
            method, path = head[0].split(" ")[:2]
            headers = dict(line.split(": ", 1) for line in head[1:] if ": " in line)
            length = int({k.lower(): v for k, v in headers.items()}.get("content-length", "0"))
            raw = await reader.readexactly(length) if length else b""
            handler = self.routes.get(f"{method} {path}")
            if handler is None:
                status, reply = 404, {"error": "not found"}
            else:
                status, reply = await handler(json.loads(raw) if raw else {})
        except NotImplementedError as missing:
            status, reply = 501, {"error": f"not implemented: {missing}"}
        except (ValueError, KeyError, asyncio.IncompleteReadError) as problem:
            status, reply = 400, {"error": str(problem)}
        data = json.dumps(reply).encode()
        head = f"HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n"
        writer.write(head.encode() + b"Connection: close\r\n\r\n" + data)
        await writer.drain()
        writer.close()


async def run(port: int, db: str) -> None:
    app = App(Storage(db))
    server = await asyncio.start_server(app.serve, "127.0.0.1", port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=7202)
    parser.add_argument("--db", default="./state/router-next/queue.db")
    args = parser.parse_args()
    asyncio.run(run(args.port, args.db))
