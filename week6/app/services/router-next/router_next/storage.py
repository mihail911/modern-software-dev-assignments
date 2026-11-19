"""SQLite persistence. The queue lives in memory (queue.py); this writes it through."""

import json
import os
import sqlite3
from typing import List, Optional

from .models import INVISIBLE, Delivery, Endpoint, Event

SCHEMA = """
create table if not exists endpoints (id text primary key, merchant_id text not null, body text not null);
create table if not exists events (
    id text primary key, merchant_id text not null, body text not null, routed_at integer not null
);
create table if not exists deliveries (
    id text primary key, seq integer not null, state text not null, visible_at integer not null, body text not null
);
create index if not exists deliveries_due on deliveries (state, visible_at);
"""


class Storage:
    def __init__(self, path: str):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.db.commit()

    def save_endpoint(self, endpoint: Endpoint) -> None:
        self.db.execute(
            "insert into endpoints (id, merchant_id, body) values (?, ?, ?) "
            "on conflict (id) do update set body = excluded.body",
            (endpoint.id, endpoint.merchant_id, json.dumps(endpoint.__dict__)),
        )
        self.db.commit()

    def endpoints(self, merchant_id: Optional[str] = None) -> List[Endpoint]:
        rows = self.db.execute("select body from endpoints order by id").fetchall()
        found = [Endpoint(**json.loads(body)) for (body,) in rows]
        return [e for e in found if merchant_id is None or e.merchant_id == merchant_id]

    def save_event(self, event: Event, now: int) -> bool:
        """False if the event was already stored."""
        cursor = self.db.execute(
            "insert or ignore into events (id, merchant_id, body, routed_at) values (?, ?, ?, ?)",
            (event.id, event.merchant_id, json.dumps(event.__dict__), now),
        )
        self.db.commit()
        return cursor.rowcount == 1

    def save_delivery(self, d: Delivery) -> None:
        self.db.execute(
            "insert into deliveries (id, seq, state, visible_at, body) "
            "values (?, (select coalesce(max(seq), 0) + 1 from deliveries), ?, ?, ?) "
            "on conflict (id) do update set state = excluded.state, visible_at = excluded.visible_at, "
            "body = excluded.body",
            (d.id, d.state, d.visible_at, json.dumps(d.__dict__)),
        )
        self.db.commit()

    def load_deliveries(self) -> List[Delivery]:
        rows = self.db.execute("select body from deliveries order by seq").fetchall()
        return [Delivery(**json.loads(body)) for (body,) in rows]

    def recover_inflight(self, now: int) -> List[Delivery]:
        """Deliveries that were invisible when the router stopped.

        TODO(jun): not decided. Either keep receipt and deadline as stored, so a
        worker that is still going can ack (needs the receipt counter persisted
        too), or drop every receipt and make them visible at `now`. Until then
        the app refuses to start on a database that has any.
        """
        inflight = [d for d in self.load_deliveries() if d.state == INVISIBLE]
        if inflight:
            raise NotImplementedError(f"{len(inflight)} deliveries were in flight at shutdown")
        return inflight
