"""SQLite storage for the dispatcher: schema and queries."""

import json
import sqlite3

SCHEMA = """
create table if not exists endpoints (
    id text primary key,
    merchant text not null,
    url text not null,
    secret text not null,
    event_types text not null,
    created_at integer not null
);
create table if not exists events (
    id text primary key,
    merchant text not null,
    event_type text not null,
    body text not null,
    received_at integer not null
);
create table if not exists deliveries (
    id text primary key,
    event_id text not null,
    endpoint_id text not null,
    status text not null,
    attempts integer not null default 0,
    next_attempt_at integer,
    confirmed integer,
    last_status integer,
    last_error text,
    created_at integer not null,
    finished_at integer
);
create index if not exists deliveries_due on deliveries (status, next_attempt_at);
"""

# deliveries.status is one of: pending, sending, delivered, failed.
# deliveries.confirmed is 1 when the merchant answered 2xx, 0 when the delivery
# was closed at start without knowing the answer, null otherwise.


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self):
        self.db.close()

    def put_endpoint(self, endpoint, now):
        self.db.execute(
            "insert into endpoints (id, merchant, url, secret, event_types, created_at)"
            " values (?, ?, ?, ?, ?, ?)"
            " on conflict (id) do update set merchant = excluded.merchant, url = excluded.url,"
            " secret = excluded.secret, event_types = excluded.event_types",
            (endpoint["id"], endpoint["merchant"], endpoint["url"], endpoint["secret"],
             json.dumps(endpoint["event_types"]), now),
        )
        self.db.commit()

    def endpoints_for(self, merchant, event_type):
        """The merchant's endpoints subscribed to event_type, in the order they were registered."""
        rows = self.db.execute(
            "select * from endpoints where merchant = ? order by rowid", (merchant,)
        ).fetchall()
        matching = []
        for row in rows:
            types = json.loads(row["event_types"])
            if "*" in types or event_type in types:
                matching.append(row)
        return matching

    def has_event(self, event_id):
        return self.db.execute("select 1 from events where id = ?", (event_id,)).fetchone() is not None

    def add_event(self, event, body, deliveries, now):
        """Store an event and its deliveries, a list of (delivery id, endpoint id), together."""
        with self.db:
            self.db.execute(
                "insert into events (id, merchant, event_type, body, received_at) values (?, ?, ?, ?, ?)",
                (event["id"], event["merchant"], event["event_type"], body, now),
            )
            for delivery_id, endpoint_id in deliveries:
                self.db.execute(
                    "insert into deliveries (id, event_id, endpoint_id, status, next_attempt_at, created_at)"
                    " values (?, ?, ?, 'pending', ?, ?)",
                    (delivery_id, event["id"], endpoint_id, now, now),
                )

    def delivery_ids(self, event_id):
        rows = self.db.execute("select id from deliveries where event_id = ? order by rowid", (event_id,))
        return [row["id"] for row in rows]

    def get_delivery(self, delivery_id):
        row = self.db.execute("select * from deliveries where id = ?", (delivery_id,)).fetchone()
        return dict(row) if row else None

    def due(self, now):
        """Pending deliveries whose time has come, with what is needed to send them."""
        return self.db.execute(
            "select d.id, d.attempts, e.body, p.url, p.secret"
            " from deliveries d join events e on e.id = d.event_id join endpoints p on p.id = d.endpoint_id"
            " where d.status = 'pending' and d.next_attempt_at <= ?"
            " order by d.next_attempt_at, d.rowid",
            (now,),
        ).fetchall()

    def sending(self):
        return self.db.execute("select id from deliveries where status = 'sending' order by rowid").fetchall()

    def mark_sending(self, delivery_id, attempts):
        self.db.execute(
            "update deliveries set status = 'sending', attempts = ?, next_attempt_at = null where id = ?",
            (attempts, delivery_id),
        )
        self.db.commit()

    def retry(self, delivery_id, next_attempt_at, status, error):
        self.db.execute(
            "update deliveries set status = 'pending', next_attempt_at = ?, last_status = ?, last_error = ?"
            " where id = ?",
            (next_attempt_at, status, error, delivery_id),
        )
        self.db.commit()

    def finish(self, delivery_id, status, now, confirmed=None, last_status=None, error=None):
        """Close a delivery as 'delivered' or 'failed'. It is not picked up again."""
        self.db.execute(
            "update deliveries set status = ?, confirmed = ?, last_status = ?, last_error = ?,"
            " next_attempt_at = null, finished_at = ? where id = ?",
            (status, confirmed, last_status, error, now, delivery_id),
        )
        self.db.commit()
