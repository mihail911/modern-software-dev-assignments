"""SQLite storage for the router: schema, id allocation and the log file."""

import hashlib
import os
import sqlite3
import time

SCHEMA = """
create table if not exists meta (
    key text primary key,
    value integer not null
);
create table if not exists bus_seen (
    producer text not null,
    ident text not null,
    primary key (producer, ident)
);
create table if not exists endpoints (
    id text primary key,
    merchant_id text not null,
    url text not null,
    secret text not null,
    event_types text not null,
    enabled integer not null,
    version integer not null,
    disabled_at integer
);
create table if not exists events (
    id text primary key,
    merchant_id text not null,
    type text not null,
    created integer not null,
    data text not null,
    routed_at integer not null
);
create table if not exists deliveries (
    id text primary key,
    ord integer not null,
    event_id text not null,
    endpoint_id text not null,
    endpoint_version integer not null,
    resend_of text,
    state text not null,
    attempts integer not null default 0,
    available_at integer not null,
    lease_id text,
    created_at integer not null,
    finished_at integer,
    last_status integer
);
create index if not exists deliveries_by_key on deliveries (event_id, endpoint_id, endpoint_version);
create index if not exists deliveries_by_state on deliveries (state, available_at);
create table if not exists leases (
    id text primary key,
    worker_id text not null,
    granted_at integer not null,
    expires_at integer not null,
    closed_at integer
);
create table if not exists attempts (
    attempt_id text primary key,
    delivery_id text not null,
    lease_id text not null,
    outcome text not null,
    status integer,
    error text,
    reported_at integer not null
);
"""


def open_db(state_dir):
    path = os.path.join(state_dir, "router")
    os.makedirs(path, exist_ok=True)
    db = sqlite3.connect(os.path.join(path, "router.db"), check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("pragma journal_mode = wal")
    db.execute("pragma synchronous = normal")
    db.executescript(SCHEMA)
    db.commit()
    return db


def next_id(db, prefix):
    """Allocate an opaque id. The counter is part of the caller's transaction."""
    db.execute(
        "insert into meta (key, value) values (?, 1) "
        "on conflict (key) do update set value = value + 1",
        (prefix,),
    )
    n = db.execute("select value from meta where key = ?", (prefix,)).fetchone()[0]
    digest = hashlib.sha256(f"switchyard:{prefix}:{n}".encode()).hexdigest()
    return f"{prefix}_{digest[:14]}"


class Clock:
    """Wall clock until operations set it (contracts/harness.md)."""

    def __init__(self):
        start = os.environ.get("SWITCHYARD_CLOCK")
        self._now = int(start) if start else None

    def set(self, now):
        self._now = int(now)

    def now(self):
        return int(time.time()) if self._now is None else self._now


class Log:
    """router.log: one text line per record."""

    def __init__(self, state_dir, clock):
        os.makedirs(os.path.join(state_dir, "logs"), exist_ok=True)
        self.path = os.path.join(state_dir, "logs", "router.log")
        self.clock = clock
        self.file = open(self.path, "a", encoding="utf-8")

    def write(self, msg, level="info", **fields):
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.clock.now()))
        parts = [stamp, f"level={level}", f"msg={msg}"]
        for key, value in fields.items():
            if value is None:
                continue
            text = str(value)
            if " " in text or '"' in text or text == "":
                text = '"' + text.replace('"', "'") + '"'
            parts.append(f"{key}={text}")
        self.file.write(" ".join(parts) + "\n")
        self.file.flush()
