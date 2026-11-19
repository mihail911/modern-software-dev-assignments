"""The delivery queue: lease, report, close, expire."""

import json

from store import next_id

LEASE_SECONDS = 120
MAX_ATTEMPTS = 6
RETRY_AFTER = [30, 120, 600, 3600, 21600]
OUTCOMES = {"delivered", "failed", "unknown"}


def lease(db, log, now, worker_id):
    row = db.execute(
        "select d.id, d.attempts, d.event_id, d.endpoint_id, e.type, e.created, e.data, "
        "       p.url, p.secret "
        "from deliveries d "
        "join events e on e.id = d.event_id "
        "join endpoints p on p.id = d.endpoint_id "
        "where d.state = 'queued' and d.available_at <= ? and p.enabled = 1 "
        "order by d.available_at, d.ord limit 1",
        (now,),
    ).fetchone()
    if row is None:
        return {"lease_id": None, "delivery": None}
    lease_id = next_id(db, "lse")
    expires_at = now + LEASE_SECONDS
    db.execute(
        "insert into leases (id, worker_id, granted_at, expires_at) values (?, ?, ?, ?)",
        (lease_id, worker_id, now, expires_at),
    )
    db.execute("update deliveries set state = 'leased', lease_id = ? where id = ?", (lease_id, row["id"]))
    delivery = {
        "delivery_id": row["id"],
        "attempt_no": row["attempts"] + 1,
        "event": {
            "id": row["event_id"],
            "type": row["type"],
            "created": row["created"],
            "data": json.loads(row["data"]),
        },
        "endpoint": {"id": row["endpoint_id"], "url": row["url"], "secret": row["secret"]},
    }
    log.write("lease.granted", lease=lease_id, worker=worker_id, delivery=row["id"], attempt_no=delivery["attempt_no"],
              expires=expires_at)
    return {"lease_id": lease_id, "expires_at": expires_at, "delivery": delivery}


def report(db, log, now, lease_id, results):
    ignored = []
    for result in results:
        delivery_id = result.get("delivery_id")
        attempt_id = result.get("attempt_id")
        outcome = result.get("outcome")
        if not isinstance(attempt_id, str) or outcome not in OUTCOMES:
            ignored.append(delivery_id)
            continue
        row = db.execute(
            "select attempts from deliveries where id = ? and lease_id = ? and state = 'leased'",
            (delivery_id, lease_id),
        ).fetchone()
        if row is None:
            if not db.execute("select 1 from attempts where attempt_id = ?", (attempt_id,)).fetchone():
                ignored.append(delivery_id)
                log.write("report.ignored", level="warn", delivery=delivery_id, lease=lease_id, attempt=attempt_id)
            continue
        # Idempotent by attempt_id: the first report of an attempt is the one that stands.
        db.execute(
            "insert or ignore into attempts (attempt_id, delivery_id, lease_id, outcome, status, error, reported_at) "
            "values (?, ?, ?, ?, ?, ?, ?)",
            (attempt_id, delivery_id, lease_id, outcome, result.get("status"), result.get("error"), now),
        )
        outcome, status = db.execute(
            "select outcome, status from attempts where attempt_id = ?", (attempt_id,)
        ).fetchone()
        answered = result.get("status")
        if outcome == "delivered":
            db.execute(
                "update deliveries set state = 'delivered', lease_id = null, finished_at = ?, last_status = ? "
                "where id = ?",
                (now, answered, delivery_id),
            )
            log.write("delivery.delivered", delivery=delivery_id, status=status)
            continue
        attempts = row["attempts"] + 1
        if attempts >= MAX_ATTEMPTS:
            db.execute(
                "update deliveries set state = 'dead', attempts = ?, lease_id = null, finished_at = ?, "
                "last_status = ? where id = ?",
                (attempts, now, answered, delivery_id),
            )
            log.write("delivery.dead", delivery=delivery_id, attempt=attempt_id, attempts=attempts, outcome=outcome)
            continue
        retry_at = now + RETRY_AFTER[attempts - 1]
        db.execute(
            "update deliveries set state = 'queued', attempts = ?, lease_id = null, available_at = ?, "
            "last_status = ? where id = ?",
            (attempts, retry_at, answered, delivery_id),
        )
        log.write(
            "delivery.retry",
            delivery=delivery_id,
            attempt=attempt_id,
            attempts=attempts,
            outcome=outcome,
            status=status,
            retry_at=retry_at,
        )
    return {"ignored": ignored}


def close(db, log, now, lease_id, reason="closed"):
    row = db.execute("select closed_at from leases where id = ?", (lease_id,)).fetchone()
    if row is None or row["closed_at"] is not None:
        return 0
    returned = db.execute(
        "select id from deliveries where lease_id = ? and state = 'leased' order by ord", (lease_id,)
    ).fetchall()
    for delivery in returned:
        db.execute(
            "update deliveries set state = 'queued', lease_id = null, available_at = ? where id = ?",
            (now, delivery["id"]),
        )
        log.write("delivery.returned", delivery=delivery["id"], lease=lease_id)
    db.execute("update leases set closed_at = ? where id = ?", (now, lease_id))
    log.write("lease." + reason, lease=lease_id, returned=len(returned))
    return 1


def expire(db, log, now):
    overdue = db.execute(
        "select id from leases where closed_at is null and expires_at <= ? order by expires_at, id", (now,)
    ).fetchall()
    for row in overdue:
        close(db, log, now, row["id"], reason="expired")
    return len(overdue)


def next_wake(db):
    """The earliest time at which something becomes due. None when nothing is waiting."""
    due = db.execute(
        "select min(d.available_at) from deliveries d join endpoints p on p.id = d.endpoint_id "
        "where d.state = 'queued' and p.enabled = 1"
    ).fetchone()[0]
    expiry = db.execute("select min(expires_at) from leases where closed_at is null").fetchone()[0]
    times = [t for t in (due, expiry) if t is not None]
    return min(times) if times else None
