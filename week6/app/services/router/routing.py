"""Applying bus messages: events become deliveries, endpoint changes are recorded."""

import json
import os

from store import next_id

CATCH_UP_SECONDS = 24 * 3600

# INC-2188 workaround (2.3.1). With it on, an endpoint change does not create deliveries
# at the endpoint's new version, so a delivery that was queued keeps going under the
# version it was created for. Leave it on until deliveries stop being keyed by version.
INC_2188_WORKAROUND = os.environ.get("SWITCHYARD_INC2188_WORKAROUND", "on") != "off"


def subscribed(event_types, type_):
    return "*" in event_types or type_ in event_types


def create_delivery(db, log, now, event_id, endpoint_id, resend_of=None):
    version = db.execute("select version from endpoints where id = ?", (endpoint_id,)).fetchone()[0]
    delivery_id = next_id(db, "dlv")
    ord_ = db.execute("select value from meta where key = 'dlv'").fetchone()[0]
    db.execute(
        "insert into deliveries (id, ord, event_id, endpoint_id, endpoint_version, resend_of, state, "
        "available_at, created_at) values (?, ?, ?, ?, ?, ?, 'queued', ?, ?)",
        (delivery_id, ord_, event_id, endpoint_id, version, resend_of, now, now),
    )
    log.write(
        "delivery.created", delivery=delivery_id, event=event_id, endpoint=endpoint_id, version=version,
        resend_of=resend_of,
    )
    return delivery_id


def delivery_at(db, event_id, endpoint_id, version):
    """A delivery is one event to one version of an endpoint (2.3)."""
    return db.execute(
        "select id from deliveries where event_id = ? and endpoint_id = ? and endpoint_version = ? "
        "and resend_of is null",
        (event_id, endpoint_id, version),
    ).fetchone()


def first_delivery(db, event_id, endpoint_id):
    return db.execute(
        "select id from deliveries where event_id = ? and endpoint_id = ? and resend_of is null order by ord",
        (event_id, endpoint_id),
    ).fetchone()


def follow_version(db, log, now, endpoint_id, version):
    """Give every unfinished event of the endpoint a delivery at its current version."""
    unfinished = db.execute(
        "select distinct event_id from deliveries where endpoint_id = ? and state in ('queued', 'leased') "
        "and resend_of is null order by ord",
        (endpoint_id,),
    ).fetchall()
    for row in unfinished:
        if not delivery_at(db, row["event_id"], endpoint_id, version):
            create_delivery(db, log, now, row["event_id"], endpoint_id)


def route_event(db, log, now, event):
    if db.execute("select 1 from events where id = ?", (event["id"],)).fetchone():
        log.write("event.duplicate", event=event["id"])
        return
    db.execute(
        "insert into events (id, merchant_id, type, created, data, routed_at) values (?, ?, ?, ?, ?, ?)",
        (event["id"], event["merchant_id"], event["type"], event["created"], json.dumps(event["data"]), now),
    )
    log.write(
        "event.routed",
        event=event["id"],
        merchant=event["merchant_id"],
        type=event["type"],
        amount_minor=event["data"]["amount_minor"],
        currency=event["data"]["currency"],
    )
    endpoints = db.execute(
        "select id, event_types from endpoints where merchant_id = ? and enabled = 1 order by id",
        (event["merchant_id"],),
    ).fetchall()
    for ep in endpoints:
        if subscribed(json.loads(ep["event_types"]), event["type"]):
            create_delivery(db, log, now, event["id"], ep["id"])


def catch_up(db, log, now, endpoint, since):
    """Deliver what the endpoint missed while it was disabled."""
    floor = max(since, now - CATCH_UP_SECONDS)
    missed = db.execute(
        "select id, type from events where merchant_id = ? and routed_at >= ? order by routed_at, rowid",
        (endpoint["merchant_id"], floor),
    ).fetchall()
    for event in missed:
        if not subscribed(endpoint["event_types"], event["type"]):
            continue
        if first_delivery(db, event["id"], endpoint["id"]):
            continue
        create_delivery(db, log, now, event["id"], endpoint["id"])


def apply_endpoint(db, log, now, endpoint):
    old = db.execute("select * from endpoints where id = ?", (endpoint["id"],)).fetchone()
    if old is not None and endpoint["version"] <= old["version"]:
        log.write("endpoint.stale", endpoint=endpoint["id"], version=endpoint["version"])
        return
    was_enabled = bool(old["enabled"]) if old is not None else None
    disabled_at = old["disabled_at"] if old is not None else None
    if not endpoint["enabled"] and was_enabled is not False:
        disabled_at = now
    db.execute(
        "insert into endpoints (id, merchant_id, url, secret, event_types, enabled, version, disabled_at) "
        "values (?, ?, ?, ?, ?, ?, ?, ?) "
        "on conflict (id) do update set url = excluded.url, secret = excluded.secret, "
        "event_types = excluded.event_types, enabled = excluded.enabled, version = excluded.version, "
        "disabled_at = excluded.disabled_at",
        (
            endpoint["id"],
            endpoint["merchant_id"],
            endpoint["url"],
            endpoint["secret"],
            json.dumps(endpoint["event_types"]),
            int(endpoint["enabled"]),
            endpoint["version"],
            None if endpoint["enabled"] else disabled_at,
        ),
    )
    log.write(
        "endpoint.changed",
        endpoint=endpoint["id"],
        merchant=endpoint["merchant_id"],
        version=endpoint["version"],
        enabled=str(endpoint["enabled"]).lower(),
    )
    if old is not None and not INC_2188_WORKAROUND:
        follow_version(db, log, now, endpoint["id"], endpoint["version"])
    if endpoint["enabled"] and was_enabled is False and disabled_at is not None:
        catch_up(db, log, now, endpoint, disabled_at)


def apply_resend(db, log, now, body):
    original = first_delivery(db, body["event_id"], body["endpoint_id"])
    if original is None:
        log.write("resend.skipped", level="warn", event=body["event_id"], endpoint=body["endpoint_id"])
        return
    create_delivery(db, log, now, body["event_id"], body["endpoint_id"], resend_of=original["id"])


def apply(db, log, now, kind, body):
    if kind == "event":
        route_event(db, log, now, body)
    elif kind == "endpoint":
        apply_endpoint(db, log, now, body)
    else:
        apply_resend(db, log, now, body)
