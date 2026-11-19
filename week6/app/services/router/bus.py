"""Bus message versions 1 to 3, reduced to one internal shape.

See contracts/messages/README.md. `parse` returns (producer, ident, kind, body)
or raises Malformed. `kind` is "event", "endpoint" or "resend".
"""

import calendar
import re
import time

EVENT_TYPES = {"payment.captured", "payment.failed", "refund.created", "payout.paid"}
MINOR_DIGITS = {"USD": 2, "EUR": 2, "GBP": 2, "JPY": 0, "KWD": 3}
DECIMAL = re.compile(r"^[0-9]+(\.[0-9]+)?$")


class Malformed(Exception):
    pass


def need(obj, key, kind):
    if not isinstance(obj, dict) or key not in obj:
        raise Malformed(f"missing {key}")
    value = obj[key]
    if kind is int and isinstance(value, bool):
        raise Malformed(f"{key} must be an integer")
    if not isinstance(value, kind):
        raise Malformed(f"{key} has the wrong type")
    return value


def to_minor(amount, currency):
    """Version 1 carried decimal strings in the major unit."""
    if not isinstance(amount, str) or not DECIMAL.match(amount):
        raise Malformed("amount is not a decimal string")
    digits = MINOR_DIGITS[currency]
    whole, _, fraction = amount.partition(".")
    if "." in amount and digits == 0:
        raise Malformed("currency has no minor unit")
    if len(fraction) > digits:
        raise Malformed("too many fractional digits")
    return int(whole + fraction.ljust(digits, "0"))


def event_body(event_id, merchant, type_, created, payment_id, amount_minor, currency):
    if type_ not in EVENT_TYPES:
        raise Malformed("unknown event type")
    if currency not in MINOR_DIGITS:
        raise Malformed("unknown currency")
    if amount_minor < 0:
        raise Malformed("negative amount")
    return {
        "id": event_id,
        "merchant_id": merchant,
        "type": type_,
        "created": created,
        "data": {
            "object": "payment",
            "payment_id": payment_id,
            "amount_minor": amount_minor,
            "currency": currency,
        },
    }


def endpoint_body(endpoint_id, merchant, url, secret, types, enabled, version):
    if not isinstance(types, list) or not all(isinstance(t, str) for t in types):
        raise Malformed("event types must be a list of strings")
    for type_ in types:
        if type_ != "*" and type_ not in EVENT_TYPES:
            raise Malformed("unknown event type")
    return {
        "id": endpoint_id,
        "merchant_id": merchant,
        "url": url,
        "secret": secret,
        "event_types": types,
        "enabled": bool(enabled),
        "version": version,
    }


def parse_v3(msg):
    producer = need(msg, "producer", str)
    seq = need(msg, "seq", int)
    type_ = need(msg, "type", str)
    data = need(msg, "data", dict)
    if type_ == "event.accepted":
        event = need(data, "event", dict)
        inner = need(event, "data", dict)
        currency = need(inner, "currency", str)
        body = event_body(
            need(event, "id", str),
            need(event, "merchant_id", str),
            need(event, "type", str),
            need(event, "created", int),
            need(inner, "payment_id", str),
            need(inner, "amount_minor", int),
            currency,
        )
        return producer, str(seq), "event", body
    if type_ == "endpoint.changed":
        ep = need(data, "endpoint", dict)
        body = endpoint_body(
            need(ep, "id", str),
            need(ep, "merchant_id", str),
            need(ep, "url", str),
            need(ep, "secret", str),
            need(ep, "event_types", list),
            need(ep, "enabled", bool),
            need(ep, "version", int),
        )
        return producer, str(seq), "endpoint", body
    if type_ == "resend.requested":
        body = {"event_id": need(data, "event_id", str), "endpoint_id": need(data, "endpoint_id", str)}
        return producer, str(seq), "resend", body
    raise Malformed("unknown message type")


def parse_v2(msg):
    seq = need(msg, "seq", int)
    kind = need(msg, "kind", str)
    if kind == "event":
        event = need(msg, "event", dict)
        body = event_body(
            need(event, "id", str),
            need(event, "merchant", str),
            need(event, "type", str),
            need(event, "created", int),
            need(event, "payment_id", str),
            need(event, "amount_minor", int),
            need(event, "currency", str),
        )
        return "api.v2", str(seq), "event", body
    if kind == "endpoint":
        ep = need(msg, "endpoint", dict)
        # Version 2 had no endpoint versions. The sequence number orders them.
        body = endpoint_body(
            need(ep, "id", str),
            need(ep, "merchant", str),
            need(ep, "url", str),
            need(ep, "secret", str),
            need(ep, "types", list),
            need(ep, "active", bool),
            seq,
        )
        return "api.v2", str(seq), "endpoint", body
    raise Malformed("unknown kind")


def parse_v1(msg):
    event_id = need(msg, "id", str)
    currency = need(msg, "currency", str)
    if currency not in MINOR_DIGITS:
        raise Malformed("unknown currency")
    stamp = need(msg, "ts", str)
    try:
        created = calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        raise Malformed("bad timestamp") from None
    body = event_body(
        event_id,
        need(msg, "merchant", str),
        need(msg, "event_type", str),
        created,
        need(msg, "payment", str),
        to_minor(msg.get("amount"), currency),
        currency,
    )
    return "v1", event_id, "event", body


def parse(msg):
    if not isinstance(msg, dict):
        raise Malformed("not an object")
    version = msg.get("v")
    if version == 3:
        return parse_v3(msg)
    if version == 2:
        return parse_v2(msg)
    if version is None and "event_type" in msg:
        return parse_v1(msg)
    raise Malformed("unknown version")
