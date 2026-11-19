"""Ingest, the send loop, retry scheduling and recovery at start."""

import calendar
import http.client
import json
import re
import time
from urllib.parse import urlsplit

from . import signing
from .store import Store

MAX_ATTEMPTS = 5
RETRY_DELAYS = (10, 60, 300, 1800)  # seconds to wait after failed attempt 1, 2, 3, 4
SEND_TIMEOUT = 5.0
CURRENCY_DECIMALS = {"USD": 2, "EUR": 2, "GBP": 2, "JPY": 0, "KWD": 3}
EVENT_FIELDS = ("id", "merchant", "event_type", "amount", "currency", "payment", "ts")
ID_PREFIXES = {"id": "evt_", "merchant": "m_", "payment": "pay_"}
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class Invalid(Exception):
    """A request the dispatcher refuses. The message is returned to the caller."""


class Clock:
    """The system clock, in whole seconds. set() pins it to a given time."""

    def __init__(self):
        self.pinned = None

    def now(self):
        return int(time.time()) if self.pinned is None else self.pinned

    def set(self, now):
        self.pinned = int(now)


class Log:
    """Appends one line per thing that happened: "2025-03-04 10:00:00 INFO delivered dlv_... status=200"."""

    def __init__(self, path, clock):
        self.clock = clock
        self.file = open(path, "a", encoding="utf-8")

    def write(self, level, message):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.clock.now()))
        self.file.write("%s %s %s\n" % (stamp, level, message))
        self.file.flush()


def parse_time(value):
    """Seconds since the epoch, from an integer or a string like "2025-03-04T10:00:00Z"."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    try:
        return calendar.timegm(time.strptime(value, TIME_FORMAT))
    except (TypeError, ValueError):
        raise Invalid("not a time: %r" % (value,))


def check_amount(amount, currency):
    """Amounts are decimal strings in the major unit, written with the currency's decimal places."""
    decimals = CURRENCY_DECIMALS.get(currency)
    if decimals is None:
        raise Invalid("unsupported currency: %r" % (currency,))
    pattern = "[0-9]+" if decimals == 0 else "[0-9]+\\.[0-9]{%d}" % decimals
    if not isinstance(amount, str) or not re.fullmatch(pattern, amount):
        raise Invalid("amount must be a string with %d decimal places for %s" % (decimals, currency))


class Dispatcher:
    def __init__(self, db_path, clock=None, log_path="dispatcher.log", timeout=SEND_TIMEOUT):
        self.clock = clock or Clock()
        self.timeout = timeout
        self.store = Store(db_path)
        self.log = Log(log_path, self.clock)
        self.recover()

    def close(self):
        self.store.close()
        self.log.file.close()

    def recover(self):
        """Close deliveries that were being sent when the process last stopped.

        The request may or may not have reached the merchant. Sending it again
        could notify them of the same payment twice, so it is closed as
        delivered with confirmed = 0 and left alone.
        """
        now = self.clock.now()
        for row in self.store.sending():
            self.store.finish(row["id"], "delivered", now, confirmed=0)
            self.log.write("WARN", "closed %s unconfirmed, it was sending when the process stopped" % row["id"])

    def add_endpoint(self, endpoint):
        for field in ("id", "merchant", "url", "secret"):
            if not isinstance(endpoint.get(field), str) or not endpoint[field]:
                raise Invalid("%s is required" % field)
        if urlsplit(endpoint["url"]).scheme not in ("http", "https"):
            raise Invalid("url must be http or https")
        types = endpoint.get("event_types")
        if not isinstance(types, list) or not types or not all(isinstance(t, str) for t in types):
            raise Invalid('event_types must be a list of event types, or ["*"]')
        self.store.put_endpoint(endpoint, self.clock.now())
        self.log.write("INFO", "endpoint %s merchant=%s url=%s" % (endpoint["id"], endpoint["merchant"], endpoint["url"]))

    def ingest(self, event):
        """Accept an event and create one delivery per subscribed endpoint. Returns the delivery ids.

        Nothing is sent here. The deliveries are due at once and go out on the next tick.
        """
        for field in EVENT_FIELDS:
            if not isinstance(event.get(field), str) or not event[field]:
                raise Invalid("%s is required" % field)
        for field, prefix in ID_PREFIXES.items():
            if not event[field].startswith(prefix) or event[field] == prefix:
                raise Invalid("%s must start with %s" % (field, prefix))
        check_amount(event["amount"], event["currency"])
        parse_time(event["ts"])
        if self.store.has_event(event["id"]):
            return self.store.delivery_ids(event["id"])
        body = json.dumps({field: event[field] for field in EVENT_FIELDS}, separators=(",", ":"))
        endpoints = self.store.endpoints_for(event["merchant"], event["event_type"])
        suffix = event["id"][len("evt_"):]
        deliveries = [("dlv_%s_%d" % (suffix, n), endpoint["id"]) for n, endpoint in enumerate(endpoints, 1)]
        self.store.add_event(event, body, deliveries, self.clock.now())
        self.log.write("INFO", "accepted %s type=%s deliveries=%d" % (event["id"], event["event_type"], len(deliveries)))
        return [delivery_id for delivery_id, _ in deliveries]

    def tick(self):
        """Send every delivery that is due. Returns the number of requests made."""
        due = self.store.due(self.clock.now())
        for delivery in due:
            self.attempt(delivery)
        return len(due)

    def attempt(self, delivery):
        number = delivery["attempts"] + 1
        # Committed before the request goes out. If the process stops between
        # here and the result being stored, recover() finds the row at start.
        self.store.mark_sending(delivery["id"], number)
        body = delivery["body"].encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            signing.SIGNATURE_HEADER: signing.sign(delivery["secret"], body),
            signing.DELIVERY_HEADER: delivery["id"],
        }
        status, error = self.post(delivery["url"], body, headers)
        now = self.clock.now()
        result = "status=%d" % status if status is not None else "error=%s" % error
        if status is not None and 200 <= status < 300:
            self.store.finish(delivery["id"], "delivered", now, confirmed=1, last_status=status)
            self.log.write("INFO", "delivered %s %s attempt=%d" % (delivery["id"], result, number))
        elif number >= MAX_ATTEMPTS:
            self.store.finish(delivery["id"], "failed", now, last_status=status, error=error)
            self.log.write("ERROR", "failed %s %s attempt=%d, giving up" % (delivery["id"], result, number))
        else:
            delay = RETRY_DELAYS[number - 1]
            self.store.retry(delivery["id"], now + delay, status, error)
            self.log.write("WARN", "retry %s %s attempt=%d next_in=%ds" % (delivery["id"], result, number, delay))

    def post(self, url, body, headers):
        """One POST on its own connection. Returns (status, None), or (None, reason) without a response."""
        try:
            parts = urlsplit(url)
            if parts.scheme == "https":
                connection = http.client.HTTPSConnection(parts.netloc, timeout=self.timeout)
            else:
                connection = http.client.HTTPConnection(parts.netloc, timeout=self.timeout)
            try:
                path = (parts.path or "/") + ("?" + parts.query if parts.query else "")
                connection.request("POST", path, body=body, headers=headers)
                return connection.getresponse().status, None
            finally:
                connection.close()
        except (OSError, http.client.HTTPException, ValueError) as problem:
            return None, type(problem).__name__
