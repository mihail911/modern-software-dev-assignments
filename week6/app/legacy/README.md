This is the release 1 dispatcher. It was moved here from `dispatcher/` when the services
were introduced in release 2.0 and is kept for reference.

# Switchyard dispatcher

The dispatcher takes payment events from the platform and delivers them to merchants'
HTTP endpoints. It is one Python process with a SQLite file for its state. It needs
Python 3.10 or later and nothing outside the standard library.

## Running

```
python3 -m dispatcher --db switchyard.db --port 7000
```

Run it from this directory. `--db` is created if it does not exist. The log goes to
`dispatcher.log` in the working directory (`--log` to change it), one line per thing that
happened, like `2025-03-04 10:00:00 INFO delivered dlv_9f3c_1 status=200 attempt=1`.

The process has no threads and does nothing between requests. Deliveries that are due are
sent when `POST /__admin/tick` is called, so something has to call it regularly.

## Interface

| Request | Effect |
|---|---|
| `POST /endpoints` | Registers an endpoint: `{"id", "merchant", "url", "secret", "event_types"}`. `event_types` is a list of event types, or `["*"]` for all. Posting an existing id replaces it. |
| `POST /events` | Accepts an event (below). Replies `202` with `{"id", "deliveries"}`. |
| `POST /__admin/tick` | Sends every delivery that is due. Replies `{"sent": n}`. |
| `POST /__admin/clock` | `{"now": …}` sets the clock, as whole seconds or as `"2025-03-04T10:00:00Z"`. |

An event:

```json
{"id": "evt_9f3c", "merchant": "m_acme", "event_type": "payment.captured",
 "amount": "12.50", "currency": "USD", "payment": "pay_77", "ts": "2025-03-04T10:00:00Z"}
```

`amount` is a decimal string in the currency's major unit, written with the currency's
decimal places: two for USD, EUR and GBP, none for JPY, three for KWD. Anything else is
refused with `400`. The merchant receives the event as it was accepted, with the amount as
the same string.

## Guarantees

**Delivery ids are assigned at ingest.** When an event is accepted it gets one delivery for
each of the merchant's endpoints subscribed to its type, in the order the endpoints were
registered. The ids are `dlv_<event id without evt_>_<n>`, with n counting those endpoints
from 1: `evt_9f3c` gets `dlv_9f3c_1`, `dlv_9f3c_2`. They are in the reply to `POST /events`.
Posting an event id a second time returns the same ids and creates nothing.

**At most once.** A delivery is marked `sending` before each request and the result is
stored after it. If the process stops in between, we cannot know whether the merchant got
the request. On the next start every delivery found in `sending` is closed as `delivered`
with `confirmed = 0`, and it is never sent again. We consider a duplicate payment
notification worse than a missing one, and merchants reconcile missing ones from the
nightly settlement file. Deliveries the merchant acknowledged have `confirmed = 1`.

**Five attempts.** A request that definitely failed is retried: any status outside 2xx, a
refused connection, or no answer within 5 seconds. After a failed attempt the dispatcher
waits 10 seconds, then 1 minute, 5 minutes and 30 minutes. After the fifth failed attempt
the delivery is `failed` and is left alone.

**Signature.** Each request is a `POST` of the event as JSON with two headers:

```
X-Switchyard-Signature: sha256=<hex HMAC-SHA256 of the raw body, keyed with the endpoint secret>
X-Switchyard-Delivery: <delivery id>
```

A merchant checks the signature over the bytes received, before parsing them. The same
delivery id is sent on every attempt of a delivery.

## Tests

From the repository root:

```
python3 -m unittest discover -s legacy/tests -v
```

or `cd legacy && python3 -m unittest -v`. The tests start a local HTTP server in place of
the merchant and use a fake clock, so they take a few seconds.
