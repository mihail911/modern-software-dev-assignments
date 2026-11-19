# router-next

Jun, 2026-05-21

This is the future router, planned for release 2.5. It is a rewrite of `services/router`, not a branch of it, and it shares no code with it. `bin/up` does not start it and no worker talks to it yet.

## Why

The live router hands a delivery to a worker under a lease. Since 2.2.3 a lease lasts 120 seconds (it was 60 before INC-1907). One number is doing two jobs: it has to be longer than the slowest thing a healthy worker does, and it is also how long a delivery sits untouched after a worker dies. Raising it fixed the first and made the second worse. A worker that is drained with a delivery in hand now holds it for up to two minutes before anyone else can try.

## The plan

1. **Visibility timeouts instead of leases.** A worker receives a delivery and it becomes invisible to other workers for 30 seconds. While the worker is busy it heartbeats, and each heartbeat hides the delivery for another 30 seconds, up to 300 seconds in total. A worker that dies stops heartbeating and the delivery is back within 30 seconds. A timeout that fires is an attempt whose result was never recorded, so it counts as `unknown` and follows the retry schedule, the same as today.
2. **Nack.** A worker that knows it cannot finish (it is shutting down, it has not started the request) gives the delivery back and it is due at once. No attempt is counted. Today a worker can only close its lease, and a worker that is stopped before it gets that far leaves the delivery to the lease's expiry.
3. **Receipts.** Every receive returns a new receipt. Heartbeat, ack and nack quote it, and a receipt that is no longer current gets a `409`. A late call from a worker that lost its delivery cannot touch whoever holds it now.
4. **Delivery ids from a hash.** `dlv_` plus the first 14 hex digits of SHA-256 over the event id and the endpoint id. One delivery per event and endpoint stops being something the router has to check and becomes something it cannot get wrong. See the open question on resends.
5. **asyncio.** One event loop, standard library only, no lock around the queue.

## Protocol

| Request | Body | Reply |
|---|---|---|
| `POST /queue/receive` | `{"worker_id"}` | `{"receipt", "delivery_id", "attempt_no", "visible_until"}`, or `{"receipt": null}` |
| `POST /queue/heartbeat` | `{"receipt"}` | `{"visible_until"}` or `409` |
| `POST /queue/ack` | `{"receipt", "outcome", "status"}` | `{"acked": true}` or `409` |
| `POST /queue/nack` | `{"receipt"}` | `{"returned": true}` or `409` |

Retry schedule and dead-lettering are unchanged: 30 s, 2 min, 10 min, 1 h, 6 h, dead after six counted attempts.

## Done

- The queue: receive, heartbeat, ack, nack, timeouts, the retry schedule, dead-lettering, paused endpoints. Tested, except pausing.
- Routing of version-3 `event.accepted` messages.
- `endpoint.changed`: storing the endpoint, disabling it.

## Not done

- **Recovery after a router restart.** Deliveries that were invisible when the router stopped are not handled; `Storage.recover_inflight` is a stub and the app will not start if there are any. The two options are written up there.
- Acks are not written to SQLite yet, only held in memory.
- `/queue/receive` does not return the event or the endpoint's URL and secret.
- Catch-up when an endpoint is re-enabled (`501`).
- `resend.requested` (`501`). A resend is a second delivery of the same event to the same endpoint, and the hash gives the same id. Probably a salt; not decided.
- Bus versions 1 and 2.
- The operations interface beyond `health` and `tick`: no clock, no state dump.
- `router.log`.
- The worker. Nothing on the Go side has been touched. It needs a heartbeat loop and to call nack on shutdown.
- Migration of a live queue from `router.db`.

## Running

```
cd services/router-next
python3 -m unittest -v
python3 -m router_next.app --port 7202 --db ./state/router-next/queue.db
```

Run it as a module. `python3 router_next/app.py` puts the package directory on the path and `queue.py` then hides the standard library's `queue`.
