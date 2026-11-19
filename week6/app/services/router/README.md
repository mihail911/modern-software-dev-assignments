# router

Decides which endpoints an event goes to, creates the deliveries, and owns the queue the workers lease from. Python, standard library only, one SQLite file (`state/router/router.db`).

## What it owns

- **Delivery identity.** `router` assigns delivery ids (`dlv_…`): one delivery per event and endpoint. Nothing else creates a delivery, except a manual resend.
- **The queue.** `POST /queue/lease`, `/queue/report`, `/queue/close`.
- **Retry timing.** A failed or unknown attempt puts the delivery back in the queue after 30 seconds, then 2 minutes, 10 minutes, 1 hour and 6 hours. After six such attempts the delivery is dead.

## Leases

A lease holds one delivery and lasts 60 seconds. A lease that is not closed in time expires and its delivery goes back to the queue.

## Delivery is at least once

An attempt that a worker started and never recorded a result for is reported as `unknown`. It counts as an attempt and the delivery is retried, under the same delivery id. Merchants de-duplicate on the delivery id.

## Bus messages

Version 3, from `api`. Versions 1 and 2 are still accepted; see `contracts/messages/`.

## Tests

```
make test
```
