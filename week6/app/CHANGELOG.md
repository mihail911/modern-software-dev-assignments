# Changelog

Dates are the day a release finished rolling out to production.

## 2.3.1 (2026-08-14)

- router: workaround for INC-2188 behind a flag, `SWITCHYARD_INC2188_WORKAROUND`, on unless set to `off`.

## 2.3.0 (2026-07-08)

- api, router: endpoint versions.
- api: idempotency keys, per merchant (ADR-0013).
- worker: connections to merchants are no longer reused.
- worker: attempt ids are computed from the worker id and the journal record number (ADR-0011). `bin/attempt-id` looks one up.

## 2.2.3 (2026-03-19)

- router: leases last 120 seconds, up from 60. Fixes INC-1907.
- router: a report for a delivery that is not held under the reporting lease is ignored and logged as `report.ignored`.

## 2.2.0 (2026-02-03)

- Bus messages version 3. Versions 1 and 2 are still accepted.
- router assigns delivery ids: one delivery per event and endpoint.
- Delivery is at least once: an attempt with no recorded outcome is `unknown`, counts as an attempt and is retried.
- A sixth attempt, six hours after the fifth. A delivery that fails six times is `dead`.
- `POST /v1/events/{id}/resend`.

## 2.1.0 (2025-11-12)

- Endpoints can be disabled and re-enabled. Re-enabling delivers what the endpoint missed in the last 24 hours.

## 2.0.0 (2025-09-02)

- The dispatcher is split into `api`, `router` and `worker`. The old code moves to `legacy/`.
- Bus messages version 2: amounts are integers in the minor unit.
- Requests are signed as Standard Webhooks describes.
- Retry waits are now 30 seconds, 2 minutes, 10 minutes and 1 hour.

## 1.0.0 (2025-03-11)

- First release: the dispatcher.
