# ADR-0013: Idempotency keys are held in memory

- Status: accepted
- Date: 2026-06-02
- Author: Jun

## Context

Merchants retry `POST /v1/events` when they do not hear back in time, and some of those retries reach us after the first request went through. The merchant then gets two events for one payment. Three of the large merchants asked for a way to send a request twice safely, and two of them already send an `Idempotency-Key` header to their payment provider and would like to reuse it.

The api accepts about 400 events a second at peak. A lookup by key on every accept is one more indexed read on the hottest path we have, inside the transaction that writes the event and its outbox row.

## Decision

From release 2.3 a merchant may send `idempotency_key` with an event. If the same merchant sends the same key again within 24 hours, the api answers `200` with the event it accepted the first time and creates nothing.

- The keys are held in the api's memory, per merchant, for 24 hours. This is how our payment provider describes theirs, and it is what the merchants asked for.
- The key is also written to the event row, so Support can find an event by the key a merchant quotes. Nothing reads it back on the request path.
- No new index on `events`.

## Consequences

- Each api process holds the keys it has seen. Memory grows with the number of keys sent in a day: about 30 MB per process at current volumes.
- Support finds an event by the key a merchant quotes with `GET /v1/events?merchant_id=…&idempotency_key=…`, which reads the event rows.
