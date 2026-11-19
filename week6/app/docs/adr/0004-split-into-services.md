# ADR-0004: Split the dispatcher into services

- Status: accepted
- Date: 2025-06-10
- Author: Ines

## Context

The dispatcher (ADR-0001) accepts events and sends them from one process. Volume is now about forty times the estimate it was built for. Three things hurt:

- Sending is sequential, so a handful of slow merchants delay everyone. At the May peak the oldest due delivery was eleven minutes late.
- Every deploy of the merchant-facing API stops sending, and every change to sending risks the API.
- The merchant API is now maintained by the team that writes TypeScript; the dispatcher is the only Python they touch.

## Decision

From release 2.0 Switchyard is three services.

- **`api`** (TypeScript) accepts events and endpoint configuration and publishes them to the router. It assigns ids (ADR-0006).
- **`router`** (Python) matches events to endpoints and owns the delivery queue, in SQLite. It is the dispatcher's storage and routing, carried over.
- **`worker`** (Go) sends. A worker leases one delivery from the router, makes the request, reports the result and closes the lease. A lease lasts 60 seconds. A worker keeps a journal of what it holds (format 1), so that it knows after a restart.

`api` talks to `router` in bus messages, version 2. Amounts in version 2 are integers in the currency's minor unit; the decimal strings of release 1 become version 1 and are still accepted, because archived messages are replayed.

The retry schedule becomes 30 seconds, 2 minutes, 10 minutes and 1 hour: still five attempts, spread over a longer period, because most endpoint outages we see last longer than the old schedule did.

The dispatcher's code moves to `legacy/` and is kept for reference.

## Consequences

- Workers scale separately and a slow merchant occupies one worker, not the system.
- The delivery guarantee does not change (ADR-0003). An attempt with no recorded outcome is settled as delivered, now by the worker from its journal.
- There are now two places a delivery's state lives, the router's queue and a worker's journal, and a protocol between them. The lease is what keeps them honest: a delivery a worker never reports comes back when its lease expires.
- Three languages. Each service is small enough to be owned by the people who already write that language.
