# Switchyard documentation

Last updated for release 2.2 (February 2026).

What merchants are promised is in `contracts/`, not here. This directory holds the reasons for decisions and the instructions for people on call.

## Decision records (`adr/`)

One file per decision, numbered in the order they were proposed. Numbers are not reused; a number that is missing here was a proposal that was withdrawn before it was accepted. A record is never edited after it is accepted, except for its status line when a later record replaces it.

| Record | Decision | Status |
|---|---|---|
| [0001](adr/0001-single-process-dispatcher.md) | One process accepts and sends | Superseded by 0004 |
| [0003](adr/0003-at-most-once-delivery.md) | Delivery is at most once | Superseded by 0009 |
| [0004](adr/0004-split-into-services.md) | Split into `api`, `router`, `worker` | Accepted |
| [0006](adr/0006-api-assigns-delivery-ids.md) | `api` assigns delivery ids | Superseded by 0008 |
| [0008](adr/0008-router-owns-delivery-identity.md) | `router` assigns delivery ids | Accepted |
| [0009](adr/0009-at-least-once-delivery.md) | Delivery is at least once | Accepted |
| [0010](adr/0010-standard-webhooks-signing.md) | Requests are signed as Standard Webhooks describes | Accepted |

To propose a decision, copy the newest record, take the next number and open a pull request. Context, Decision, Consequences; half a page.

## Runbooks (`runbooks/`)

| Runbook | When to use it |
|---|---|
| [worker-restart.md](runbooks/worker-restart.md) | A worker was restarted, or is about to be |
| [dead-letters.md](runbooks/dead-letters.md) | Deliveries are ending `dead` |

Deployment has its own instructions in `deploy/README.md`.

## Postmortems (`postmortems/`)

One file per incident, named `YYYY-MM-INC-number-short-title.md`, with a timeline, the impact, the root cause, the fix and the follow-ups. Written by whoever led the incident, within two weeks of it.
