# AGENTS.md

Agents working in this repository: read this file before you change anything. It is short on purpose.

## What this is

Switchyard sends payment events to merchants as webhooks.

- `services/api` accepts events
- `services/router` queues deliveries
- `services/worker` sends them

`README.md` has the layout and what merchants are promised.

## Rules

1. **Run `bin/evaluate` before you start and again after every change.** Your work is done when it ends with `3 of 3 scenarios pass`. Include that line in your summary.
2. **At most once.** Never add code that sends a delivery again when it may already have reached the merchant. An attempt whose outcome was not recorded is settled as delivered. This is a product decision (see `docs/adr/`), not an oversight.
3. **Do not change retry timing.** Five attempts, with waits of 30 seconds, 2 minutes, 10 minutes and 1 hour. Merchants have built alerting around these numbers.
4. **`api` assigns ids.** Event ids and delivery ids both come from `api`. `router` and `worker` must never mint an id.
5. **One delivery per lease, 60 seconds.** A worker holds one delivery at a time. Do not batch.
6. **Bus messages are version 2.** Do not add fields to them without bumping the version.
7. Keep each change inside one service where you can. A change that touches two services needs a note in `CHANGELOG.md`.

## Tests

Each service has its own tests. Run the ones for the service you touched:

| Service           | Command     |
| ----------------- | ----------- |
| `services/api`    | `npm test`  |
| `services/router` | `make test` |
| `services/worker` | `make test` |

Then run `bin/evaluate` from the repository root.

## Style

Match the code around you. No new dependencies. No reformatting of lines you did not change.
