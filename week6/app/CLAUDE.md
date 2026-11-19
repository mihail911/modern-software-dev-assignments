# Switchyard: notes for Claude

The same rules as `AGENTS.md`, kept here because Claude Code loads this file by itself.

- Verify with `bin/evaluate`. Run it before and after your change; you are done when it prints `3 of 3 scenarios pass`.
- Delivery is at most once: one delivery attempt in flight is never repeated. If the outcome of an attempt was not recorded, it is settled as delivered. Do not "fix" this.
- Retry timing is fixed: five attempts, waiting 30 seconds, 2 minutes, 10 minutes and 1 hour.
- `api` assigns event ids and delivery ids. `router` and `worker` never create ids.
- A lease holds one delivery and lasts 60 seconds.
- Service tests: `npm test` in `services/api`, `make test` in `services/router` and `services/worker`.
- `legacy/` is the old dispatcher. It still documents the delivery guarantees best; read `legacy/README.md` if you are unsure what merchants expect.
