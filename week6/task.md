# INC-2207: merchants receiving the same payment twice

## Report

Merchant `m_maple` got some payments twice this week, as two webhooks with different event ids. A few other merchants have said the same thing.

## What we need

**1. A fix.** You are responsible for delivering:

- An update to `app/services` that fixes the cause of the repeated payments.
- For repeated events, automatically cancel outstanding deliveries, including those already queued for delivery. A cancelled delivery doesn't need to end delivered or dead. No manual follow-up.

For deployment purposes:

- Keep older bus message versions working.
- The same release goes to every region. We only have logs for this one.

**2. A blast radius report.** For each merchant and idempotency key, treat the first event accepted by the API as the original and every later event accepted under that same key as a repeat. Enumerate deliveries of those repeated events that received at least one 2xx response, regardless of whether or when the original event was delivered. Write them to `out/impact.json`:

```json
{
  "affected": [
    { "event_id": "evt_…", "endpoint_id": "ep_…", "delivery_ids": ["dlv_…"] }
  ],
  "totals_minor": { "USD": 0, "EUR": 0 },
  "skipped_rows": 0
}
```

- One entry in `affected` per repeated event and endpoint.
- `totals_minor` is the sum of those entries' event amounts in minor units, per currency. Leave out currencies with none.
- `skipped_rows` is the number of corrupted rows in `incident/`.

## Note from Support

`m_maple`'s SDK retries aggressively. We've asked them to back off.
