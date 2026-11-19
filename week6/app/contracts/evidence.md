# Evidence Packets

Incident tickets come with a directory of production logs collected in the period it covers.

- `api.ndjson` (from `api`)
- `router.log` (from `router`)
- `worker.jsonl.gz` (from `worker`)
- `egress.log` (from the egress proxy)
- `merchant_export.csv` (from the merchant who reported the incident)
- `snapshots/router-{time}.json` (from `router)

Some other conventions:

- Merchant addresses appear as `https://hooks.merchant.example/…`.
- Service logs hold the outputs of each service's code.

## `api.ndjson`

```json
{
  "ts": "2026-10-12T08:14:07Z",
  "svc": "api",
  "msg": "event.accepted",
  "event": "evt_…",
  "merchant": "m_…",
  "type": "payment.captured",
  "amount_minor": 1250,
  "currency": "USD"
}
```

`ts` is UTC.

## `router.log`

```
2026-10-12T08:14:07Z level=info msg=delivery.created delivery=dlv_… event=evt_… endpoint=ep_…
```

The timestamp is UTC. A value that contains a space is in double quotes.

## `worker.jsonl.gz`

```json
{
  "t": 1791792847,
  "worker": "wrk_1",
  "op": "attempt.finished",
  "lease": "lse_…",
  "delivery": "dlv_…",
  "attempt": "att_…",
  "status": 200
}
```

`t` is in whole seconds since 1970-01-01T00:00:00Z.

## `egress.log`

```
2026-10-12T08:14:07Z egress POST https://hooks.merchant.example/hooks/ep_… 200 webhook-id=dlv_… attempt=att_… event=evt_… bytes=174
```

The proxy is not part of this repository. It writes one line for each request that reached it, in the order it handled them. The field after the address is the status the merchant's server answered with.

| Status                    | Meaning                                                            |
| ------------------------- | ------------------------------------------------------------------ |
| A three-digit HTTP status | The merchant's server answered                                     |
| `000 err=upstream_reset`  | The request was passed on and the connection closed with no answer |
| `000 err=connect_refused` | The merchant's server could not be reached                         |

## `merchant_export.csv`

The merchant who reported the incident exports what their own server recorded: one row for each webhook request it processed.

```
received_at,webhook_id,event_id,type,amount,currency,http_status
10/12/2026 09:14,dlv_…,evt_…,payment.captured,"1,250.00",USD,200
```

- `received_at` is the merchant's local time, to the minute.
- `webhook_id` is the request's `webhook-id` header. The spreadsheet cuts some ids short. A cut id stands for the one delivery whose id starts with it.
- `amount` is in the currency's major unit, in whatever form the merchant's tools wrote it: `12.5`, `12.50` and `1,250.00` are all valid. A comma separates thousands.
- `http_status` is the status the merchant's server answered with. It is empty when the server processed the request and the connection closed before it answered.

## `snapshots/`

`snapshots/router-<time>.json` is `router`'s state at that time, with the fields `harness.md` lists. A snapshot says nothing about what happened after it was taken.

## How the logs are collected

The logs are written on different hosts and reach a packet through a log shipper. This is what that does to them.

- Each host stamps its records with its own clock. The clocks are not synchronised: the same moment can be a minute apart in two files.
- The shipper delivers some batches late, and the egress proxy writes a request's line when the request ends. Lines are not always in the order things happened.
- The shipper sends some batches twice.
- A write is sometimes cut short. In `api.ndjson` and the worker's log the line is then not valid JSON. In `router.log` and `egress.log` the line then ends in a field with no value, such as `… attempt=` or `… att`. In the export the row then has fewer than seven columns. The row is corrupted. Skip it and count it, each time it appears. Do not use what is left of it.

Because rows are lost this way, no single file is complete. No fact that an incident report depends on is missing from every file.
