# Runbook: collecting an evidence packet

Wale.

Every incident ticket gets an evidence packet: the logs for the period the incident covers, in one directory, attached to the ticket. What the files contain is defined in `contracts/evidence.md`. This is how to produce one.

## What goes in

| File | Taken from |
|---|---|
| `api.ndjson` | The `api` log, as written |
| `router.log` | The `router` log, as written |
| `worker.jsonl.gz` | The `worker` log, gzipped |
| `egress.log` | The egress proxy, from the platform team |

## Steps

1. **Fix the period.** Start a few minutes before the first sign of trouble and end after the system was quiet again. Write the period, in UTC, in the ticket.
2. **Copy the service logs** for that period from each service's `logs/` directory. Cut on whole lines. Do not filter, sort or reformat: the code is the reference for what a record means, and people will read the packet with the code open.
3. **Ask platform for `egress.log`** for the same period. The proxy writes one line for each request that reached it, in the order it handled them, with the status the merchant's server answered, or `000` and a reason.
4. **Replace merchant addresses.** Hosts become `hooks.merchant.example`; paths stay. The collection script does this in all four files.
5. **Attach the directory** to the ticket. Do not edit a packet after it is attached; if the period was wrong, attach a second one.

## Reading one

- `api.ndjson` and `router.log` carry UTC timestamps. `worker.jsonl.gz` carries `t` in whole seconds since 1970. `egress.log` is UTC.
- Ids join the files: the event id, the delivery id (`webhook-id` in `egress.log`) and the attempt id appear wherever they are known.
