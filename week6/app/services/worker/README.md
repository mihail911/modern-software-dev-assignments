# worker

Takes deliveries from `router`'s queue, signs them, sends them to the merchant and reports what happened. Go, standard library only.

## The loop

1. Lease one delivery from `router` (`POST /queue/lease`). The lease lasts 60 seconds.
2. Write the attempt to the journal.
3. Sign and send the request. Five seconds to answer.
4. Write the result to the journal, report it to `router`, close the lease.

## The journal

`state/worker/journal.jsonl`, append-only, one record per attempt:

```json
{"n": 41, "delivery": "dlv_…", "attempt": "att_…", "status": 200}
```

## After a restart

The worker reads the journal. An attempt that was written with no status is one whose outcome nobody knows: the request may or may not have reached the merchant. Delivery is at most once, so the worker does not send it again. It reports the attempt as delivered and closes the lease.

## Tests

```
make test
```
