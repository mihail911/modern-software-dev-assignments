# Runbook: a merchant quotes an attempt id

Priya. Written for release 2.3.

Use this when a ticket has an attempt id (`att_…`, from the merchant's copy of the `x-switchyard-attempt` header) and you need to know what we sent and what came back.

## Find the record

1. Work out which worker it was. If you cannot, try each; there are few.
2. Run

   ```
   bin/attempt-id --find att_3d0a77e1c2f7a0 wrk_1
   ```

   It prints the record number, or `not found` after two million tries. Not found means another worker, or an attempt from before 2.3. Those have random ids: search the logs instead.
3. Open that worker's journal (`worker/journal.jsonl`) at the line with that `n`. It has the delivery, the lease, the status the merchant answered with and whether the result was reported.

To go the other way, `bin/attempt-id wrk_1 41 42` prints the ids of records 41 and 42.

## What to tell the merchant

- A status in the journal is what their server answered. If it is a 2xx the delivery counts as delivered and we will not send it again by ourselves.
- No status means the request started and we never saw an answer. It counted as an attempt and the delivery was retried, so they may have received it twice under the same `webhook-id`.

## Do not

- Do not change how the worker makes ids because a lookup fails. Every id on file would stop resolving (ADR-0011). A lookup that fails is the wrong worker or an old attempt.
- Do not edit the journal.
