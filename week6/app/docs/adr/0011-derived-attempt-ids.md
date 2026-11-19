# ADR-0011: Attempt ids are derived, not random

- Status: accepted
- Date: 2026-05-19
- Author: Ines

## Context

Every request to a merchant carries `x-switchyard-attempt`. Merchants log it, and when something looks wrong on their side it is what they send us: 31 of the 44 delivery tickets Support handled in April quoted an attempt id and nothing else.

The worker makes the id from eight random bytes. To find the attempt behind one, Support searches the worker logs for it. Logs are kept for 14 days and a search over every worker takes most of an hour, so anything older than two weeks cannot be found at all. The journal still has the attempt, but nobody can tell which worker's journal to open or where in it to look.

## Decision

From release 2.3 an attempt id is computed from who made the attempt and where it sits in that worker's journal:

```
att_ + the first 14 hex digits of sha256("switchyard:att:<worker id>:<n>")
```

`n` is the number of the journal record that starts the attempt.

- A worker's journal numbers its records from 1 and the numbers only go up, so one worker never gives two attempts the same id. The worker id keeps two workers apart.
- The id is as opaque to a merchant as it was. Nothing changes for them.
- `bin/attempt-id` goes the other way: given an id and a worker it finds `n`. Support runs it; see `runbooks/attempt-lookup.md`.

## Consequences

- Support's lookup now depends on how the id is made. The worker has a test that pins the first ids. If that test fails the derivation has changed, and every id Support has on file stops resolving.
- Attempts made before 2.3 keep their random ids and are found the old way until their logs expire.
- 14 hex digits is 56 bits. At our volumes a chance collision between two workers is not something we plan for.
