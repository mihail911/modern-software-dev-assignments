# INC-2188: merchants receive an event twice after changing an endpoint

- Date of incident: 2026-07-08 to 2026-08-14
- Written: 2026-08-21, by Rhea with Jun.
- Releases affected: 2.3.0
- Severity: 3

## Summary

Release 2.3 introduced endpoint versions. The router began to identify a delivery by its event, its endpoint and the endpoint's version. When a merchant changed an endpoint while a delivery to it was still queued, the router did not recognise the queued delivery as belonging to the new version and created a second one. The merchant received the event twice, under two delivery ids, so de-duplicating on `webhook-id` did not help.

## Timeline

| Date | |
|---|---|
| 2026-07-08 | 2.3.0 finishes rolling out to production. |
| 2026-08-04 | First ticket: a merchant receives a `payout.paid` twice with different `webhook-id`s after rotating a secret. Treated as a problem on the merchant's side. |
| 2026-08-06 | Third ticket with the same pattern. Support opens INC-2188. Jun is on call. |
| 2026-08-07 | Jun and Rhea reproduce it in staging: queue a delivery for a retry, change the endpoint, and a second delivery appears. Support starts asking affected merchants to de-duplicate on the event id (`docs/runbooks/endpoint-changes.md`). |
| 2026-08-11 | Cause confirmed in the router's routing code. A full fix changes how deliveries are identified and is too large for a patch release. |
| 2026-08-14 | 2.3.1 is in production with a flag that stops the router creating deliveries when an endpoint changes. The flag is on. No new duplicates after this. |

## Impact

Between 2026-07-08 and 2026-08-14:

- **3,904 events** were delivered twice to the same endpoint under two delivery ids, at 212 endpoints of 87 merchants.
- Nine in ten of them were deliveries waiting for a retry. Merchants change an endpoint when it is failing, and a failing endpoint is the one with deliveries in the queue.
- Four merchants reported acting twice on an event (two sent duplicate receipts, two released an order twice). The rest de-duplicated on something else or did not notice.
- No event was lost, and nothing was sent to an endpoint that should not have had it.

## Root cause

ADR-0008 requires exactly one delivery per event and endpoint. Until 2.3 the router checked for an existing delivery by event and endpoint. In 2.3 an endpoint gained a version, raised when the endpoint is changed, and the delivery's key gained the version with it, so that a delivery could record which version of the endpoint it was created for.

When an endpoint change arrived, the router made sure that each of the endpoint's unfinished events had a delivery at the current version. The existing delivery was keyed with the old version, so the check found nothing and a new delivery was created. The old one was left as it was. Both were leased and sent.

The tests for 2.3 changed endpoints only when the queue was empty.

## Fix

2.3.1 stops the bleeding: with the flag on, an endpoint change creates no deliveries, so a queued delivery goes out once, to the endpoint's current URL and secret. The flag does not change how a delivery is identified.

The fix is to identify a delivery by its event and its endpoint, whatever the endpoint's version, so that a change to an endpoint neither creates nor removes a delivery. It needs a migration of the deliveries table and is planned for 2.4.

## Follow-ups

| | Owner | State |
|---|---|---|
| Workaround flag in the router | Rhea | Done, 2.3.1 (2026-08-14) |
| One delivery per event and endpoint regardless of version | Rhea | Planned for 2.4 |
| State in `contracts/delivery.md` that changing an endpoint never creates or removes a delivery | Ines | Done |
| Test: change an endpoint with deliveries queued, leased and waiting for a retry | Rhea | Planned for 2.4 |
| Tell the 87 merchants, and tell them when they can go back to de-duplicating on `webhook-id` | Jun | Told 2026-08-22; second note when 2.4 ships |
| Support to open an incident on the second ticket with one pattern, not the third | Wale | Done |
