# Runbook: dead deliveries

Wale. Written for release 2.2.

From 2.2 a delivery that cannot be delivered ends `dead`. Before 2.2 it ended `failed` after five attempts; the handling below replaces what support did for those.

## When a delivery dies

A delivery is dead after six counted attempts without a 2xx. A counted attempt is one that ended `failed` (any status outside 2xx, or no response within five seconds) or `unknown` (the attempt started and its result was never recorded).

| After counted attempt | Wait before the next |
|---|---|
| 1 | 30 seconds |
| 2 | 2 minutes |
| 3 | 10 minutes |
| 4 | 1 hour |
| 5 | 6 hours |
| 6 | none: the delivery is dead |

From the first attempt to the sixth is a little over 7 hours 12 minutes. A dead delivery is never sent again by itself.

## Finding them

- The router logs `delivery.dead` with the delivery, the last attempt and the outcome of that attempt.
- The router's state lists every delivery with its `state`, `attempts` and `last_status`. Dead ones have `state` `dead` and `attempts` 6.
- The dashboard's "Needs attention" tab is the same list, per merchant.

Look at `last_status` first. A run of `404` or `410` is an endpoint that has moved. `401` or `403` is usually a merchant verifying with the wrong secret. No status at all means their server did not answer.

## What to do

1. **Find out whether the endpoint is fixed.** Resending to an endpoint that is still broken produces another dead delivery seven hours later.
2. **Resend.** `POST /v1/events/{id}/resend` with `{"endpoint_id": "ep_…"}`. This creates a new delivery with its own id; its `resend_of` names the original. It is attempted, retried and counted on its own, from zero.
3. **Tell the merchant** that the resend arrives under a new delivery id. A merchant who de-duplicates on `webhook-id` will process it, which is the point.

A resend for an event and endpoint that never had a delivery does nothing. That is not an error: the endpoint was not subscribed, or was disabled, when the event was routed.

## An endpoint that is going to be down for a while

Have the merchant disable it, or disable it for them. Deliveries to a disabled endpoint wait and no attempt is counted while they wait, so nothing dies. When it is re-enabled, the waiting deliveries are sent, and events from the last 24 hours that were routed while it was disabled get a delivery. Anything older than 24 hours has to be resent by hand.

## Do not

- Do not change a delivery's state in the router's database. A dead delivery set back to `queued` has six attempts on it and dies again on the next failure.
- Do not resend in bulk without the merchant's agreement. Some would rather reconcile from the settlement file than receive four hundred webhooks at once.
