# Runbook: a merchant gets an event twice after changing an endpoint

Jun. Written 2026-08-07, release 2.3.0. Incident INC-2188.

## Symptom

A merchant reports the same event arriving twice at one endpoint, a few seconds to a few hours apart. The two requests have:

- the same body, the same event `id` and the same `x-switchyard-event`;
- **different `webhook-id`s**, so the merchant's de-duplication does not catch the second.

It happens shortly after the merchant changed the endpoint: a new URL, a rotated secret, a changed subscription. Most often the merchant changed it *because* deliveries were failing, which is exactly when there are deliveries in the queue waiting for a retry.

This is not the ordinary at-least-once repeat. An ordinary repeat carries the same `webhook-id` both times.

## Confirming it

In `router.log`, look for the event id. The pattern is two `delivery.created` records for the same `event` and `endpoint`, neither with a `resend_of`, and an `endpoint.changed` record for that endpoint between them:

```
… msg=delivery.created delivery=dlv_… event=evt_… endpoint=ep_…
… msg=endpoint.changed endpoint=ep_… merchant=m_… version=4 enabled=true
… msg=delivery.created delivery=dlv_… event=evt_… endpoint=ep_…
```

If the second `delivery.created` has a `resend_of`, somebody resent the event by hand and this runbook does not apply.

## Cause

Since 2.3 the router identifies a delivery by its event, its endpoint and the endpoint's version. A change raises the version. A delivery that was still queued under the old version is not recognised as the delivery for the new one, and the router creates another. Both are sent.

Deliveries that had already finished when the endpoint changed are not affected.

## What to tell the merchant

Until this is fixed, ask the merchant to **de-duplicate on the event id** (the body's `id`, or `x-switchyard-event`) and not on `webhook-id`.

Tell them the one thing this costs: a manual resend is also the same event under a new delivery id, so while they de-duplicate on the event id a resend will be dropped on their side. If they need a resend during this period, they have to let it through by hand.

If they ask whether they were charged or paid twice: no. The event is a notification; there is one payment.

## What not to do

- Do not delete the second delivery from the router's database. If the first one then fails, the merchant has nothing.
- Do not ask merchants to avoid changing endpoints. They change them to fix failures, and we want that.
- Do not disable and re-enable the endpoint to "reset" it. Catch-up will add to the confusion.

## Status

Rhea is working on the router. Add affected merchants to the INC-2188 sheet so they can be told when the fix is out.
