# Runbook: a worker was restarted

Wale. Written for release 2.0.

Use this when a worker has been restarted, by a deploy, a node drain or a crash, and you want to know whether anything needs doing.

**Short answer: no action is needed.**

## What a worker holds

A worker holds one delivery at a time, under a lease from the router that lasts 60 seconds. It writes what it is doing to its journal (`worker/journal`, format 1) before it does it: that it took the lease, that it started the request, and what the result was.

## What happens on restart

When the worker starts it reads its journal before it asks the router for anything.

| The journal says | The worker does |
|---|---|
| Lease taken, request not started | Closes the lease. The router queues the delivery again. |
| Request started, no result | **Settles the attempt as delivered** and reports that to the router. |
| Request finished, result not reported | Reports the result. |

The second row is deliberate. We do not know whether the merchant received the request, and delivery is at most once (ADR-0003): a delivery that may have arrived is never sent again. The merchant picks up anything that did not arrive from the settlement file.

## If the worker does not come back

Its lease expires 60 seconds after it was granted. The router then gives the delivery to another worker. A delivery whose request had already started can in that case be sent a second time; this is the only way it can be, and it needs the worker to be gone for good.

So when you restart workers by hand, do it one at a time and let each come back within the minute.

## Checking

- `GET /__admin/health` on the worker answers `200`.
- The router's state shows no delivery that has been `leased` for more than 60 seconds.
- The worker's log has a `settled` line for anything it settled from the journal. A handful per restart is normal. Tens means the worker was stopped while it was busy and is still nothing to act on.

## Do not

- Do not delete or edit the journal. A worker that starts without one has forgotten what it held, and the delivery waits out its lease.
- Do not requeue a settled delivery by hand. If a merchant says an event is missing, point them to the settlement file.
