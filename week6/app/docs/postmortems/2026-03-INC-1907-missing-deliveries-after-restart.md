# INC-1907: deliveries missing after worker restarts

- Date of incident: 2026-03-11
- Written: 2026-03-24, by Wale with Priya and Rhea
- Release at the time: 2.2
- Severity: 2

## Summary

During a planned node drain, workers were restarted and took longer to come back than their leases lasted. The router handed their deliveries to other workers while the first workers' requests were still in flight or still unreported. One worker host's clock was about 90 seconds behind the router's, so workers there did not know their leases had gone. Reports that arrived late were accepted. 47 deliveries were recorded as delivered that no merchant had acknowledged, and 1,310 were sent twice.

The cause was the length of the lease and the clock skew. The worker's recovery after a restart did what it was designed to do.

## Timeline (UTC, 2026-03-11)

| Time | |
|---|---|
| 13:40 | Platform starts the scheduled node pool upgrade in production. Nodes are drained one at a time. |
| 13:52 | The first node carrying workers is drained. Replacement pods take between 70 seconds and 4 minutes to become ready on the new nodes (image pull, volume attach). |
| 13:53 | The router starts expiring leases held by workers that are down and granting the same deliveries to others. This continues with each node until 14:26. |
| 14:26 | The drain finishes. All workers are running. |
| 14:47 | First support ticket: a merchant is missing `payment.captured` events that the dashboard shows as delivered. |
| 15:10 | Third ticket of the same kind. Support pages on-call (Wale). |
| 15:35 | Priya finds deliveries in the router marked delivered for which no worker journal has an attempt with a 2xx. |
| 16:02 | Rhea finds reports accepted after the lease they name had expired. One new node is found running 91 seconds behind the router's host; time sync had not completed after provisioning. |
| 16:05 | The node's clock is stepped. |
| 16:30 | Queue behaviour is normal. Work moves to finding what was affected. |

On 2026-03-12 the affected deliveries were identified by comparing the router's records with the workers' logs, and the 47 were resent by hand. Merchants were told the same day.

## Impact

- **47 deliveries, 19 merchants:** recorded `delivered` with no 2xx from the merchant. They would never have been retried. All were resent on 2026-03-12, between 20 and 26 hours late.
- **1,310 deliveries, 204 merchants:** sent twice under the same delivery id. This is within what at-least-once allows, and merchants who de-duplicate on `webhook-id` saw nothing, but it is some hundreds of times the normal rate.
- Nothing was dead-lettered and no event was lost.

## Root cause

**The lease was shorter than a restart.** A lease lasted 60 seconds. In a rolling deploy a worker is back in about 15. During the drain the replacements needed 70 seconds or more, so every delivery held by a drained worker was back in the queue, and usually with another worker, before its first holder returned.

**Clock skew hid this from some workers.** A worker treats a lease as its own until the lease's `expires_at`, which is the router's time, has passed on the worker's own clock. On the node that was 91 seconds behind, a lease looked valid for about 150 seconds. Seen from that node the router was expiring leases a minute and a half early. Workers restarted there carried on under leases the router had already granted to someone else: they sent the request, which is where most of the duplicates come from, and reported the result.

**Late reports were accepted.** In 2.2 the router reuses a lease id once the lease has expired, and it applied a report to whatever delivery the lease held when the report arrived. The report named its delivery; the router did not compare. A late `delivered` from a lease's previous holder therefore closed the delivery of its current holder. Where the current holder's own attempt then failed, its report came second and was dropped, and nothing was ever retried. Those are the 47.

Worker recovery is not at fault. A restarted worker reported what its journal showed, as designed (ADR-0009). The reports were wrong only because the leases behind them had moved on, which neither the worker (skew) nor the router (no check) could see.

## Fix

1. **Leases last 120 seconds**, up from 60. Released in 2.2.3 on 2026-03-19. 120 is longer than all but two of the restarts seen during the drain.
2. **The router ignores a report for a delivery that is not held under the reporting lease.** It changes nothing and logs `report.ignored` with the delivery, the lease and the attempt.
3. **Alarms on time sync.** Every node alarms when its offset is more than 2 seconds for 5 minutes, and a node does not take pods until it has synchronised once. Platform, done 2026-03-17.

## Follow-ups

| | Owner | State |
|---|---|---|
| Stop reusing lease ids; every grant gets its own | Rhea | Open, with the next change to the lease table |
| `drain-workers.sh` waits for each replacement to be ready, and one lease length, before the next | Wale | Done |
| Dashboard query for deliveries marked delivered with no 2xx attempt, run weekly | Priya | Done 2026-03-20 |
| Platform to tell `#switchyard-oncall` before node pool upgrades | Wale | Done |
| Look at replacing the fixed lease with something a busy worker extends | Jun | Open |
