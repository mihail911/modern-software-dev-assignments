# ADR-0009: Delivery is at least once

- Status: accepted
- Date: 2026-01-13 (accepted)
- Author: Rhea

## Context

Since release 1 an attempt whose outcome was never recorded has been settled as delivered (ADR-0003). The reasoning was that merchants did not de-duplicate and that the settlement file covered the gaps.

Both have changed. Since 2.0 every request carries a `webhook-id` (ADR-0010), and from 2.2 that id is the delivery id, assigned once by the router and the same on every attempt (ADR-0008), so a merchant has something reliable to de-duplicate on. And merchants now drive fulfilment from webhooks alone: a `payment.captured` that never arrives is an order that is never shipped. In the last quarter of 2025 we settled about 2,300 attempts as delivered without knowing; support traced 61 merchant complaints to them.

## Decision

From release 2.2 delivery is at least once.

- An attempt that started and whose result was never recorded has the outcome `unknown`. It is not delivered. It counts as an attempt and the delivery is retried, under the same delivery id.
- A delivery is `delivered` only if a 2xx response to one of its attempts was recorded.
- A sixth attempt is added, 6 hours after the fifth. The waits are 30 seconds, 2 minutes, 10 minutes, 1 hour and 6 hours. After six counted attempts without a 2xx the delivery is `dead`.
- Switchyard does not promise exactly once and will not try to.

## Consequences

- A merchant can receive the same delivery more than once, for instance when its answer did not reach us. It always carries the same delivery id. Merchants must de-duplicate on `webhook-id`; the merchant documentation and the SDK examples say so from 2.2.
- A merchant never receives one event at one endpoint under two delivery ids, except through a manual resend.
- A worker that stops in the middle of a request no longer loses the delivery. It costs an attempt.
- `unknown` is a third outcome everywhere outcomes appear: the queue protocol, the router's state, the dashboard.
- Every delivery now ends `delivered` or `dead`. A dead delivery is something support can see and act on (manual resend).
