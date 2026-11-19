# ADR-0003: Delivery is at most once

- Status: superseded by ADR-0009
- Date: 2025-03-04 (accepted)
- Author: Tomas

## Context

The dispatcher marks a delivery as being sent, makes the request, and stores the result. If the process stops between the first step and the last, we do not know whether the merchant's server received the request. On the next start we have to choose: send it again and risk a duplicate, or leave it and risk a gap.

We asked the eleven merchants in the pilot how their handlers behave. Eight act on `payment.captured` directly: they release an order, credit an account or send a receipt. None of them de-duplicates. All of them already reconcile against the nightly settlement file.

## Decision

Delivery is at most once.

- A request that is in flight when the process stops is closed as delivered when the process starts again. It is never repeated.
- A request that is known to have failed, because the server answered outside 2xx, refused the connection or did not answer within five seconds, is retried. There are five attempts, with waits of 10 seconds, 1 minute, 5 minutes and 30 minutes between them. After the fifth the delivery is failed and left alone.
- The same delivery id is sent on every attempt of a delivery.

## Consequences

- A merchant can miss an event when the dispatcher is restarted at the wrong moment. The settlement file covers it the next morning. We say so in the merchant documentation.
- A restart never causes a merchant to receive an event it has already acted on.
- Deliveries closed this way are marked so that we can count them. If the count stops being small, this decision should be looked at again.
- A request that timed out on our side but was processed on theirs is retried and can arrive twice. We accept that case; it is rare and merchants see the same delivery id both times.
