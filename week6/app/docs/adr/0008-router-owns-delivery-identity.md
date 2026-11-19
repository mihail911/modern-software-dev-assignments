# ADR-0008: `router` owns delivery identity

- Status: accepted
- Date: 2025-12-09
- Author: Rhea

## Context

ADR-0006 has `api` assign delivery ids when an event is accepted, as a function of the event id and the position of each subscribed endpoint. That rested on two things that are no longer true.

- Release 2.1 lets an endpoint be disabled and re-enabled. Positions shift when an endpoint drops out, so `api` and `router` can disagree about which endpoint `_2` is. We have seen it twice in staging.
- Re-enabling an endpoint catches it up: the router creates deliveries for events accepted hours earlier. Those deliveries have no id from `api`, because none existed when the reply was sent. The manual resend planned for 2.2 has the same problem.

A delivery is created by routing. The service that routes is the only one that knows which deliveries exist.

## Decision

From release 2.2:

- `router` assigns delivery ids, when it creates the delivery. An id is `dlv_` and an opaque suffix; nothing can be read from it.
- Routing creates exactly one delivery per event and endpoint. Before it creates one, the router checks that the event and endpoint do not already have one.
- A manual resend is the one exception: it creates a second delivery, with its own id, that names the first in `resend_of`.
- `api` assigns event ids only. `worker` assigns attempt ids.
- Bus messages move to version 3, which names its producer and carries no delivery ids. Versions 1 and 2 are still accepted; their deliveries get router-assigned ids like any other.

## Consequences

- The reply to `POST /v1/events` no longer lists delivery ids. The dashboard reads them from the router's state. Delivery ids created before 2.2 keep the old form.
- "One delivery per event and endpoint" is now a rule the router enforces in one place, and the id a merchant de-duplicates on is stable for the life of the delivery.
- Catch-up and resend need nothing from `api`.
