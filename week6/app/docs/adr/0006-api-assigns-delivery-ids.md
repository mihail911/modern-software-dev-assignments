# ADR-0006: `api` assigns delivery ids

- Status: superseded by ADR-0008
- Date: 2025-07-08
- Author: Ines

## Context

In release 1 a delivery got its id when its event was accepted: `dlv_<event suffix>_<n>`, with n counting the merchant's subscribed endpoints from 1. The ids are in the reply to the platform, the platform stores them, and the merchant dashboard shows them next to each event. Support looks deliveries up by them.

With the split (ADR-0004) an event is accepted by `api` and its deliveries are created later, by `router`. Something has to say what their ids are.

## Decision

`api` assigns the event id and the delivery ids, at the moment the event is accepted, by the release 1 rule. They are in the reply to `POST /v1/events`, as before.

`router` and `worker` never invent an id. The version-2 bus message does not list the delivery ids; `router` arrives at them by applying the same rule to the event id and the merchant's endpoints. The rule is fixed by this record and may not change in one service without the other.

## Consequences

- Nothing changes for the platform, the dashboard or support. An id shown to a merchant is the id in the request.
- The id of a delivery is known before the delivery exists. A reply can name a delivery that the router has not created yet.
- `api` and `router` must agree on a merchant's endpoints, and their order, at the moment an event is accepted. They do as long as endpoints are only ever added and endpoint messages are applied in order.
- A delivery cannot be created for an event after the fact, since its id would have had to be in the reply.
