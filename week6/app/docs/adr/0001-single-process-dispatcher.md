# ADR-0001: A single-process dispatcher

- Status: superseded by ADR-0004
- Date: 2025-01-14
- Author: Tomas

## Context

Merchants want to be told when a payment is captured, fails, is refunded or is paid out, without polling the payments API. Today they are told by email and by the nightly settlement file. We expect a few thousand events a day in the first year, from a few hundred merchants, most with one endpoint.

We are two people. We need something we can run, watch and change without a platform underneath it.

## Decision

Switchyard is one Python process, the dispatcher, with one SQLite file.

- The platform posts an event to it. The dispatcher stores the event and creates one delivery for each of the merchant's endpoints that subscribes to the event's type. Delivery ids are fixed at that moment and returned in the reply.
- The same process sends the deliveries that are due, one after another, and stores the result of each.
- An event carries its amount as a decimal string in the currency's major unit (`"12.50"`), as the platform's own API does.
- Each request carries `X-Switchyard-Signature: sha256=<hex>`, an HMAC-SHA256 of the body keyed with the endpoint's secret.
- Standard library only. No broker, no worker pool, no second process.

## Consequences

- One thing to deploy, one log, one file to back up.
- Accepting and sending share a process. While it is restarted nothing is accepted and nothing is sent; the platform retries its post.
- Sending is sequential. One merchant whose server takes the full five seconds to answer holds up everyone behind it. This is acceptable at the volume we expect and will not be at ten times that.
- What happens to a request that is in flight when the process stops is not settled here. It needs its own decision before the first release.
