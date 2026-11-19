# Delivery contract

Accepted events get IDs and should not be lost.

Deliveries become `delivered` after a 2XX response to one of its attempts is recorded.

Unless an endpoint is disabled, a delivery ends as `delivered` or `dead`.

Merchants can receive a delivery more than once (under the same ID).

Only manual resends can send an event under two delivery IDs to the same endpoint.

Requests are signed according to the [Standard Webhooks](https://www.standardwebhooks.com/) specification
