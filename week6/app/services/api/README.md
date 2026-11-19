# api

Accepts events and endpoint configuration from merchants' systems and publishes them to the bus. TypeScript, run directly by Node; there is no build step.

## What it owns

- **Events.** `POST /v1/events` stores the event and assigns its id (`evt_…`).
- **Deliveries' ids.** When an event is accepted, `api` works out which endpoints it goes to and assigns one delivery id (`dlv_…`) per endpoint. The ids travel with the event message, so `router` and `worker` use them as given.
- **Endpoints.** `POST /v1/endpoints` and `PUT /v1/endpoints/{id}`.

## Bus messages

`api` publishes version 2 messages to `router`:

```json
{"v": 2, "seq": 17, "kind": "event",
 "event": {"id": "evt_…", "merchant": "m_…", "type": "payment.captured", "created": 1756800000,
           "payment_id": "pay_…", "amount_minor": 1250, "currency": "USD"}}
```

`seq` rises by one per message. `router` drops a `seq` it has already seen.

## Storage

One SQLite file, `state/api/api.db`. Events and the outbox row for their bus message are written in one transaction.

## Tests

```
npm test
```
