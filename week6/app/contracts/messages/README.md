# Bus messages

`api` publishes to `router` with `POST /bus`. Three message versions exist. `router` accepts all three, because archived messages are replayed and older producers are still in service.

`router` answers `200` when a message is applied or was already applied, and `422` when it is malformed. A malformed message changes nothing.

## Version 3 (release 2.2 onward)

Schema: `bus.v3.schema.json`.

```json
{
  "v": 3,
  "producer": "api",
  "seq": 17,
  "type": "event.accepted",
  "at": 1793577600,
  "data": {
    "event": {
      "id": "evt_…",
      "merchant_id": "m_…",
      "type": "payment.captured",
      "created": 1793577600,
      "data": {
        "object": "payment",
        "payment_id": "pay_…",
        "amount_minor": 1250,
        "currency": "USD"
      }
    }
  }
}
```

| `type`             | `data`                                                                                |
| ------------------ | ------------------------------------------------------------------------------------- |
| `event.accepted`   | `event`                                                                               |
| `endpoint.changed` | `endpoint`: `id`, `merchant_id`, `url`, `secret`, `event_types`, `enabled`, `version` |
| `resend.requested` | `event_id`, `endpoint_id`                                                             |

A message is identified by `producer` and `seq`. The same message can arrive more than once and is applied once.

## Version 2 (releases 2.0 and 2.1)

Schema: `bus.v2.schema.json`. Identified by `seq` among version-2 messages; the numbering is separate from version 3's.

```json
{
  "v": 2,
  "seq": 9,
  "kind": "event",
  "event": {
    "id": "evt_…",
    "merchant": "m_…",
    "type": "payment.captured",
    "created": 1793577600,
    "payment_id": "pay_…",
    "amount_minor": 1250,
    "currency": "USD"
  }
}
```

`kind` is `event` or `endpoint`. An `endpoint` message carries `endpoint`: `id`, `merchant`, `url`, `secret`, `types`, `active`.

## Version 1 (release 1)

Schema: `bus.v1.schema.json`. No version field and no sequence number; identified by the event `id`. Events only.

```json
{
  "id": "evt_…",
  "merchant": "m_…",
  "event_type": "payment.captured",
  "amount": "12.50",
  "currency": "USD",
  "payment": "pay_…",
  "ts": "2026-11-02T00:00:00Z"
}
```

`amount` is a decimal string in the currency's major unit. It is converted to minor units exactly:

- It must match `^[0-9]+(\.[0-9]+)?$`.
- It may not have more fractional digits than the currency has; fewer are padded with zeros.
- Anything else is malformed.

## Currencies

| Currency            | Minor-unit digits | `"12.5"` in version 1 becomes |
| ------------------- | ----------------- | ----------------------------- |
| `USD`, `EUR`, `GBP` | 2                 | 1250                          |
| `JPY`               | 0                 | malformed (`"12"` becomes 12) |
| `KWD`               | 3                 | 12500                         |

Any other currency is malformed in every version.

## Event types

`payment.captured`, `payment.failed`, `refund.created`, `payout.paid`. An endpoint subscribes to a list of types, or to `"*"` for all.
