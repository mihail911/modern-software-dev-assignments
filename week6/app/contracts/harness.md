# Operations interface

Every service exposes the same small interface for operations and for `bin/replay`. A change to a service must keep it working.

## Environment

| Variable               | Meaning                                                                   | Default                 |
| ---------------------- | ------------------------------------------------------------------------- | ----------------------- |
| `SWITCHYARD_STATE`     | Directory for this deployment's state and logs                            | `./state`               |
| `SWITCHYARD_PORT_BASE` | `api` listens on base+1, `router` on +2, `worker` on +3, `receiver` on +4 | `7100`                  |
| `SWITCHYARD_CLOCK`     | The clock's value at start, in whole seconds                              | unset: the system clock |

Services find each other on `127.0.0.1` at those ports.

## Endpoints

| Request                                                 | Effect                                                                                                                     |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `GET /__admin/health`                                   | `200` with `{"service": …, "release": …}` once the service is ready                                                        |
| `POST /__admin/clock` with `{"now": 1793577600}`        | Sets the service's clock, in whole seconds. Until it is set the service uses the system clock.                             |
| `POST /__admin/tick`                                    | Does the service's pending background work once and replies `{"did": n}`, the number of things it changed. `0` means idle. |
| `POST /__admin/crash_at` with `{"point": 4, "hits": 1}` | Arms a crash point: the process exits with status 86 the `hits`-th time it reaches that point.                             |
| `GET /__admin/state`                                    | The service's state as JSON                                                                                                |

Services do no background work on their own. Work happens in request handlers and in `tick`.

## State

`GET /__admin/state` answers `200` with a JSON object on every service. `router`'s has at least these fields, with these meanings:

| Field          | Meaning                                                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `next_wake`    | The earliest time at which a queued delivery of an enabled endpoint is due or an open lease expires; `null` when there is none                   |
| `endpoints[]`  | `id`, `merchant_id`, `enabled`, `version`                                                                                                        |
| `deliveries[]` | One per delivery, oldest first: `id`, `event_id`, `endpoint_id`, `resend_of` (`null` unless it is a resend), `state`, `attempts`, `available_at` |

- `state` is `queued`, `leased`, `delivered` or `dead`.
- `attempts` is the number of counted attempts: those reported `failed` or `unknown`. The attempt that delivers is not counted.

`bin/replay` uses `next_wake` to move the clock forward while it drains.

## Crash points

Each service has numbered crash points: `api` 1 to 4, `router` 1 to 4, `worker` 1 to 9. Each sits after a step that changes durable state, and is marked in the code by a call to the service's `crash` function.

A change to a service must leave every crash point at the same step, under the same number.

## Logs

Each service appends to `$SWITCHYARD_STATE/logs/`. Formats are in `evidence.md`.
