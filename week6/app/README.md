# Switchyard

Switchyard delivers payment events to merchants as webhooks. A merchant registers an endpoint; when a payment is captured, fails, is refunded or is paid out, Switchyard sends the event to that endpoint as an HTTP `POST`.

## Layout

| Path | What it is |
|---|---|
| `services/api` | Accepts events and endpoint configuration. Assigns event and delivery ids. |
| `services/router` | Matches events to endpoints and queues deliveries. |
| `services/worker` | Sends deliveries and retries the ones that fail. |
| `bin/` | `up`, `down`, `replay` and `evaluate` |
| `scenarios/` | The scenarios `bin/evaluate` replays |
| `legacy/` | The release 1 dispatcher |
| `sdk/` | Libraries merchants use to verify our signatures |
| `deploy/` | Kubernetes manifests |
| `docs/` | Decision records, runbooks and postmortems |

## What merchants are promised

**At most once.** Switchyard never sends the same delivery twice. If the sender stops while a request is in flight, that delivery is closed as delivered and is not repeated. A duplicate payment notification costs a merchant more than a missing one: a missing one is picked up from the nightly settlement file, a duplicate can ship an order twice.

**Retries.** A request that fails outright (any status outside 2xx, a refused connection, or no answer within 5 seconds) is tried again after 10 seconds, then 1 minute, 5 minutes and 30 minutes. After the fifth failed attempt the delivery is marked failed and support is alerted.

**Ids.** A delivery gets its id when its event is accepted, before anything is queued, so the id a merchant sees in the dashboard is the id in the request.

**Amounts.** Amounts are decimal strings in the currency's major unit, for example `"12.50"` for USD and `"1250"` for JPY.

**Signatures.** Every request is signed with the endpoint's secret. Merchants verify the signature with one of the libraries in `sdk/`.

## Running it

```
bin/up        build and start everything on local ports
bin/down      stop it
```

`bin/up` prints the address of each process. State and logs go to `state/`.

## Checking a change

```
bin/evaluate
```

`bin/evaluate` replays every scenario in `scenarios/` against the code as it is and prints what each process did, step by step, followed by one line per scenario. A change is ready to merge when every scenario passes.

To look at one scenario on its own:

```
bin/replay scenarios/02_retry.json
```

## Getting help

`#switchyard-dev` for questions, `#switchyard-oncall` if merchants are affected.
