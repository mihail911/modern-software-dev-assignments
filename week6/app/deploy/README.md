# Deploying Switchyard

Wale. Written for release 2.2.

Switchyard runs on the shared cluster in two namespaces, `switchyard-staging` and `switchyard`. There is one Deployment per service: `api`, `router` and `worker`. The `receiver` in `services/` is a test stand-in and is never deployed.

## Layout

```
deploy/
  k8s/                    base manifests (kustomize)
    overlays/staging/
    overlays/production/
  helm/switchyard/        the chart used for releases up to 2.1
  scripts/rollout.sh      roll a release out to one environment
  scripts/drain-workers.sh  move workers off a node before maintenance
```

Releases up to 2.1 were installed with the Helm chart. From 2.2 a release is a kustomize build of an overlay. The two are not kept in step; `k8s/` is what is running.

## Rolling out a release

```
deploy/scripts/rollout.sh staging 2.2.0
deploy/scripts/rollout.sh production 2.2.0
```

The script sets the image tag in the overlay, applies it and waits for each Deployment in turn. Commit the overlay afterwards so the repository says what is running.

Services are rolled in this order:

1. **api.** Three replicas in production, ordinary rolling update. Nothing to watch.
2. **router.** One replica, `strategy: Recreate`, because the queue is a SQLite file on a volume and two routers must never open it. The router is down for the few seconds between pods. `api` keeps accepting events and publishes them when the router is back; workers get connection errors from the queue and try again.
3. **worker.** Rolling restart, one pod at a time (`maxUnavailable: 1`, `maxSurge: 0`).

A new bus message version goes out router first: deploy the router that reads it, then the api that writes it. `rollout.sh` has `--only` for that.

## Rolling restart of workers

A worker holds one delivery at a time under a lease of 60 seconds.

When a worker pod is told to stop it finishes the request it has in flight, which takes at most 5 seconds, reports the result and exits. `terminationGracePeriodSeconds` is 45, which leaves a wide margin. A worker that is killed without finishing leaves its delivery leased; the router hands it out again when the lease expires, so the worst case for one delivery is a delay of a minute.

Because of that, wait for each new worker pod to be Ready before the next is stopped. The Deployment's strategy does this by itself. Do not `kubectl delete pod` several workers at once.

## Node maintenance

Run `deploy/scripts/drain-workers.sh <node>` before a node is drained. It cordons the node and moves the workers off it one at a time, the same as a rolling restart. The router has to be moved by hand (`kubectl rollout restart deployment/router`), in a quiet period.

## Configuration

All three services read their settings from the `switchyard-config` ConfigMap (`k8s/configmap.yaml`). The ones that matter:

| Key | Value | Meaning |
|---|---|---|
| `LEASE_SECONDS` | `60` | How long a worker holds a delivery |
| `MAX_ATTEMPTS` | `6` | Attempts before a delivery is dead |
| `RETRY_SCHEDULE` | `30,120,600,3600,21600` | Waits between attempts, in seconds |
| `WORKER_CONCURRENCY` | `1` | Deliveries a worker holds at once |

Signing secrets are not configuration. They belong to endpoints and live in the router's database.

Each service keeps its state under `/var/lib/switchyard` on a PersistentVolumeClaim: `api-outbox`, `router-queue` and `worker-state`. The claims are created once per namespace by the platform storage setup and are not in this directory.

## Rolling back

`rollout.sh <env> <previous version>`. The router's database is only ever migrated forwards, so check the release notes before rolling the router back across a schema change.
