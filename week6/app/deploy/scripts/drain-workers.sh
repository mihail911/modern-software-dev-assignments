#!/usr/bin/env bash
# Move workers off a node before it is drained.
#
#   drain-workers.sh <node> [namespace]
#
# Cordons the node, then deletes the worker pods on it one at a time. After
# each one it waits for the Deployment to be back at full strength, and then
# for one lease length, so that anything the old pod was holding has been
# handed out again before the next pod goes.
set -euo pipefail

node="${1:?usage: drain-workers.sh <node> [namespace]}"
namespace="${2:-switchyard}"
lease_seconds="${LEASE_SECONDS:-120}"

kubectl cordon "$node"

pods="$(kubectl get pods -n "$namespace" \
  -l app.kubernetes.io/name=worker \
  --field-selector "spec.nodeName=$node" \
  -o jsonpath='{.items[*].metadata.name}')"

if [ -z "$pods" ]; then
  echo "no workers on $node"
  exit 0
fi

for pod in $pods; do
  echo "moving $pod"
  kubectl delete pod -n "$namespace" "$pod" --wait=true
  kubectl rollout status -n "$namespace" deployment/worker --timeout=300s
  echo "waiting ${lease_seconds}s"
  sleep "$lease_seconds"
done

echo "workers are off $node. The router is not moved by this script."
