#!/usr/bin/env bash
# Roll a release out to one environment.
#
#   rollout.sh <staging|production> <version> [--only api|router|worker]
#
# Sets the image tag in the overlay, applies it, and waits for each Deployment
# in the order api, router, worker. Commit the overlay afterwards.
set -euo pipefail

usage() {
  sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

[ $# -ge 2 ] || usage
env_name="$1"
version="$2"
only=""
if [ "${3:-}" = "--only" ]; then
  only="${4:-}"
  [ -n "$only" ] || usage
fi

case "$env_name" in
  staging) namespace="switchyard-staging" ;;
  production) namespace="switchyard" ;;
  *) usage ;;
esac

here="$(cd "$(dirname "$0")" && pwd)"
overlay="$here/../k8s/overlays/$env_name"
registry="registry.switchyard.example/switchyard"

services="api router worker"
[ -z "$only" ] || services="$only"

if [ "$env_name" = "production" ]; then
  printf 'Roll %s to production at %s? [y/N] ' "$services" "$version"
  read -r answer
  [ "$answer" = "y" ] || exit 1
fi

cd "$overlay"
for service in $services; do
  kustomize edit set image "$registry/$service=$registry/$service:$version"
done

kustomize build . | kubectl apply -n "$namespace" -f -

for service in $services; do
  case "$service" in
    api) timeout=180s ;;
    router) timeout=120s ;;
    worker) timeout=600s ;;
    *) usage ;;
  esac
  echo "waiting for $service"
  kubectl rollout status -n "$namespace" "deployment/$service" --timeout="$timeout"
done

echo "$env_name is at $version ($services). Commit $overlay/kustomization.yaml."
