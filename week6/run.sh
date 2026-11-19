#!/usr/bin/env bash
# Entry point. The grader runs `./run.sh task.md` from this directory, with no
# terminal attached and nobody to answer questions. When this script exits,
# app/services/api, app/services/router and app/services/worker are scored,
# and so is out/impact.json.
#
# This is the baseline: one Claude Code call on the medium model with default
# settings. Change anything you like.
#
# --output-format stream-json --verbose prints every turn as it happens, so
# practice.py's agent.log has the whole session even if the run is stopped at
# the time limit. It changes only what is printed, not what is sent to the model.
set -euo pipefail

TASK_FILE="${1:?usage: ./run.sh <task-file>}"
cd "$(dirname "$0")"

claude -p "$(cat "$TASK_FILE")" \
  --model claude-sonnet-5-5 \
  --effort high \
  --permission-mode bypassPermissions \
  --output-format stream-json --verbose
