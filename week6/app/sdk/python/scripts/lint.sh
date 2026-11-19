#!/usr/bin/env sh
set -ex

mypy standardwebhooks tests
ruff check .
ruff format --check .
ty check .
