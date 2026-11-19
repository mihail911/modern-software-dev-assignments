#!/usr/bin/env sh
set -ex

ruff check --fix .
ruff format .
