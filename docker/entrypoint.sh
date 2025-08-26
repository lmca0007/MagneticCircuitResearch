#!/usr/bin/env bash
set -e

# Default behavior: start an interactive bash
if [ "$#" -eq 0 ]; then
  exec bash
else
  exec "$@"
fi