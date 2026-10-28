#!/usr/bin/env bash
# Full search: every candidate on the whole validation set.
# Runs one search and evaluates its winner on the held-out test set.
#
#   CONFIRM_COST=1 bash scripts/run_full_search.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec bash scripts/search.sh full
