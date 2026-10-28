#!/usr/bin/env bash
# Naive: the same fixed 20% of the validation set every iteration, no estimator.
# Runs one search and evaluates its winner on the held-out test set.
#
#   CONFIRM_COST=1 bash scripts/run_naive_20pct.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec bash scripts/search.sh naive 0.20
