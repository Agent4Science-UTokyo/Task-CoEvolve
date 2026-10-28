#!/usr/bin/env bash
# Task-CoEvolve: 20% of the validation set per iteration, re-selected each time,
# with the full-set accuracy estimated by the Hajek estimator.
# Runs one search and evaluates its winner on the held-out test set.
#
#   CONFIRM_COST=1 bash scripts/run_task_coevolve_20pct.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec bash scripts/search.sh hajek 0.20
