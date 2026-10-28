#!/usr/bin/env bash
# One Task-CoEvolve search at the 7% budget, no repeats.
#
#   bash scripts/run_task_coevolve_once_7pct.sh list
#   CONFIRM_COST=1 bash scripts/run_task_coevolve_once_7pct.sh
#
# Environment: ESTIMATOR (hajek or difference, default hajek), ITERATIONS
# (default 20), CONFIRM_COST=1.

set -euo pipefail
BUDGET="0.07" exec bash "$(dirname "${BASH_SOURCE[0]}")/run_task_coevolve_once.sh" "$@"
