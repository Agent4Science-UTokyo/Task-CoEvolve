#!/usr/bin/env bash
# One Task-CoEvolve search (adaptive subset + full-set estimation), no repeats.
#
#   bash scripts/run_task_coevolve_once.sh list
#   CONFIRM_COST=1 bash scripts/run_task_coevolve_once.sh
#
# Environment: BUDGET (default 0.20), ESTIMATOR (hajek or difference, default
# hajek), ITERATIONS (default 20), CONFIRM_COST=1.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BUDGET="${BUDGET:-0.20}"
ESTIMATOR="${ESTIMATOR:-hajek}"
export ITERATIONS="${ITERATIONS:-20}"

echo "arm:        Task-CoEvolve ($ESTIMATOR)"
echo "budget:     $BUDGET"
echo "iterations: $ITERATIONS"
echo "repeats:    1"
[[ "${1:-run}" == "list" ]] && exit 0

if [[ "${CONFIRM_COST:-0}" != "1" ]]; then
    echo "ERROR: 1 search x $ITERATIONS iterations of paid LLM calls." >&2
    echo "       Set CONFIRM_COST=1 to allow this." >&2
    exit 1
fi

bash scripts/search.sh "$ESTIMATOR" "$BUDGET"
