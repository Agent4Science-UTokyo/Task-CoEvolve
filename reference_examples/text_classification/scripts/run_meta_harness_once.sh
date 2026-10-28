#!/usr/bin/env bash
# One Meta-Harness search (the reference arm), no repeats. Every candidate is
# evaluated on the whole validation set, so there is no subset and no estimator.
#
#   bash scripts/run_meta_harness_once.sh list
#   CONFIRM_COST=1 bash scripts/run_meta_harness_once.sh
#
# Environment: ITERATIONS (default 20), CONFIRM_COST=1.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export ITERATIONS="${ITERATIONS:-20}"

echo "arm:        Meta-Harness (100% of the validation set per iteration)"
echo "iterations: $ITERATIONS"
echo "repeats:    1"
[[ "${1:-run}" == "list" ]] && exit 0

if [[ "${CONFIRM_COST:-0}" != "1" ]]; then
    echo "ERROR: 1 search x $ITERATIONS iterations of paid LLM calls." >&2
    echo "       Set CONFIRM_COST=1 to allow this." >&2
    exit 1
fi

bash scripts/search.sh full
