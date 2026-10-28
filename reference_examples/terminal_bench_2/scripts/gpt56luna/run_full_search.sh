#!/usr/bin/env bash
# Reference arm for GPT-5.6-Luna: full-set search. Every candidate is evaluated
# on all 89 tasks, so there is no subset and no estimator.
#
#   bash scripts/gpt56luna/run_full_search.sh list
#   CONFIRM_COST=1 bash scripts/gpt56luna/run_full_search.sh
#
# Runs the search, then evaluates its winner on all 89 tasks.
#
# Collect the Phase 0 baselines first, once:
#   bash scripts/gpt56luna/collect_baselines.sh
#
# Environment: ITERATIONS (default 10), CONFIRM_COST=1.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODEL="gpt56luna"
ITERATIONS="${ITERATIONS:-10}"

echo "model:      $MODEL"
echo "arm:        full search (all 89 tasks per candidate)"
echo "iterations: $ITERATIONS"
[[ "${1:-run}" == "list" ]] && exit 0

[[ -d "jobs/${MODEL}_tb21" ]] || {
    echo "ERROR: jobs/${MODEL}_tb21 is missing. Collect the Phase 0 baselines first:" >&2
    echo "         bash scripts/$MODEL/collect_baselines.sh" >&2
    exit 1
}
if [[ "${CONFIRM_COST:-0}" != "1" ]]; then
    echo "ERROR: $ITERATIONS iterations of 89 tasks, plus a final run of 89 tasks." >&2
    echo "       Set CONFIRM_COST=1 to allow this." >&2
    exit 1
fi

bash scripts/search.sh "$MODEL" full "" "$ITERATIONS"
run="$(basename "$(ls -dt logs/${MODEL}_full_* | grep -v '\.out$' | head -1)")"
echo
echo "### evaluating the winner of $run"
bash scripts/eval_winner.sh "$run"
