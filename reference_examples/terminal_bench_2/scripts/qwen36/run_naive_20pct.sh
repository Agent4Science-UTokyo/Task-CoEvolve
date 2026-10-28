#!/usr/bin/env bash
# Naive search for Qwen3.6-35B-A3B at the 20% budget.
#
#   bash scripts/qwen36/run_naive_20pct.sh list
#   OPENAI_API_BASE=http://<host>:<port>/v1 CONFIRM_COST=1 bash scripts/qwen36/run_naive_20pct.sh
#
# Naive subset search: every candidate is evaluated on the same fixed 18 of the
# 89 tasks, and the winner is chosen by its raw subset score (avg_pass_rate).
#
# Runs the search, then evaluates its winner on all 89 tasks.
#
# Collect the Phase 0 baselines first, once:
#   OPENAI_API_BASE=http://<host>:<port>/v1 bash scripts/qwen36/collect_baselines.sh
#
# Environment: ITERATIONS (default 10), CONFIRM_COST=1.
# OPENAI_API_BASE is required (the vLLM endpoint).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MODEL="qwen36"
ITERATIONS="${ITERATIONS:-10}"

echo "model:      $MODEL"
echo "arm:        Naive (18 of 89 tasks per candidate)"
echo "iterations: $ITERATIONS"
[[ "${1:-run}" == "list" ]] && exit 0

[[ -d "jobs/${MODEL}_tb21" ]] || {
    echo "ERROR: jobs/${MODEL}_tb21 is missing. Collect the Phase 0 baselines first:" >&2
    echo "         bash scripts/$MODEL/collect_baselines.sh" >&2
    exit 1
}

if [[ "${CONFIRM_COST:-0}" != "1" ]]; then
    echo "ERROR: $ITERATIONS iterations of 18 tasks, plus a final run of 89 tasks." >&2
    echo "       Set CONFIRM_COST=1 to allow this." >&2
    exit 1
fi

bash scripts/search.sh "$MODEL" naive "" "$ITERATIONS"
run="$(basename "$(ls -dt logs/${MODEL}_naive20pct_* | grep -v '\.out$' | head -1)")"
echo
echo "### evaluating the winner of $run"
bash scripts/eval_winner.sh "$run"
