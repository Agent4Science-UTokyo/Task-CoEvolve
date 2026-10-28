#!/usr/bin/env bash
# Run one evolutionary search on Terminal-Bench 2.1.
#
#   bash scripts/search.sh <model> <protocol> [run_name] [iterations]
#
#   model     gpt56luna | qwen36
#   protocol  full           evaluate every candidate on all 89 tasks (reference arm)
#             naive          fixed 20% subset, same tasks every iteration
#             task-coevolve  adaptive 20% subset + full-set estimation
#
# The paper reports, for each model:
#   bash scripts/search.sh <model> full
#   bash scripts/search.sh <model> naive
#   bash scripts/search.sh <model> task-coevolve
#
# Then evaluate the winner on all 89 tasks:
#   bash scripts/eval_winner.sh <run_name>
#
# Phase 0 reuses the baseline rollouts under jobs/<model>_tb21, collected once by
# scripts/<model>/collect_baselines.sh. Without them the first iteration re-runs both
# baselines over the full task set.
#
# The search runs in the FOREGROUND and is managed by scripts/run_evolve_subset.sh,
# so the same entry point also takes status / resume / stop / log:
#   bash scripts/search.sh <model> <protocol> <run_name> <iters> -- status
#
# Environment:
#   VWS_FRACTION=<f>   subset fraction for task-coevolve (default 0.2)
#   TRIALS=<n>         trials per task during the search (default 1)
#   OPENAI_API_BASE    required for qwen36 (the local vLLM endpoint)
#   DRY_RUN=1          print what would run and exit
#   ALLOW_MISSING_BASELINES=1   run without the cached Phase 0 baselines

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

MODEL_KEY="${1:?Usage: $0 <gpt56luna|qwen36> <full|naive|task-coevolve> [run_name] [iters]}"
PROTOCOL="${2:?Usage: $0 <gpt56luna|qwen36> <full|naive|task-coevolve> [run_name] [iters]}"
RUN_NAME="${3:-}"
ITERATIONS="${4:-10}"
CMD="${CMD:-start}"

case "$MODEL_KEY" in
    gpt56luna) export HARBOR_MODEL="azure/gpt-5.6-luna-2026-07-09-azure" ;;
    qwen36)    export HARBOR_MODEL="openai/Qwen3.6-35B-A3B"
               export OPENAI_API_BASE="${OPENAI_API_BASE:?qwen36 needs OPENAI_API_BASE (the vLLM endpoint)}" ;;
    *) echo "ERROR: unknown model '$MODEL_KEY' (use gpt56luna or qwen36)" >&2; exit 1 ;;
esac

export BASELINE_JOBS_SOURCE="$REPO_DIR/jobs/${MODEL_KEY}_tb21"
export DISABLE_AIOHTTP_TRANSPORT="${DISABLE_AIOHTTP_TRANSPORT:-True}"
TRIALS="${TRIALS:-1}"
EXTRA=()

case "$PROTOCOL" in
    full)
        export EVAL_TASK_SET="full"
        RUN_NAME="${RUN_NAME:-${MODEL_KEY}_full_${ITERATIONS}iter}"
        ;;
    naive)
        export EVAL_TASK_SET="full20pct"
        RUN_NAME="${RUN_NAME:-${MODEL_KEY}_naive20pct_${ITERATIONS}iter}"
        ;;
    task-coevolve)
        export EVAL_TASK_SET="full"
        export VWS_FRACTION="${VWS_FRACTION:-0.2}"
        PCT="$(awk -v b="$VWS_FRACTION" 'BEGIN{printf "%g", b*100}')"
        RUN_NAME="${RUN_NAME:-${MODEL_KEY}_taskcoevolve${PCT}pct_${ITERATIONS}iter}"
        # Terminal-Bench 2.1 has a bimodal task pool, so Phase 2 uses the
        # anchored difference estimator (see extensions/__init__.py).
        EXTRA=(--task-selector task-coevolve
               --anchored-summary "${ANCHORED_SUMMARY:-difference}")
        ;;
    *) echo "ERROR: unknown protocol '$PROTOCOL' (use full, naive or task-coevolve)" >&2; exit 1 ;;
esac

echo "model:      $MODEL_KEY ($HARBOR_MODEL)"
echo "protocol:   $PROTOCOL  (EVAL_TASK_SET=$EVAL_TASK_SET${VWS_FRACTION:+, VWS_FRACTION=$VWS_FRACTION})"
echo "run:        $RUN_NAME"
echo "iterations: $ITERATIONS   trials: $TRIALS"
echo "baselines:  $BASELINE_JOBS_SOURCE"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "would run: run_evolve_subset.sh $CMD $RUN_NAME $ITERATIONS --trials $TRIALS" \
         "--concurrent 10 ${EXTRA[*]} --snapshot-code"
    echo "DRY_RUN=1: nothing was executed"
    exit 0
fi

# Phase 0 reuses the cached baselines only when their trial count and model
# match this search. Without them the first iteration silently re-evaluates both
# starting harnesses over all 89 tasks, once per search.
for bl in kira-baseline terminus2-baseline; do
    [[ -d "$BASELINE_JOBS_SOURCE/${bl}-t${TRIALS}" ]] && continue
    echo
    echo "ERROR: no Phase 0 baseline at $BASELINE_JOBS_SOURCE/${bl}-t${TRIALS}" >&2
    echo "       Collect them once, before any search:" >&2
    echo "         bash scripts/${MODEL_KEY}/collect_baselines.sh" >&2
    echo "       (TRIALS=$TRIALS must match; the collector defaults to 1.)" >&2
    echo "       To run anyway and pay for the baselines in this search," >&2
    echo "       set ALLOW_MISSING_BASELINES=1." >&2
    [[ "${ALLOW_MISSING_BASELINES:-0}" == "1" ]] || exit 1
    echo "  ALLOW_MISSING_BASELINES=1: continuing without them" >&2
    break
done

if [[ "$HARBOR_MODEL" == openai/* ]]; then
    want="${HARBOR_MODEL#openai/}"
    curl -sf --max-time 10 "$OPENAI_API_BASE/models" | grep -q "$want" \
        || { echo "ERROR: $OPENAI_API_BASE does not serve $want" >&2; exit 1; }
elif [[ "$HARBOR_MODEL" == azure/* ]]; then
    grep -qE "^AZURE_OPENAI_API_KEY=.+" ../../.env 2>/dev/null \
        || [[ -n "${AZURE_OPENAI_API_KEY:-}${AZURE_API_KEY:-}" ]] \
        || { echo "ERROR: AZURE_OPENAI_API_KEY is set neither in .env nor in the environment" >&2; exit 1; }
fi

exec bash "$SCRIPT_DIR/run_evolve_subset.sh" "$CMD" "$RUN_NAME" "$ITERATIONS" \
    --trials "$TRIALS" --concurrent 10 "${EXTRA[@]}" --snapshot-code
