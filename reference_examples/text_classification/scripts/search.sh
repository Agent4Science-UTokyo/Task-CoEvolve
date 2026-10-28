#!/usr/bin/env bash
# Run one search, select its winner, and evaluate the winner on the test set.
#
#   CONFIRM_COST=1 bash scripts/search.sh <protocol> [budget]
#
#   protocol  full        every candidate on the whole validation set
#             naive       a fixed subset of size <budget>, no estimator
#             hajek       Task-CoEvolve, Hajek estimator (Eq. 3)
#             difference  Task-CoEvolve, anchored difference estimator (Eq. 4)
#   budget    fraction of the validation set per iteration, e.g. 0.20
#
# The winner is the highest est_full_val (Task-CoEvolve) or avg_val (full,
# naive), ties to the earliest iteration. ITERATIONS defaults to 20; DRY_RUN=1
# prints the command without running it.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROTOCOL="${1:?Usage: $0 <full|naive|hajek|difference> [budget]}"
BUDGET="${2:-}"
ITERATIONS="${ITERATIONS:-20}"

if [[ "$PROTOCOL" != "full" ]]; then
    : "${BUDGET:?Usage: $0 $PROTOCOL <budget>   (e.g. 0.20)}"
    PCT="$(awk -v b="$BUDGET" 'BEGIN{printf "%g", b*100}')"
fi
case "$PROTOCOL" in
    full)
        # No fraction flags: --search-fraction defaults to 1.0, so every
        # candidate is evaluated on the whole validation set.
        SELECTOR_ARGS=(--skip-top3-full)
        SCORE_KEY="avg_val"
        BUDGET="1.0"
        RUN_NAME="$(date +%Y%m%d_%H%M%S)_it${ITERATIONS}_full"
        ;;
    naive)
        SELECTOR_ARGS=(--search-fraction "$BUDGET" --skip-top3-full)
        SCORE_KEY="avg_val"
        RUN_NAME="$(date +%Y%m%d_%H%M%S)_it${ITERATIONS}_naive_${PCT}pct"
        ;;
    hajek|difference)
        SELECTOR_ARGS=(--sample-selector task-coevolve --anchored-summary "$PROTOCOL")
        SCORE_KEY="est_full_val"
        RUN_NAME="$(date +%Y%m%d_%H%M%S)_it${ITERATIONS}_task-coevolve_${PCT}pct_${PROTOCOL}"
        export VWS_FRACTION="$BUDGET"
        ;;
    *)
        echo "ERROR: unknown protocol '$PROTOCOL' (use full, naive, hajek or difference)" >&2
        exit 1
        ;;
esac

echo "protocol:   $PROTOCOL"
echo "budget:     $BUDGET"
echo "iterations: $ITERATIONS"
echo "run:        $RUN_NAME"
echo "selection:  max $SCORE_KEY, ties to the earliest iteration"
echo "command:    ${VWS_FRACTION:+VWS_FRACTION=$VWS_FRACTION }uv run python meta_harness.py" \
     "--iterations $ITERATIONS --fresh ${SELECTOR_ARGS[*]} --snapshot-code --run-name $RUN_NAME"
echo
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: nothing was executed"
    exit 0
fi
if [[ "${CONFIRM_COST:-0}" != "1" ]]; then
    echo "ERROR: this runs $ITERATIONS iterations of paid LLM calls. Set CONFIRM_COST=1." >&2
    exit 1
fi

uv run python meta_harness.py --iterations "$ITERATIONS" --fresh \
    "${SELECTOR_ARGS[@]}" --snapshot-code --run-name "$RUN_NAME"

WINNER=$(python3 - "$RUN_NAME" "$SCORE_KEY" <<'PYEOF'
import json, sys
run, key = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(f"logs/{run}/evolution_summary.jsonl") if l.strip()]
rows = [r for r in rows if r.get(key) is not None]
if not rows:
    sys.exit(f"ERROR: no {key} rows found (run incomplete?)")
best = max(r[key] for r in rows)
tied = sorted([r for r in rows if r[key] == best], key=lambda r: r["iteration"])
w = tied[0]
print(w["system"])
print(f"winner: {w['system']} (iteration {w['iteration']}, {key} {best}, "
      f"{len(tied)} tied)", file=sys.stderr)
PYEOF
)
echo "selected winner: $WINNER"

uv run bash scripts/eval_extra_candidate.sh "$RUN_NAME" "$WINNER"
