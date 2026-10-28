#!/usr/bin/env bash
# Manage a long-running meta_harness evolution run on the cheaper 30-task
# `hard` subset (start/stop/status/log). Same interface as run_evolve.sh,
# but exports EVAL_TASK_SET=hard so evaluation runs 30 tasks instead of 89.
#
# Usage:
#   scripts/run_evolve_subset.sh start  [run_name] [iterations] [extra_meta_harness_flags...]
#   scripts/run_evolve_subset.sh resume [run_name] [iterations] [extra_meta_harness_flags...]
#   scripts/run_evolve_subset.sh stop   [run_name]
#   scripts/run_evolve_subset.sh status [run_name]
#   scripts/run_evolve_subset.sh log    [run_name]
#
# The run executes in the FOREGROUND: output streams to the terminal (and is
# also appended to logs/<run_name>.out via tee), and Ctrl-C stops the run.
# Closing the terminal also stops it — run inside tmux for long runs.
# `start` passes --fresh: it clears generated agents/*.py (baselines are kept)
# and the run's history files, then starts from iteration 1.
# `resume` omits --fresh and continues from evolution_summary.jsonl.
# bash scripts/run_evolve_subset.sh start hard_10iter 10

set -euo pipefail

export EVAL_TASK_SET="${EVAL_TASK_SET:-hard}"

CMD="${1:-status}"
RUN_NAME="${2:-hard_10iter}"
ITERATIONS="${3:-10}"
shift "$(( $# < 3 ? $# : 3 ))"
EXTRA_FLAGS=("$@")

# `start` appends a timestamp so each run gets fresh logs/ and jobs/ dirs.
# Use the full timestamped name (shown at startup) for resume/stop/status/log.
if [[ "$CMD" == "start" ]]; then
    RUN_NAME="${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$REPO_DIR/logs/$RUN_NAME.out"

# Matches the meta_harness.py process and its harbor children for this run only,
# so other runs (e.g. text_classification) are never touched.
PY_PATTERN="terminal_bench_2/.*meta_harness\.py .*--run-name $RUN_NAME"
HARBOR_PATTERN="terminal_bench_2/.*harbor run .*jobs/$RUN_NAME"

run_pids() {
    pgrep -f "$PY_PATTERN" 2>/dev/null || true
    pgrep -f "$HARBOR_PATTERN" 2>/dev/null || true
}

case "$CMD" in
    start|resume)
        if [[ -n "$(run_pids)" ]]; then
            echo "Run '$RUN_NAME' is already running:"
            pgrep -af "$PY_PATTERN" || true
            exit 1
        fi
        FRESH_FLAG=()
        [[ "$CMD" == "start" ]] && FRESH_FLAG=(--fresh)
        mkdir -p "$REPO_DIR/logs"
        cd "$REPO_DIR"
        echo "Running '$RUN_NAME' in the foreground (${FRESH_FLAG[0]:-resume mode})"
        echo "  log copy: $LOG_FILE"
        echo "  Ctrl-C stops the run."
        echo ""
        # Run under a pseudo-terminal (via `script`) so harbor's rich-based
        # progress bar still detects a tty and renders live, even though our
        # own stdout here is a pipe into `tee`. Without this, rich sees a
        # non-tty pipe and silently drops all per-task progress output.
        META_HARNESS_ARGS=(uv run python meta_harness.py
            --iterations "$ITERATIONS" --run-name "$RUN_NAME")
        [[ "$CMD" == "start" ]] && META_HARNESS_ARGS+=(--fresh)
        META_HARNESS_ARGS+=("${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}")
        QUOTED_CMD=""
        for arg in "${META_HARNESS_ARGS[@]}"; do
            QUOTED_CMD+="$(printf '%q ' "$arg")"
        done
        script -qec "$QUOTED_CMD" /dev/null 2>&1 | tee -a "$LOG_FILE"
        ;;

    stop)
        PIDS="$(run_pids)"
        if [[ -z "$PIDS" ]]; then
            echo "No processes found for run '$RUN_NAME'."
            exit 0
        fi
        echo "Sending SIGTERM to:"
        pgrep -af "$PY_PATTERN" || true
        pgrep -af "$HARBOR_PATTERN" || true
        # shellcheck disable=SC2086
        kill $PIDS 2>/dev/null || true
        # harbor needs time to cancel trials and tear down sandboxes
        for _ in $(seq 1 60); do
            [[ -z "$(run_pids)" ]] && { echo "Stopped."; exit 0; }
            sleep 2
        done
        echo "Still alive after 120s, sending SIGKILL."
        pkill -9 -f "$PY_PATTERN" 2>/dev/null || true
        pkill -9 -f "$HARBOR_PATTERN" 2>/dev/null || true
        echo "Stopped (forced)."
        ;;

    status)
        if [[ -n "$(run_pids)" ]]; then
            echo "Run '$RUN_NAME' is RUNNING:"
            pgrep -af "$PY_PATTERN" || true
            pgrep -af "$HARBOR_PATTERN" || true
        else
            echo "Run '$RUN_NAME' is not running."
        fi
        SUMMARY="$REPO_DIR/logs/$RUN_NAME/evolution_summary.jsonl"
        FRONTIER="$REPO_DIR/logs/$RUN_NAME/frontier_val.json"
        if [[ -f "$SUMMARY" ]]; then
            echo ""
            echo "Iterations completed: $(wc -l < "$SUMMARY")"
        fi
        if [[ -f "$FRONTIER" ]]; then
            echo "Frontier:"
            python3 -c "import json;d=json.load(open('$FRONTIER'));b=d.get('_best',{});print(f\"  {b.get('agent','?')} @ {b.get('avg_pass_rate',0):.1%}\")" 2>/dev/null || true
        fi
        if [[ -f "$LOG_FILE" ]]; then
            echo ""
            echo "Last log lines ($LOG_FILE):"
            tail -5 "$LOG_FILE"
        fi
        ;;

    log)
        exec tail -f "$LOG_FILE"
        ;;

    *)
        echo "Unknown command '$CMD'. Use: start | stop | status | log" >&2
        exit 1
        ;;
esac
