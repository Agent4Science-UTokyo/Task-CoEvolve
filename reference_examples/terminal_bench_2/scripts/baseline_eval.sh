#!/usr/bin/env bash
# Evaluate a selected harness on all 89 tasks and print it next to the baselines.
#
#   bash scripts/baseline_eval.sh final <agent_import_path> [label]
#
# Called by scripts/eval_winner.sh; use that instead, since it also derives the
# winner and the model from the run and applies the retry rule afterwards.
#
# The baseline scores come from SOURCE_JOB, the Phase 0 rollouts collected once
# per model, so no extra rollouts are needed for the comparison.
#
# The candidate is imported from the run's code snapshot rather than copied into
# the shared agents/ directory, so a stray file cannot leak into another run. If
# the candidate was archived to logs/<run>/agents/ at the end of the run, it is
# restored into the snapshot first.
#
# Scoring follows the benchmark's own semantics: the verifier reward counts even
# when the trial timed out, and a trial with no recorded reward counts as 0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

export RUN_NAME="${RUN_NAME:-gpt_oss_val}"
export HARBOR_MODEL="${HARBOR_MODEL:-openrouter/openai/gpt-oss-120b}"
TRIALS="${TRIALS:-2}"
N_CONCURRENT="${N_CONCURRENT:-10}"
# Phase 0 baseline rollouts; collected per model by scripts/collect_baselines_*.sh
SOURCE_JOB="${SOURCE_JOB:-jobs/gpt_oss_tb21}"
# Task set to score over (a key of task_split.json)
SUBSET="${SUBSET:-val}"

MODE="${1:-val}"

# Average the per-task scores of a job directory over a named task set
score_job () {  # <job_dir> <val|test>
    uv run python - "$1" "$2" <<'PYEOF'
import json, sys, collections
from pathlib import Path
job_dir, subset = Path(sys.argv[1]), sys.argv[2]
split = json.load(open("task_split.json"))  # cwd = REPO_DIR
tasks = set(split[subset])
per = collections.defaultdict(list)
for t in sorted(job_dir.iterdir()):
    if not t.is_dir() or "__" not in t.name:
        continue
    task = t.name.rsplit("__", 1)[0]
    if task not in tasks:
        continue
    reward = 0.0
    try:
        r = json.loads((t / "result.json").read_text())
        rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        reward = float(rew) if rew is not None else 0.0
    except Exception:
        pass
    per[task].append(reward)
rates = {t: sum(v) / len(v) for t, v in per.items()}
missing = tasks - set(rates)
avg = 100 * sum(rates.values()) / len(rates) if rates else 0.0
print(f"{avg:.1f}% ({len(rates)}/{len(tasks)} tasks{', MISSING: ' + ', '.join(sorted(missing)) if missing else ''})")
PYEOF
}

cd "$REPO_DIR"

if [[ "$MODE" == "final" ]]; then
    WINNER_IMPORT="${2:?Usage: $0 $MODE <agent_import_path> [label]}"
    LABEL="${3:-winner}"
    EVAL_SET="full"
    JOB_NAME="final-${LABEL}-t${TRIALS}"
    echo "baseline_eval: evaluating $WINNER_IMPORT on all 89 tasks -> jobs/$RUN_NAME/$JOB_NAME"

    # Import the candidate from the run's code snapshot. Restore it from the
    # run's archive first if the end of the run moved it there.
    SNAP="logs/$RUN_NAME/code/terminal_bench_2"
    if [[ -d "$SNAP" ]]; then
        MODULE_REL="${WINNER_IMPORT%%:*}"          # agents.smart_truncate
        MODULE_REL="${MODULE_REL//./\/}.py"        # agents/smart_truncate.py
        base="$(basename "$MODULE_REL" .py)"
        if [[ ! -f "$SNAP/$MODULE_REL" && -f "logs/$RUN_NAME/agents/$base.py" ]]; then
            cp "logs/$RUN_NAME/agents/$base.py" "$SNAP/$MODULE_REL"
            echo "  restored $base.py: run archive -> snapshot"
        fi
        for t in "logs/$RUN_NAME/prompt-templates/"*.txt; do
            [[ -f "$t" && ! -f "$SNAP/prompt-templates/$(basename "$t")" ]] \
                && cp "$t" "$SNAP/prompt-templates/"
        done
        export AGENT_PYTHONPATH="$(cd "$SNAP" && pwd)"
        echo "  agent import: snapshot ($AGENT_PYTHONPATH)"
    else
        echo "  WARN: no snapshot for this run; importing $WINNER_IMPORT from agents/"
    fi
    echo
    uv run bash "$SCRIPT_DIR/run_eval.sh" \
        "$WINNER_IMPORT" "$EVAL_SET" "$TRIALS" "$N_CONCURRENT" \
        --job-name "$JOB_NAME" --jobs-dir "jobs/$RUN_NAME"

    echo
    echo "=== comparison over all $EVAL_SET tasks ==="
    echo "  ${LABEL}: $(score_job "jobs/$RUN_NAME/$JOB_NAME" "$EVAL_SET")"
    for agent in kira-baseline terminus2-baseline; do
        src="$SOURCE_JOB/${agent}-t${TRIALS}"
        [[ -d "$src" ]] && echo "  $agent (from $SOURCE_JOB): $(score_job "$src" "$EVAL_SET")"
    done

else
    echo "Unknown mode '$MODE'. Use: final <agent_import_path> [label]" >&2
    exit 1
fi
