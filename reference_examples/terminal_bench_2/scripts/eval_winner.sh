#!/usr/bin/env bash
# Evaluate the winner of a finished search on all 89 tasks.
#
# The winner, the model and the baseline job are all derived from the run, so
# the only argument is the run name. The six cells of the paper's table are all
# reproduced with this one script.
#
#   bash scripts/eval_winner.sh <run_name>
#   bash scripts/eval_winner.sh <run_name> show    # print what it would do
#
# Winner selection (paper Section 3.3, Phase 3): the highest est_full_rate if
# the run recorded one, otherwise the highest raw avg_pass_rate, with exact ties
# broken towards the earliest iteration. The search writes that decision to
# logs/<run>/winner.json; if the file is missing the rule is applied again to
# evolution_summary.jsonl.
#
# After the evaluation, scripts/repair_job.py is applied automatically. The
# final-evaluation path does not go through meta_harness.harbor_run(), so
# without this the same retry rule the search uses (tmux startup failures,
# keyring exhaustion, setup timeouts, InternalServerError) would not be applied
# and an infrastructure failure would be recorded as a reward of 0.
# Set NO_REPAIR=1 to skip it.
#
# Environment:
#   TRIALS=<n>        trials per task (default 1, as in the search)
#   SOURCE_JOB=<dir>  Phase 0 baseline job (default: inferred from the run name)
#   N_CONCURRENT=<n>  concurrent trials (default 10)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_DIR="$PWD"

RUN_NAME="${1:?Usage: $0 <run_name> [show]}"
CMD="${2:-run}"
LOG_DIR="logs/$RUN_NAME"
SNAP="$LOG_DIR/code/terminal_bench_2"

[[ -d "$LOG_DIR" ]] || { echo "ERROR: $LOG_DIR does not exist" >&2; exit 1; }

# --- determine the winner ---------------------------------------------------
read -r AGENT IMPORT_PATH ITER METHOD < <(uv run python - "$LOG_DIR" <<'PYEOF'
import json, os, sys
log_dir = sys.argv[1]
w = os.path.join(log_dir, "winner.json")
if os.path.exists(w):
    d = json.load(open(w))
    print(d["winner"], d.get("import_path") or f"agents.{d['winner']}:AgentHarness",
          d.get("iteration", "?"), d.get("method", "winner.json"))
    raise SystemExit
rows = [json.loads(l) for l in open(os.path.join(log_dir, "evolution_summary.jsonl")) if l.strip()]
key = "est_full_rate" if any(r.get("est_full_rate") is not None for r in rows) else "avg_pass_rate"
scored = [(r[key], r) for r in rows if r.get(key) is not None]
if not scored:
    raise SystemExit("ERROR: no evaluated candidates found")
best = max(s for s, _ in scored)
r = sorted([x for s, x in scored if s == best], key=lambda x: x["iteration"])[0]
print(r["agent"], r.get("import_path") or f"agents.{r['agent']}:AgentHarness",
      r["iteration"], key + " (recomputed)")
PYEOF
)

# --- take the model from the run's job config -------------------------------
MODEL="$(uv run python - "$RUN_NAME" <<'PYEOF'
import glob, json, sys
for f in sorted(glob.glob(f"jobs/{sys.argv[1]}/*/config.json")):
    try:
        print(json.load(open(f))["agents"][0]["model_name"]); break
    except Exception:
        continue
else:
    raise SystemExit("ERROR: cannot determine the model from jobs/<run>/*/config.json")
PYEOF
)"

export HARBOR_MODEL="$MODEL"
export RUN_NAME
export TRIALS="${TRIALS:-1}"
export N_CONCURRENT="${N_CONCURRENT:-10}"
export DISABLE_AIOHTTP_TRANSPORT="${DISABLE_AIOHTTP_TRANSPORT:-True}"

# Infer the baseline job from the run-name prefix (gpt56luna_... -> jobs/gpt56luna_tb21)
if [[ -z "${SOURCE_JOB:-}" ]]; then
    SOURCE_JOB="jobs/${RUN_NAME%%_*}_tb21"
fi
export SOURCE_JOB

JOB="jobs/$RUN_NAME/final-${AGENT}-t${TRIALS}"

echo "run:     $RUN_NAME"
echo "winner:  $AGENT  (iteration $ITER, $METHOD)"
echo "model:   $MODEL"
echo "trials:  $TRIALS"
echo "output:  $JOB"
[[ "$CMD" == "show" ]] && exit 0

if [[ -d "$JOB" ]]; then
    echo
    echo "ERROR: $JOB already exists. harbor appends trials to an existing job" >&2
    echo "       directory, which would leave the tasks with unequal trial counts." >&2
    echo "       To continue an interrupted job:" >&2
    echo "         uv run python scripts/repair_job.py $JOB --complete" >&2
    echo "       To start over, move the directory aside first." >&2
    exit 1
fi

# For a locally served model, check the endpoint really serves it: the same
# host has served other models, and hitting the wrong one silently is worse
# than failing here.
if [[ "$MODEL" == openai/* ]]; then
    : "${OPENAI_API_BASE:?openai/* models need OPENAI_API_BASE}"
    want="${MODEL#openai/}"
    curl -sf --max-time 10 "$OPENAI_API_BASE/models" | grep -q "$want" \
        || { echo "ERROR: $OPENAI_API_BASE does not serve $want" >&2; exit 1; }
    echo "vLLM:    $OPENAI_API_BASE ($want)"
elif [[ "$MODEL" == azure/* ]]; then
    grep -qE "^AZURE_OPENAI_API_KEY=.+" ../../.env 2>/dev/null \
        || [[ -n "${AZURE_OPENAI_API_KEY:-}${AZURE_API_KEY:-}" ]] \
        || { echo "ERROR: AZURE_OPENAI_API_KEY is set neither in .env nor in the environment" >&2; exit 1; }
fi

echo
bash scripts/baseline_eval.sh final "$IMPORT_PATH" "$AGENT"

if [[ "${NO_REPAIR:-0}" == "1" ]]; then
    echo; echo "NO_REPAIR=1: skipped the transient-failure repair"
else
    echo
    echo "=== repairing transient failures (same rule as the search) ==="
    SNAP_ARG=(); [[ -d "$SNAP" ]] && SNAP_ARG=(--snapshot "$SNAP")
    uv run python scripts/repair_job.py "$JOB" "${SNAP_ARG[@]}"
fi

echo
echo "=== score over all 89 tasks ==="
uv run python - "$JOB" "$AGENT" <<'PYEOF'
import collections, json, os, sys
job, agent = sys.argv[1], sys.argv[2]
per = collections.defaultdict(list)
for d in os.listdir(job):
    p = os.path.join(job, d)
    if "__" not in d or not os.path.isdir(p):
        continue
    reward = 0.0
    try:
        r = json.load(open(os.path.join(p, "result.json")))
        v = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        reward = float(v) if v is not None else 0.0
    except Exception:
        pass
    per[d.rsplit("__", 1)[0]].append(1.0 if reward > 0 else 0.0)
rates = {t: sum(v) / len(v) for t, v in per.items()}
dist = dict(sorted(collections.Counter(len(v) for v in per.values()).items()))
print(f"  {agent}: {100 * sum(rates.values()) / len(rates):.1f}%  "
      f"({len(rates)}/89 tasks, trial {dist})")
if len(rates) != 89:
    print(f"  WARN: fewer than 89 tasks. Continue with "
          f"`uv run python scripts/repair_job.py {job} --complete`")
PYEOF
