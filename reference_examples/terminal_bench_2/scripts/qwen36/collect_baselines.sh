#!/usr/bin/env bash
# Collect the Phase 0 baselines for Qwen3.6-35B-A3B: both starting harnesses
# over all 89 tasks, into jobs/qwen36_tb21/.
#
#   OPENAI_API_BASE=http://<host>:<port>/v1 bash scripts/qwen36/collect_baselines.sh
#
# Run this once before any search. Every search reads these rollouts as its
# Phase 0 instead of re-evaluating the baselines, and scripts/eval_winner.sh
# compares the winner against them.
#
# Requires a vLLM server serving Qwen3.6-35B-A3B. The runs reported in the paper
# used four NVIDIA RTX 6000 Ada GPUs (48 GB each) and this command:
#   vllm serve Qwen/Qwen3.6-35B-A3B \
#     --tensor-parallel-size 4 \
#     --max-model-len 262144 \
#     --served-model-name Qwen3.6-35B-A3B Qwen/Qwen3.6-35B-A3B \
#     --enable-auto-tool-choice --tool-call-parser qwen3_xml \
#     --reasoning-parser qwen3 \
#     --enable-prefix-caching --port 8000
#
# The endpoint is checked before anything runs: the same host has served other
# models, and silently evaluating against the wrong one is worse than failing.
#
# TRIALS defaults to 1, matching the searches: Phase 0 only reuses a cached
# baseline whose trial count equals the search's, so a 2-trial collection would
# be rejected as stale and every search would re-run both baselines.
#
# The baseline row of the paper's table is measured with 2 trials, collected
# separately:
#   TRIALS=2 bash scripts/qwen36/collect_baselines.sh
#
# N_CONCURRENT defaults to 10; higher values produced frequent sandbox errors,
# which are then scored as failures.
#
# After each job, transient environment failures are repaired with the same rule
# the search uses (see scripts/repair_job.py). Re-running this script on a job
# that already exists resumes it instead of skipping it: unfinished trials (e.g.
# from an interrupted run) are completed and the same repair is applied, so the
# command is safe to repeat until the baselines are complete.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export HARBOR_MODEL="openai/Qwen3.6-35B-A3B"
export OPENAI_API_BASE="${OPENAI_API_BASE:?set OPENAI_API_BASE to the vLLM endpoint, e.g. http://localhost:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export DISABLE_AIOHTTP_TRANSPORT="${DISABLE_AIOHTTP_TRANSPORT:-True}"
TRIALS="${TRIALS:-1}"
N_CONCURRENT="${N_CONCURRENT:-10}"
JOBS_DIR="jobs/qwen36_tb21"

if ! curl -sf --max-time 10 "$OPENAI_API_BASE/models" | grep -q "Qwen3.6-35B-A3B"; then
    echo "ERROR: $OPENAI_API_BASE does not serve Qwen3.6-35B-A3B:" >&2
    curl -sf --max-time 10 "$OPENAI_API_BASE/models" \
        | uv run python -c "import json,sys; print('  serving:', [m['id'] for m in json.load(sys.stdin)['data']])" >&2 || true
    exit 1
fi

for pair in "kira-baseline agents.baseline_kira:AgentHarness" \
            "terminus2-baseline agents.baseline_terminus2:AgentHarness"; do
    read -r name import_path <<<"$pair"
    job="$JOBS_DIR/${name}-t${TRIALS}"
    echo
    if [[ -d "$job" ]]; then
        echo "=== $name: $job exists, resuming and repairing ==="
    else
        echo "=== $name ==="
        bash scripts/run_eval.sh "$import_path" full "$TRIALS" "$N_CONCURRENT" \
            --job-name "${name}-t${TRIALS}" --jobs-dir "$JOBS_DIR"
    fi
    uv run python scripts/repair_job.py "$job" --complete || true
done

echo
echo "Baselines are in $JOBS_DIR. Searches will reuse them as Phase 0."
