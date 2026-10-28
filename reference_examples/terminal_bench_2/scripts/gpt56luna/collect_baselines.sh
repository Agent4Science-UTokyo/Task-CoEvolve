#!/usr/bin/env bash
# Collect the Phase 0 baselines for GPT-5.6-Luna: both starting harnesses over
# all 89 tasks, into jobs/gpt56luna_tb21/.
#
#   bash scripts/gpt56luna/collect_baselines.sh
#
# Run this once before any search. Every search reads these rollouts as its
# Phase 0 instead of re-evaluating the baselines, and scripts/eval_winner.sh
# compares the winner against them. Without them the first iteration of every
# search pays for both baselines again.
#
# TRIALS defaults to 1, matching the searches: Phase 0 only reuses a cached
# baseline whose trial count equals the search's, so a 2-trial collection would
# be rejected as stale and every search would re-run both baselines.
#
# The baseline row of the paper's table is measured with 2 trials, collected
# separately:
#   TRIALS=2 bash scripts/gpt56luna/collect_baselines.sh
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

export HARBOR_MODEL="azure/gpt-5.6-luna-2026-07-09-azure"
export DISABLE_AIOHTTP_TRANSPORT="${DISABLE_AIOHTTP_TRANSPORT:-True}"
TRIALS="${TRIALS:-1}"
N_CONCURRENT="${N_CONCURRENT:-10}"
JOBS_DIR="jobs/gpt56luna_tb21"

grep -qE "^AZURE_OPENAI_API_KEY=.+" ../../.env 2>/dev/null \
    || [[ -n "${AZURE_OPENAI_API_KEY:-}${AZURE_API_KEY:-}" ]] \
    || { echo "ERROR: AZURE_OPENAI_API_KEY is set neither in .env nor in the environment" >&2; exit 1; }

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
