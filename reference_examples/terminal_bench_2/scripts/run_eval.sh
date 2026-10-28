#!/usr/bin/env bash
# Usage: scripts/run_eval.sh <agent_import_path> [hard|full|val|test|val10pct|val30pct|full10pct|full20pct|full30pct] [runs] [n_concurrent] [extra_harbor_flags...]
# uv run bash scripts/run_eval.sh agents.baseline_kira:AgentHarness full 1 1 -i extract-elf
#
# val / test read task_split.json, a seed-fixed random split. Search and
# development use val; test is only for the final evaluation and is never
# touched during evolution.

set -euo pipefail

AGENT_IMPORT_PATH="${1:?Usage: $0 <agent_import_path> [task_set] [runs] [n_concurrent] [extra_harbor_flags...]}"
TASK_SET="${2:-full}"
RUNS="${3:-2}"
# Default 10: at 50 the sandbox layer produced frequent write and block errors
N_CONCURRENT="${4:-10}"
shift "$(( $# < 4 ? $# : 4 ))"
EXTRA_FLAGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
# AGENT_PYTHONPATH redirects where the agent class is imported from, e.g. a
# run's code snapshot, so a candidate need not be copied into the shared agents/
export PYTHONPATH="${AGENT_PYTHONPATH:-$REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"

# Fill in variables from the repository-level .env (and from a .env next to this
# setting, if any). A value already exported in the shell wins, and empty values
# are skipped, so an unfilled line copied from .env.example cannot replace a
# valid key. Same rule as meta_harness.py.
for env_file in "$REPO_DIR/.env" "$REPO_DIR/../../.env"; do
    [[ -f "$env_file" ]] || continue
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
        key="${BASH_REMATCH[2]}"
        value="${BASH_REMATCH[3]}"
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        [[ -n "$value" && -z "${!key:-}" ]] && export "$key=$value"
    done < "$env_file"
done

# The aiohttp transport turns a task cancellation into a connection error, which
# the agent's own retry logic treats as a transient network failure and retries.
# That defeats the agent timeout entirely and the trial runs forever: seven
# trials once overran a 900-3600s limit by more than 2.5 hours. The httpx
# transport is cancellation-safe, so aiohttp is disabled by default.
export DISABLE_AIOHTTP_TRANSPORT="${DISABLE_AIOHTTP_TRANSPORT:-True}"

# LiteLLM's Azure route reads its endpoint from AZURE_API_BASE; with only
# AZURE_OPENAI_ENDPOINT set, its pre-call logging prints a TypeError traceback on
# every call (the request itself still goes to AZURE_OPENAI_ENDPOINT). Mirror it.
if [[ -n "${AZURE_OPENAI_ENDPOINT:-}" && -z "${AZURE_API_BASE:-}" ]]; then
    export AZURE_API_BASE="$AZURE_OPENAI_ENDPOINT"
fi

# Defaults to a locally served model. Set HARBOR_MODEL explicitly for anything
# else, e.g. HARBOR_MODEL=openrouter/openai/gpt-oss-120b.
MODEL="${HARBOR_MODEL:-openai/gpt-oss-120b}"
if [[ "$MODEL" == openai/* ]]; then
    export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:8000/v1}"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
fi

# Refuse to run rollouts against a Claude model unless explicitly allowed.
# Forgetting to set HARBOR_MODEL once started an expensive rollout unnoticed.
case "$MODEL" in
    anthropic/*|claude-*)
        if [[ "${ALLOW_ANTHROPIC_MODEL:-0}" != "1" ]]; then
            echo "ERROR: rollout model resolved to a Claude model ($MODEL)." >&2
            echo "  This is usually accidental (HARBOR_MODEL not set, defaults to Opus)." >&2
            echo "  Claude rollouts are far more expensive than the OpenRouter models this" >&2
            echo "  project normally uses (~\$1.5-2/rollout vs ~\$0.01-0.1)." >&2
            echo "  Set HARBOR_MODEL to the intended model, e.g.:" >&2
            echo "    HARBOR_MODEL=openrouter/openai/gpt-oss-120b $0 ..." >&2
            echo "  If you really want Claude rollouts, opt in explicitly:" >&2
            echo "    ALLOW_ANTHROPIC_MODEL=1 HARBOR_MODEL=$MODEL $0 ..." >&2
            exit 1
        fi
        ;;
esac

HARBOR_TIMEOUT_SECONDS="${HARBOR_TIMEOUT_SECONDS:-28800}"
AGENT_TIMEOUT_MULTIPLIER="${HARBOR_AGENT_TIMEOUT_MULTIPLIER:-1.0}"
# Tasks run in local docker containers.
HARBOR_ENVIRONMENT="${HARBOR_ENVIRONMENT:-docker}"

case "$HARBOR_ENVIRONMENT" in
    modal|docker) ;;
    *) echo "Unknown HARBOR_ENVIRONMENT '$HARBOR_ENVIRONMENT'. Use: docker | modal" >&2; exit 1 ;;
esac

# With AGENT_PYTHONPATH set, validate using the snapshot's own meta_harness:
# running from the repository would put it first on sys.path and hide the
# snapshot's agents.
uv run python "${AGENT_PYTHONPATH:-$REPO_DIR}/meta_harness.py" --validate-agent "$AGENT_IMPORT_PATH"

# Dataset: Terminal-Bench 2.1, pinned to a specific version for reproducibility.
# Task names are unchanged from 2.0, so task_split.json stays valid.
DATASET="${TB_DATASET:-terminal-bench/terminal-bench-2-1@6}"

# A registry row for one task went missing upstream, which made the dataset
# metadata lookup fail outright. An idempotent patch fills it in from the local
# cache; it is re-applied on every run because `uv sync` reinstalls the package.
# Guarded by existence: an older code snapshot may not carry this script.
if [[ -f "$SCRIPT_DIR/apply_harbor_registry_fix.py" ]]; then
    uv run python "$SCRIPT_DIR/apply_harbor_registry_fix.py"
fi

# Hub datasets namespace their task ids as "org/task", so a bare -i name matches
# nothing and the run dies at startup. Bare names are prefixed automatically;
# task_split.json and friends keep the bare form.
TASK_NS=""
[[ "$DATASET" == */* ]] && TASK_NS="${DATASET%%/*}/"
ns_task() {
    if [[ -n "$TASK_NS" && "$1" != */* ]]; then
        echo "${TASK_NS}$1"
    else
        echo "$1"
    fi
}

# 30-task hard subset for cheaper development and debugging loops.
HARD_TASKS=(
    bn-fit-modify cancel-async-tasks circuit-fibsqrt configure-git-webserver
    dna-assembly extract-moves-from-video feal-differential-cryptanalysis
    feal-linear-cryptanalysis fix-code-vulnerability fix-ocaml-gc
    gpt2-codegolf install-windows-3.11 llm-inference-batching-scheduler
    make-doom-for-mips make-mips-interpreter mcmc-sampling-stan
    model-extraction-relu-logits password-recovery path-tracing
    path-tracing-reverse polyglot-rust-c protein-assembly regex-chess
    sam-cell-seg sparql-university torch-pipeline-parallelism
    torch-tensor-parallelism train-fasttext video-processing write-compressor
)

TASK_FLAGS=()
case "$TASK_SET" in
    hard)
        for t in "${HARD_TASKS[@]}"; do TASK_FLAGS+=(-i "$(ns_task "$t")"); done ;;
    full)
        ;;  # no task flags: harbor runs the full dataset
    val|test|val10pct|val30pct|full10pct|full20pct|full30pct)
        # task_split.json holds the seed-fixed splits and the fixed subsets used
        # by the naive arm: full10pct (9) is a subset of full20pct (18), which is
        # a subset of full30pct (27).
        if [[ ! -f "$REPO_DIR/task_split.json" ]]; then
            echo "ERROR: $REPO_DIR/task_split.json not found (needed for task_set=$TASK_SET)" >&2
            exit 1
        fi
        while IFS= read -r t; do
            TASK_FLAGS+=(-i "$(ns_task "$t")")
        done < <(uv run python -c "import json; [print(t) for t in json.load(open('$REPO_DIR/task_split.json'))['$TASK_SET']]") ;;
    *)
        echo "Unknown task_set '$TASK_SET'. Use: hard | full | val | test | val10pct | val30pct | full10pct | full20pct | full30pct" >&2; exit 1 ;;
esac

echo "agent:       $AGENT_IMPORT_PATH"
echo "environment: $HARBOR_ENVIRONMENT"
echo "task_set:    $TASK_SET"
echo "model:       $MODEL"
echo "concurrent:  $N_CONCURRENT"
echo "runs:        $RUNS"
echo "timeout:     ${HARBOR_TIMEOUT_SECONDS}s (agent timeout x${AGENT_TIMEOUT_MULTIPLIER})"
echo ""

TIMEOUT_CMD=()
if [[ "${HARBOR_EXTERNAL_TIMEOUT:-0}" != "1" ]]; then
    if command -v timeout >/dev/null 2>&1; then
        TIMEOUT_CMD=(timeout --signal=TERM --kill-after=60 "$HARBOR_TIMEOUT_SECONDS")
    elif command -v gtimeout >/dev/null 2>&1; then
        TIMEOUT_CMD=(gtimeout --signal=TERM --kill-after=60 "$HARBOR_TIMEOUT_SECONDS")
    fi
fi

# Apply the same namespacing to extra -i flags passed in by the caller
if [[ ${#EXTRA_FLAGS[@]} -gt 0 ]]; then
    _mapped=()
    _expect_task=0
    for _a in "${EXTRA_FLAGS[@]}"; do
        if [[ "$_expect_task" == 1 ]]; then
            _a="$(ns_task "$_a")"
            _expect_task=0
        elif [[ "$_a" == "-i" ]]; then
            _expect_task=1
        fi
        _mapped+=("$_a")
    done
    EXTRA_FLAGS=("${_mapped[@]}")
fi

CMD=(
    uv run harbor run
    --agent "$AGENT_IMPORT_PATH"
    -d "$DATASET"
    -m "$MODEL"
    -e "$HARBOR_ENVIRONMENT"
    -n "$N_CONCURRENT"
    --n-attempts "$RUNS"
    --agent-timeout-multiplier "$AGENT_TIMEOUT_MULTIPLIER"
    --ak temperature=0.7
)
if [[ "$HARBOR_ENVIRONMENT" == "docker" ]]; then
    # Per-task CPU limits fail to apply on kernels without CFS scheduler
    # support, so they are skipped and the tasks share the host's cores. Keep
    # concurrency modest.
    CMD+=(--cpus ignore)
fi
if [[ ${#TASK_FLAGS[@]} -gt 0 ]]; then
    CMD+=("${TASK_FLAGS[@]}")
fi
if [[ ${#EXTRA_FLAGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_FLAGS[@]}")
fi

if [[ ${#TIMEOUT_CMD[@]} -gt 0 ]]; then
    "${TIMEOUT_CMD[@]}" "${CMD[@]}"
else
    "${CMD[@]}"
fi
