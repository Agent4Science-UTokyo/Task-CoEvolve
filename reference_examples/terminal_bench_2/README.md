# Terminal-Bench 2.1

Task-CoEvolve on Terminal-Bench 2.1: searching for agent scaffolds, each
candidate evaluated on terminal tasks in a Docker sandbox.

## Setup

### Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Docker, running locally (every task runs in a container)
- Node.js / npm (to install the pinned Claude Code CLI)
- A Claude subscription for the proposer, which runs through the Claude Code CLI
- Access to the evaluated model: GPT-5.6-Luna on Azure OpenAI, or
  Qwen3.6-35B-A3B served with vLLM

### Install

```bash
cd reference_examples/terminal_bench_2
uv sync
bash scripts/install_claude_cli.sh    # Claude Code 2.1.220, into .claude-cli/
```

### Proposer login

The proposer uses its own Claude config directory, `~/.claude-tb2` (override with
`TB2_CLAUDE_CONFIG_DIR`), and subscription auth. Log in once, with
`ANTHROPIC_API_KEY` unset so the CLI does not fall back to API billing:

```bash
env -u ANTHROPIC_API_KEY CLAUDE_CONFIG_DIR=$HOME/.claude-tb2 ./.claude-cli/node_modules/.bin/claude
# then /login, choose the Claude subscription account, and /exit
```

Check it with:

```bash
env -u ANTHROPIC_API_KEY CLAUDE_CONFIG_DIR=$HOME/.claude-tb2 ./.claude-cli/node_modules/.bin/claude -p "Reply with OK"
```

### Model credentials

Put these in the repository-level `.env` (`cp .env.example .env`) or export them
in the shell. A value exported in the shell takes precedence, and blank lines in
`.env` are ignored.

| Model | Variables |
| --- | --- |
| GPT-5.6-Luna | `AZURE_OPENAI_API_KEY` (the key) and `AZURE_OPENAI_ENDPOINT` (`https://<resource>.openai.azure.com/`) |
| Qwen3.6-35B-A3B | `OPENAI_API_BASE`, the vLLM endpoint, passed on each command |

To serve Qwen3.6-35B-A3B as in the paper (four 48 GB GPUs):

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --served-model-name Qwen3.6-35B-A3B Qwen/Qwen3.6-35B-A3B \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --enable-prefix-caching --port 8000
```

### Smoke check

One task, one trial:

```bash
uv run bash scripts/run_eval.sh agents.baseline_kira:AgentHarness full 1 1 -i extract-elf
```

## Quick Start

Collect the baselines once per model (Phase 0, reused by every search):

```bash
bash scripts/gpt56luna/collect_baselines.sh
```

Then run a search. Each script runs 10 iterations and evaluates the winner on
all 89 tasks; append `list` to print the plan without running anything.

```bash
CONFIRM_COST=1 bash scripts/gpt56luna/run_task_coevolve_20pct.sh  # Task-CoEvolve, 18 tasks per iteration
CONFIRM_COST=1 bash scripts/gpt56luna/run_naive_20pct.sh          # naive, a fixed 18 tasks
CONFIRM_COST=1 bash scripts/gpt56luna/run_full_search.sh          # full search, all 89 tasks
```

For Qwen3.6-35B-A3B, use `scripts/qwen36/` and prefix each command with
`OPENAI_API_BASE=http://<host>:<port>/v1`.

Results are written to `logs/<run_name>/` (`winner.json`,
`evolution_summary.jsonl`) and `jobs/<run_name>/`.
