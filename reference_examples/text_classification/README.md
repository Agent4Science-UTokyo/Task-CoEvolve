# Online Text Classification

Task-CoEvolve on online text classification: searching for memory systems, each
candidate evaluated on the three datasets in `config.yaml` (included in `data/`).

## Setup

### Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Node.js / npm (to install the pinned Claude Code CLI)
- A Claude subscription for the proposer, which runs through the Claude Code CLI
- An [OpenRouter](https://openrouter.ai/) API key for the evaluated model,
  `openrouter/openai/gpt-oss-120b`

### Install

```bash
cd reference_examples/text_classification
uv sync
bash scripts/install_claude_cli.sh    # Claude Code 2.1.56, into .claude-cli/
```

### Proposer login

The proposer uses the default Claude config directory (`~/.claude`) and
subscription auth. Log in once, with `ANTHROPIC_API_KEY` unset so the CLI does
not fall back to API billing:

```bash
env -u ANTHROPIC_API_KEY ./.claude-cli/node_modules/.bin/claude
# then /login, choose the Claude subscription account, and /exit
```

### Model credentials

Put `OPENROUTER_API_KEY` in the repository-level `.env` (`cp .env.example .env`)
or export it in the shell.

### Smoke check

One iteration:

```bash
uv run python meta_harness.py --iterations 1
```

## Quick Start

Each script runs one search of 20 iterations and evaluates its winner on the
held-out test set. Set `DRY_RUN=1` to print the command without running it.

```bash
CONFIRM_COST=1 bash scripts/run_task_coevolve_20pct.sh  # Task-CoEvolve, 20% of val per iteration
CONFIRM_COST=1 bash scripts/run_naive_20pct.sh          # naive, a fixed 20% of val
CONFIRM_COST=1 bash scripts/run_full_search.sh          # full search, 100% of val
```

Results are written to `logs/<run_name>/` (`evolution_summary.jsonl`, and the
test scores under `results/`).
