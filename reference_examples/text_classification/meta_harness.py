"""Autonomous evolution loop for memory systems.

Validation-only during evolution. Test evaluation requires explicit finalization.
Uses claude_wrapper + meta-harness skill to propose new memory systems.

    uv run python meta_harness.py --iterations 20 --fresh
    uv run python meta_harness.py --iterations 10 --run-name my-run
    uv run python meta_harness.py --run-name my-run --test
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv

import claude_wrapper
from benchmark import get_model_short_name, load_results

# Real env vars win, then .env, then the committed .env.example template.
# Subprocesses (benchmark.py, inner_loop) inherit these via the environment.
load_dotenv(find_dotenv(".env"))


EVOLVE_DIR = Path(__file__).parent
CONFIG_PATH = EVOLVE_DIR / "config.yaml"
AGENTS_DIR = EVOLVE_DIR / "agents"
BASELINE_FILES = {"__init__.py", "no_memory.py", "fewshot_memory.py", "fewshot_all.py"}

# These are updated per-run if --run-name is set
LOGS_DIR = EVOLVE_DIR / "logs"
PENDING_EVAL = LOGS_DIR / "pending_eval.json"
FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
RESULTS_DIR = LOGS_DIR / "results"
FINALIZED = LOGS_DIR / "finalized.json"
# Per-run home for proposed candidates while the run is not executing.
# Set in run_evolve once the run dir is known; None means "no run claimed
# agents/ yet" and archive_candidates() is a no-op.
ARCHIVED_AGENTS = None

# Per-run code snapshot (logs/<run>/code/text_classification). When set,
# subprocesses (benchmark.py, inner_loop, proposer, import checks) execute the
# snapshot's code, so editing the main repo mid-run cannot change the run's
# evaluation conditions. Set in run_evolve when --snapshot-code is passed (or
# an existing run already has a snapshot).
CODE_SNAPSHOT = None

# The model that writes the candidates. Pinned by exact id rather than the
# "opus" alias: the alias resolved to claude-opus-4-6 under CLI 2.1.56 and to
# claude-opus-5 under 2.1.220. Changing the proposer changes the search itself,
# so this must not drift between runs.
#
# The text classification runs reported in the paper used claude-opus-4-6, so
# additional runs stay comparable with them at this value. terminal_bench_2 pins
# claude-opus-5, which is what its runs used.
PROPOSER_MODEL = "claude-opus-4-6"

# The Claude Code CLI version the proposer runs on. Its behaviour depends on the
# version (between 2.1.56 and 2.1.220 the output token count roughly doubled),
# so it is pinned. This is the version the text classification runs used.
CLAUDE_CODE_VERSION = "2.1.56"

PROPOSER_ALLOWED_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "Agent",
    "Write",
    "Edit",
    "Bash",
]

_interrupted = False

# ── ANSI colors ──────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(t):
    return _c("1", t)


def _dim(t):
    return _c("2", t)


def _green(t):
    return _c("32", t)


def _red(t):
    return _c("31", t)


def _yellow(t):
    return _c("33", t)


def _cyan(t):
    return _c("36", t)


def _ts():
    return _dim(datetime.now().strftime("[%H:%M:%S]"))


def _elapsed(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _pct(val):
    s = f"{val:.1f}%"
    if val >= 60:
        return _green(s)
    elif val >= 40:
        return _yellow(s)
    return _red(s)


def _handle_signal(signum, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupted, finishing current step...", flush=True)


def run_cmd(cmd, timeout=7200, cwd=None):
    """Wraps subprocess.run; returns CompletedProcess with returncode=124 on timeout."""
    try:
        return subprocess.run(
            cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            cmd, returncode=124, stdout="", stderr=f"Timed out after {timeout}s"
        )


def run_benchmark(args, logs_dir=None):
    if CODE_SNAPSHOT:
        # Execute the snapshot's benchmark.py, but reuse the main project's
        # .venv (--project) instead of building one per run.
        prefix = ["uv", "run", "--project", str(EVOLVE_DIR), "python", "benchmark.py"]
        cwd = CODE_SNAPSHOT
    else:
        prefix = ["uv", "run", "python", "benchmark.py"]
        cwd = EVOLVE_DIR
    return run_cmd(
        prefix
        + [
            "--logs-dir",
            str(logs_dir or LOGS_DIR),
            "--results-dir",
            str(RESULTS_DIR),
        ]
        + args,
        cwd=str(cwd),
    )


def snapshot_code(snap_dir):
    """Copy the project code into logs/<run>/code/text_classification.

    Freezes everything subprocesses re-read from disk (benchmark.py,
    inner_loop.py, config.yaml, data loaders, the proposer skill, baseline
    agents), so editing the main repo while a run executes cannot change the
    run's evaluation conditions. logs/, .venv and stray candidate files in
    agents/ are excluded; the run proposes its own candidates into the copy.
    """

    def _ignore(src, names):
        # note/ and memo.md are the researcher's private notes — keep them out
        # of snapshots so the proposer never reads them.
        drop = {
            n
            for n in names
            if n in {"logs", ".venv", "__pycache__", ".pytest_cache", "note", "memo.md"}
        }
        if Path(src) == EVOLVE_DIR / "agents":
            drop |= {
                n for n in names if n.endswith(".py") and n not in BASELINE_FILES
            }
        return drop

    shutil.copytree(EVOLVE_DIR, snap_dir, ignore=_ignore, symlinks=True)


def render_task_prompt(iteration, num_datasets, search_fraction=1.0):
    """Build the prompt for the proposer Claude session."""
    project_root = CODE_SNAPSHOT or EVOLVE_DIR
    prompt = (
        f"Run iteration {iteration} of the evolution loop. There are {num_datasets} datasets.\n\n"
        f"## Run directories\n"
        f"All logs and results for this run are under `{LOGS_DIR}/`.\n"
        f"- `{EVOLUTION_SUMMARY}` — past results\n"
        f"- `{FRONTIER_VAL}` — frontier\n"
        f"- `{LOGS_DIR / 'reports'}/` — post-eval reports\n"
        f"- Write pending_eval.json to: `{PENDING_EVAL}`\n"
        f"- Project root for THIS run's code (read config/agents here): `{project_root}`\n"
        f"- Write candidate files ONLY to: `{project_root / 'agents'}/<name>.py`"
    )
    if search_fraction < 1.0:
        prompt += (
            f"\n\n## Data subsampling\n"
            f"All validation scores in this run are computed on a {search_fraction:.0%} "
            f"subsample of the val split (training uses the full train split), so "
            f"scores are noisier than usual. Small deltas (a few percent) may be "
            f"noise; prefer mechanisms with large, consistent effects across "
            f"datasets. The top-3 systems will be re-evaluated on the full val "
            f"at the end of the run."
        )
    return prompt


def build_dataset_history(model_short, datasets):
    """Per-dataset accuracy observations (0..1) across every evaluated system.

    Reads this run's val.json results, so resumed runs recover full history.
    Returns {dataset: [accuracy, ...]}.
    """
    history = {ds: [] for ds in datasets}
    for (model, ds, _mem), data in load_results(LOGS_DIR, "val.json").items():
        if model != model_short or ds not in history:
            continue
        acc = data.get("accuracy")
        if acc is not None:
            history[ds].append(acc)
    return {ds: accs for ds, accs in history.items() if accs}


def build_sample_history(model_short, datasets):
    """Per-sample outcome observations across every evaluated system.

    Reads this run's val.json per_sample records (written by inner_loop),
    ordered chronologically by run timestamp, so resumed runs recover full
    history. Returns {dataset: {sample_index: [was_correct, ...]}} for
    sample-selector extensions.
    """
    entries = []
    for (model, ds, _mem), data in load_results(LOGS_DIR, "val.json").items():
        if model != model_short or ds not in datasets:
            continue
        per_sample = data.get("per_sample")
        if per_sample:
            entries.append((data.get("timestamp", ""), ds, per_sample))
    history = {ds: {} for ds in datasets}
    for _ts_, ds, per_sample in sorted(entries, key=lambda e: e[0]):
        for idx, ok in per_sample.items():
            history[ds].setdefault(int(idx), []).append(bool(ok))
    return {ds: samples for ds, samples in history.items() if samples}


def auto_run_name(args):
    """Timestamped run name embedding the main search settings.

    Used when --run-name is omitted, so logs/<run>/ is self-describing:
    e.g. 20260720_143052_it20_task-coevolve_fraction0.2. Selector env-var knobs
    (HASS_*, HATS_*) are included only when explicitly set.
    """
    parts = [datetime.now().strftime("%Y%m%d_%H%M%S"), f"it{args.iterations}"]
    if args.search_fraction < 1.0:
        parts.append(f"val{args.search_fraction:g}")
    if args.search_train_fraction < 1.0:
        parts.append(f"train{args.search_train_fraction:g}")
    selector_flag = (
        (args.sample_selector, "sample") if args.sample_selector else None
    )
    if selector_flag:
        from extensions import selector_env_prefix

        name, kind = selector_flag
        parts.append(name)
        prefix = selector_env_prefix(name, kind)
        for key in sorted(os.environ):
            if key.startswith(prefix):
                val = "".join(
                    c for c in os.environ[key] if c.isalnum() or c in "._-"
                )
                parts.append(f"{key[len(prefix):].replace('_', '').lower()}{val}")
    if getattr(args, "anchored_summary", None):
        parts.append(args.anchored_summary)
    model_override = os.environ.get("TC_MODEL")
    if model_override:
        tag = "".join(c for c in model_override.split("/")[-1] if c.isalnum() or c in "._-")
        parts.append(tag)
    return "_".join(parts)


def count_iterations_from_summary():
    """Highest iteration number in evolution_summary.jsonl (for resume)."""
    if not EVOLUTION_SUMMARY.exists():
        return 0
    max_iter = 0
    for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            max_iter = max(max_iter, json.loads(line).get("iteration", 0))
        except json.JSONDecodeError:
            continue
    return max_iter


def propose_claude(task_prompt, iteration, timeout=2400, skill_name="meta-harness"):
    """Returns True if candidates were produced (pending_eval.json exists)."""
    os.environ.pop("CLAUDECODE", None)
    # Strip API key so claude CLI uses subscription auth (avoids rate limits)
    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    # With a code snapshot, the proposer works inside the snapshot: it reads
    # the snapshot's skill and writes candidates into the snapshot's agents/.
    proposer_base = CODE_SNAPSHOT or EVOLVE_DIR
    skill_dir = proposer_base / ".claude/skills" / skill_name
    if not skill_dir.is_dir():
        # Snapshots frozen before the skill split only ship meta-harness;
        # resumed runs keep using it so the run stays internally consistent.
        skill_dir = proposer_base / ".claude/skills/meta-harness"
    claude_wrapper.ensure_version(CLAUDE_CODE_VERSION)
    result = claude_wrapper.run(
        prompt=task_prompt,
        model=PROPOSER_MODEL,
        allowed_tools=PROPOSER_ALLOWED_TOOLS,
        skills=[str(skill_dir)],
        cwd=str(proposer_base),
        log_dir=str(LOGS_DIR / "claude_sessions"),
        name=f"iter{iteration}",
        timeout_seconds=timeout,
        effort="high",
    )
    # Restore API key
    if saved_key:
        os.environ["ANTHROPIC_API_KEY"] = saved_key
    if result.exit_code != 0:
        print(f"  {_red('proposer failed')} exit={result.exit_code}")
        if result.stderr:
            print(f"  {_dim(result.stderr[:500])}")
        return False
    result.show()
    return PENDING_EVAL.exists()


def validate_candidates(candidates):
    """Import-check each candidate. Returns list of valid candidates."""
    valid = []
    for c in candidates:
        name = c["name"]
        if CODE_SNAPSHOT:
            # Import from the snapshot package (cwd on sys.path), main .venv
            cmd_prefix = ["uv", "run", "--project", str(EVOLVE_DIR), "python"]
            cwd = CODE_SNAPSHOT.parent
        else:
            cmd_prefix = ["uv", "run", "python"]
            cwd = EVOLVE_DIR.parent
        result = run_cmd(
            cmd_prefix
            + [
                "-c",
                f"from text_classification.agents.{name} import *; print('OK')",
            ],
            cwd=str(cwd),
            timeout=30,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            print(f"    {_green('OK')} {name}")
            valid.append(c)
        else:
            print(f"    {_red('FAIL')} {name}")
            if result.stderr:
                print(f"      {_dim(result.stderr[:200])}")
    return valid


def update_evolution_summary(
    iteration,
    candidates,
    val_scores,
    propose_time=None,
    bench_time=None,
    wall_time=None,
    selected_datasets=None,
    selected_samples=None,
    subset_ref=None,
    anchored_scores=None,
):
    """Append one JSONL row per candidate to evolution_summary.jsonl.

    selected_datasets: dataset subset used this iteration (dataset selector
    active). Subset averages are not comparable to the full-pool frontier, so
    delta is omitted and the outcome is labeled with the subset size.

    selected_samples: {dataset: [index, ...]} subset used this iteration
    (sample selector active). subset_ref is the frontier-best accuracy
    restricted to the same samples, so delta stays comparable.

    anchored_scores: {name: {est_full_val, est_by_dataset, anchors}} from the
    anchored estimator. est_full_val is a full-val-equivalent estimate,
    comparable across iterations regardless of the sampled subset.
    """
    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    pareto = frontier.get("_pareto", [])
    best_val = pareto[0].get("val_accuracy", 0) if pareto else 0

    with open(EVOLUTION_SUMMARY, "a") as f:
        for i, c in enumerate(candidates):
            name = c["name"]
            avg_val = val_scores.get(name, 0)
            if selected_datasets:
                delta = None
                outcome = (
                    f"{avg_val:.1f}% on {len(selected_datasets)}-dataset subset"
                    if avg_val > 0
                    else "failed"
                )
            elif selected_samples:
                n_samples = sum(len(v) for v in selected_samples.values())
                anchored = (anchored_scores or {}).get(name)
                if avg_val <= 0:
                    delta = None
                    outcome = "failed"
                elif subset_ref is not None:
                    delta = round(avg_val - subset_ref, 1)
                    outcome = (
                        f"{avg_val:.1f}% on {n_samples}-sample subset "
                        f"({avg_val - subset_ref:+.1f} vs frontier on same samples)"
                    )
                else:
                    delta = None
                    outcome = f"{avg_val:.1f}% on {n_samples}-sample subset"
                if avg_val > 0 and anchored:
                    outcome = f"est_full {anchored['est_full_val']:.1f}% | {outcome}"
            else:
                delta = round(avg_val - best_val, 1) if best_val else None
                outcome = (
                    f"{avg_val:.1f}% ({avg_val - best_val:+.1f})"
                    if avg_val > 0
                    else "failed"
                )
            row = {
                "iteration": iteration,
                "system": name,
                "avg_val": round(avg_val, 1),
                "axis": c.get("axis", "?"),
                "hypothesis": c.get("hypothesis", ""),
                "delta": delta,
                "outcome": outcome,
            }
            if i == 0 and selected_datasets:
                row["selected_datasets"] = selected_datasets
            if i == 0 and selected_samples:
                row["selected_samples"] = {
                    ds: idxs for ds, idxs in sorted(selected_samples.items())
                }
            anchored = (anchored_scores or {}).get(name)
            if anchored:
                row["est_full_val"] = anchored["est_full_val"]
                row["est_by_dataset"] = anchored["est_by_dataset"]
                if "est_full_se" in anchored:
                    row["est_full_se"] = anchored["est_full_se"]
                if i == 0:
                    # Anchor systems/levels are shared across the iteration
                    row["anchors"] = anchored["anchors"]
            if "components" in c:
                row["components"] = c["components"]
            if i == 0 and wall_time is not None:
                row["timing_s"] = {
                    "propose": round(propose_time, 1),
                    "bench": round(bench_time, 1),
                    "wall": round(wall_time, 1),
                }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _candidate_files():
    return [f for f in AGENTS_DIR.glob("*.py") if f.name not in BASELINE_FILES]


def rescue_misplaced_candidates(candidate_names):
    """Move THIS iteration's candidates the proposer misplaced into main agents/.

    With a code snapshot the proposer works in logs/<run>/code, but it may
    still resolve absolute paths to the main repo and write candidates there.
    Only files named in the current pending_eval are rescued: anything else in
    the main agents/ (e.g. another run's candidates left behind by an
    interrupted --test) must NOT be swept into this run's snapshot.
    """
    if not CODE_SNAPSHOT:
        return
    main_agents = EVOLVE_DIR / "agents"
    moved = []
    for name in candidate_names:
        f = main_agents / f"{name}.py"
        if f.name in BASELINE_FILES or not f.exists():
            continue
        f.replace(AGENTS_DIR / f.name)
        moved.append(f.name)
    if moved:
        print(
            f"  {_yellow('rescued')} {len(moved)} candidate file(s) written to the "
            f"main agents/: {', '.join(moved)}"
        )


def restore_candidates():
    """Move this run's archived candidates back into agents/ (resume / --test)."""
    if ARCHIVED_AGENTS.exists():
        files = list(ARCHIVED_AGENTS.glob("*.py"))
        for f in files:
            shutil.move(str(f), str(AGENTS_DIR / f.name))
        if files:
            print(f"  Restored {len(files)} candidate file(s) from {ARCHIVED_AGENTS}")


def archive_candidates():
    """Move proposed candidates from agents/ into this run's dir.

    Keeps agents/ at baselines-only between runs, so runs cannot clobber
    each other's candidate files and each run dir snapshots its own code.
    """
    if ARCHIVED_AGENTS is None:
        return
    files = _candidate_files()
    if not files:
        return
    ARCHIVED_AGENTS.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.move(str(f), str(ARCHIVED_AGENTS / f.name))
    print(f"  Archived {len(files)} candidate file(s) to {ARCHIVED_AGENTS}")


def fresh_start():
    """Clear proposed memory systems and reset logs for a fresh run."""
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    if AGENTS_DIR.exists():
        files = _candidate_files()
        for f in files:
            f.unlink()
        if files:
            print(f"  Cleared {len(files)} candidate file(s) from agents/")

    for f in [
        EVOLUTION_SUMMARY,
        FRONTIER_VAL,
        LOGS_DIR / "frontier.json",
        PENDING_EVAL,
    ]:
        if f.exists():
            f.unlink()

    if LOGS_DIR.exists():
        val_files = list(LOGS_DIR.rglob("val.json"))
        for f in val_files:
            f.unlink()
        if val_files:
            print(f"  Cleared {len(val_files)} val result files")

    print(f"  {_green('Fresh start')}: cleared generated agents and log files")


def finalize_run(baselines, datasets, model_short):
    """Evaluate test once, after freezing this run against further evolution."""
    if not FRONTIER_VAL.exists():
        print(f"ERROR: no validation frontier for run at {LOGS_DIR}")
        raise SystemExit(1)

    if FINALIZED.exists():
        state = json.loads(FINALIZED.read_text())
        if state.get("status") == "complete":
            print(f"Run already finalized: {LOGS_DIR.name}")
            result = run_benchmark(["--results", "--test"])
            if result.stdout:
                print(result.stdout)
            return

    frontier = json.loads(FRONTIER_VAL.read_text())
    pareto = frontier.get("_pareto", [])
    test_systems = set(baselines)
    test_systems.update(entry["system"] for entry in pareto)
    for key, value in frontier.items():
        if not key.startswith("_") and isinstance(value, dict):
            if "best_system" in value:
                test_systems.add(value["best_system"])

    # Subset runs keep candidates out of the frontier, so the selection-stage
    # winners live in top3_full.json — test those too.
    top3_file = LOGS_DIR / "top3_full.json"
    if top3_file.exists():
        t3 = json.loads(top3_file.read_text())
        test_systems.update(s["system"] for s in t3.get("systems", []))

    # Systems missing a saved memory state for any dataset cannot be test
    # evaluated (mid-search crashes); requiring them would block completion.
    untestable = sorted(
        s
        for s in test_systems
        if any(
            not (LOGS_DIR / ds / s / model_short / "memory.json").exists()
            for ds in datasets
        )
    )
    if untestable:
        print(
            f"  {_yellow('skipping untestable')} (no saved memory for some "
            f"dataset): {', '.join(untestable)}"
        )
        test_systems -= set(untestable)

    FINALIZED.write_text(
        json.dumps(
            {
                "status": "in_progress",
                "started_at": datetime.now().isoformat(),
                "systems": sorted(test_systems),
            },
            indent=2,
        )
    )

    print(f"\n{_ts()} {_bold('Phase Final: Test evaluation')}")
    failed = False
    for name in sorted(test_systems):
        print(f"  {_ts()} test eval: {_bold(name)}", flush=True)
        result = run_benchmark(["--memory", name, "--test"])
        if result.returncode != 0:
            failed = True
            print(f"    {_red('FAIL')} {name} test eval failed")

    frontier_result = run_benchmark(["--frontier", "--test", "--model", model_short])
    failed = failed or frontier_result.returncode != 0

    result = run_benchmark(["--results", "--test"])
    failed = failed or result.returncode != 0
    if result.stdout:
        print(result.stdout)

    test_results = load_results(RESULTS_DIR, "test.json")
    missing = [
        (model_short, dataset, system)
        for system in test_systems
        for dataset in datasets
        if (model_short, dataset, system) not in test_results
    ]
    if missing:
        failed = True
        print(f"Missing {len(missing)} complete test result group(s).")

    if failed:
        print("Test finalization incomplete. Fix failures, then rerun --test.")
        raise SystemExit(1)

    state = json.loads(FINALIZED.read_text())
    state["status"] = "complete"
    state["completed_at"] = datetime.now().isoformat()
    FINALIZED.write_text(json.dumps(state, indent=2))
    print(f"\n{_ts()} {_bold('Test finalization complete.')}")


def run_top3_full(
    model_short,
    datasets,
    baselines,
    top_k=3,
    search_fraction=1.0,
    search_train_fraction=1.0,
):
    """Re-evaluate the top-K candidates (by search avg_val) on the full data.

    Search-phase val results stay under LOGS_DIR; full-data val runs are
    written to LOGS_DIR/full_eval so the two never collide in caching or
    frontier computation.
    """
    full_dir = LOGS_DIR / "full_eval"

    results = load_results(LOGS_DIR, "val.json")
    scores = {}
    for (model, ds, mem), data in results.items():
        if model != model_short or mem in baselines:
            continue
        acc = data.get("accuracy")
        if acc is not None:
            scores.setdefault(mem, {})[ds] = acc * 100
    if not scores:
        print(f"  {_red('no candidate val results')}; nothing to evaluate on full data")
        return

    search_avg = {
        mem: sum(by_ds.get(ds, 0.0) for ds in datasets) / len(datasets)
        for mem, by_ds in scores.items()
    }

    # Rank by est_full_val (full-val-equivalent estimate) when the run recorded
    # one (--anchored-summary); raw subset averages are subset-biased and noisy.
    est_scores = {}
    if EVOLUTION_SUMMARY.exists():
        for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("est_full_val") is not None:
                est_scores[row["system"]] = row["est_full_val"]
    rank_by = {mem: est_scores.get(mem, search_avg[mem]) for mem in search_avg}
    top = sorted(rank_by, key=rank_by.get, reverse=True)[:top_k]

    criterion = "est_full_val" if est_scores else "search avg_val"
    print(
        f"\n{_ts()} {_bold(f'Phase Final: full-data val of top {len(top)}')} "
        f"(ranked by {criterion})"
    )
    for name in top:
        est_str = f"  est_full={_pct(est_scores[name])}" if name in est_scores else ""
        print(f"    {_bold(name)}: search avg_val={_pct(search_avg[name])}{est_str}")

    for name in top:
        print(f"  {_ts()} full val: {_bold(name)}...", flush=True)
        t0 = time.time()
        result = run_benchmark(["--memory", name], logs_dir=full_dir)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"    {_red('FAIL')} ({_elapsed(elapsed)})")
            if result.stderr:
                print(f"    {_dim(result.stderr[:300])}")
        else:
            print(f"    {_green('OK')} ({_elapsed(elapsed)})")

    full_results = load_results(full_dir, "val.json")
    full_avg = {}
    print(f"\n  {_bold('search (subsampled) vs full validation:')}")
    for name in top:
        accs = [
            full_results[k]["accuracy"] * 100
            for ds in datasets
            for k in [(model_short, ds, name)]
            if k in full_results and full_results[k].get("accuracy") is not None
        ]
        full_avg[name] = sum(accs) / len(datasets) if accs else 0.0
        print(
            f"    {_bold(name)}: search={_pct(search_avg[name])} -> "
            f"full={_pct(full_avg[name])}"
        )

    summary = {
        "search_val_fraction": search_fraction,
        "search_train_fraction": search_train_fraction,
        "model": model_short,
        "ranked_by": criterion,
        "systems": [
            {
                "system": name,
                "search_avg_val": round(search_avg[name], 1),
                **(
                    {"est_full_val": est_scores[name]}
                    if name in est_scores
                    else {}
                ),
                "full_avg_val": round(full_avg.get(name, 0.0), 1),
            }
            for name in top
        ],
        "completed_at": datetime.now().isoformat(),
    }
    (LOGS_DIR / "top3_full.json").write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {LOGS_DIR / 'top3_full.json'}")

    result = run_benchmark(["--results"], logs_dir=full_dir)
    if result.stdout:
        print(result.stdout)


def run_evolve(args):
    global LOGS_DIR, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY
    global AGENTS_DIR, CODE_SNAPSHOT
    global RESULTS_DIR, FINALIZED, ARCHIVED_AGENTS

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    datasets = cfg["datasets"]

    config_model_ids = [m["model"] for m in cfg.get("models", [])]
    if args.model not in config_model_ids:
        print(f"ERROR: --model {args.model} not in config.yaml: {config_model_ids}")
        sys.exit(1)

    model_short = get_model_short_name(args.model)

    # Isolate run outputs under run-name subdirs
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = auto_run_name(args)
    LOGS_DIR = EVOLVE_DIR / "logs" / run_name
    PENDING_EVAL = LOGS_DIR / "pending_eval.json"
    FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
    RESULTS_DIR = LOGS_DIR / "results"
    FINALIZED = LOGS_DIR / "finalized.json"
    ARCHIVED_AGENTS = LOGS_DIR / "agents"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Code snapshot: freeze this run's code under logs/<run>/code. An existing
    # snapshot (resume / --test / --top3-full) is always reused so the run
    # keeps executing the code it started with. --fresh re-snapshots.
    snap_parent = LOGS_DIR / "code"
    snap_dir = snap_parent / "text_classification"
    if args.fresh and snap_parent.exists():
        shutil.rmtree(snap_parent)
    if args.snapshot_code or snap_dir.exists():
        if snap_dir.exists():
            print(f"  code snapshot: {_cyan('reusing')} {snap_dir}")
        else:
            snapshot_code(snap_dir)
            print(f"  code snapshot: {_cyan('created')} {snap_dir}")
        CODE_SNAPSHOT = snap_dir
        AGENTS_DIR = snap_dir / "agents"
        # inner-loop workers reuse the main project's .venv
        os.environ["TC_UV_PROJECT"] = str(EVOLVE_DIR)

    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    restore_candidates()

    baselines = cfg["memory_systems"]["baselines"]
    if args.test:
        finalize_run(baselines, datasets, model_short)
        return

    if args.top3_full:
        run_top3_full(
            model_short,
            datasets,
            baselines,
            top_k=args.top_k,
            search_fraction=args.search_fraction,
            search_train_fraction=args.search_train_fraction,
        )
        return

    if (
        FINALIZED.exists()
        and json.loads(FINALIZED.read_text()).get("status") == "complete"
    ):
        print(
            f"ERROR: run '{run_name}' is finalized; use a new --run-name to continue evolution"
        )
        raise SystemExit(1)

    if args.fresh:
        fresh_start()

    fraction_args = []
    if args.search_fraction < 1.0:
        fraction_args += ["--val-fraction", str(args.search_fraction)]
    if args.search_train_fraction < 1.0:
        fraction_args += ["--train-fraction", str(args.search_train_fraction)]

    selector = None

    sample_selector = None
    if args.sample_selector:
        from extensions import get_sample_selector

        sample_selector = get_sample_selector(args.sample_selector)
        print(
            f"  sample selector: {_cyan(args.sample_selector)} ({sample_selector.__name__})"
        )

    anchor = None
    if args.anchored_summary:
        from extensions import get_summary_estimator

        anchor = get_summary_estimator(args.anchored_summary)
        print(
            f"  summary estimator: {_cyan(args.anchored_summary)} ({anchor.__name__})"
        )
        if getattr(anchor, "REQUIRES_INCLUSION", False) and not hasattr(
            sample_selector, "inclusion_probs"
        ):
            print(
                f"  {_yellow('WARN')} estimator requires inclusion probabilities "
                f"but the sample selector does not provide inclusion_probs(); "
                f"est_full_val will be skipped"
            )

    print(
        f"{_ts()} {_bold('Evolution (memory systems)')}  "
        f"run={_cyan(run_name)}  model={_cyan(args.model)}  "
        f"iters={args.iterations}  datasets={datasets}"
        + (
            f"  search_val_fraction={_cyan(args.search_fraction)}"
            if args.search_fraction < 1.0
            else ""
        )
        + (
            f"  search_train_fraction={_cyan(args.search_train_fraction)}"
            if args.search_train_fraction < 1.0
            else ""
        )
    )

    # ── Phase 0: Baselines ─────────────────────────────────────
    if not args.skip_baseline:
        print(f"\n{_ts()} {_bold('Phase 0: Baselines')}  systems={baselines}")
        for bl in baselines:
            if _interrupted:
                break
            print(f"  {_ts()} benchmarking {_bold(bl)}...", flush=True)
            t0 = time.time()
            result = run_benchmark(["--memory", bl] + fraction_args)
            elapsed = time.time() - t0
            if result.returncode != 0:
                print(f"    {_red('FAIL')} {bl}: {result.stderr[:200]}")
            else:
                print(f"    {_green('OK')} ({_elapsed(elapsed)})")

        run_benchmark(["--frontier", "--model", model_short])

        # Show baseline results
        results = load_results(LOGS_DIR, "val.json")
        for bl in baselines:
            accs = [
                results[k]["accuracy"] * 100
                for ds in datasets
                for k in [(model_short, ds, bl)]
                if k in results and results[k].get("accuracy") is not None
            ]
            if accs:
                avg = sum(accs) / len(accs)
                print(f"    {_bold(bl)}: avg_val={_pct(avg)}")

    # ── Phase 1..N: Evolution ──────────────────────────────────
    start_iteration = count_iterations_from_summary() + 1
    for i in range(args.iterations):
        if _interrupted:
            print("Interrupted.")
            break

        iteration = start_iteration + i
        iter_start = time.time()

        # Show frontier status
        frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
        pareto = frontier.get("_pareto", [])
        best_val = pareto[0].get("val_accuracy", 0) if pareto else 0
        best_sys = pareto[0].get("system", "none") if pareto else "none"

        print(
            f"\n{_ts()} {_bold(f'Iteration {iteration}')} ({i + 1}/{args.iterations})  "
            f"frontier={best_sys} @ {_pct(best_val * 100 if best_val <= 1 else best_val)}"
        )
        print(f"{'─' * 60}")

        task_prompt = render_task_prompt(
            iteration, len(datasets), args.search_fraction
        )

        if PENDING_EVAL.exists():
            PENDING_EVAL.unlink()

        # Propose
        propose_start = time.time()
        print(f"  {_ts()} {_cyan('proposing')} new candidates...", flush=True)
        # Selector runs need the est_full_val/delta ranking caveat (avg_val is
        # measured on a different subset each iteration). Naive fixed-subsample
        # and full runs evaluate the same samples every iteration, so they get
        # the original skill without that instruction.
        # A selector run without --anchored-summary also gets the naive skill:
        # there is no est_full_val, so the other instructions would not match
        # what the proposer actually sees.
        proposer_skill = (
            "meta-harness"
            if args.sample_selector and args.anchored_summary
            else "meta-harness-naive"
        )
        ok = propose_claude(
            task_prompt,
            iteration,
            timeout=args.propose_timeout,
            skill_name=proposer_skill,
        )
        propose_time = time.time() - propose_start

        if not ok:
            print(
                f"  {_red('FAIL')} proposer returned no candidates after {_elapsed(propose_time)}"
            )
            continue

        candidates = json.loads(PENDING_EVAL.read_text()).get("candidates", [])
        rescue_misplaced_candidates([c["name"] for c in candidates])
        print(
            f"  {_ts()} proposed {len(candidates)} candidate(s) in {_elapsed(propose_time)}"
        )
        for ci, c in enumerate(candidates):
            hyp = c.get("hypothesis", "")
            print(f"    {ci + 1}. {_bold(c['name'])}: {hyp[:80]}")

        # Validate
        print(f"  {_ts()} {_cyan('validating')} {len(candidates)} candidate(s)...")
        valid_candidates = validate_candidates(candidates)

        if not valid_candidates:
            print(
                f"  {_red('0 valid')} out of {len(candidates)} candidates, skipping iteration"
            )
            update_evolution_summary(
                iteration, candidates, {}, propose_time=propose_time
            )
            continue
        print(
            f"  {_green(f'{len(valid_candidates)} valid')} out of {len(candidates)} candidates"
        )

        # Benchmark
        selected_datasets = None
        selected_samples = None
        if selector:
            selected_datasets = selector.select_datasets(
                build_dataset_history(model_short, datasets), iteration
            )
            if selected_datasets:
                print(
                    f"  {_cyan('dataset selector')} picked "
                    f"{len(selected_datasets)}/{len(datasets)} datasets: "
                    + ", ".join(selected_datasets)
                )
            else:
                print(
                    f"  {_cyan('dataset selector')} full-pool evaluation (exploration)"
                )
        elif sample_selector:
            sample_history = build_sample_history(model_short, datasets)
            selected_samples = sample_selector.select_samples(
                sample_history, iteration
            )
            if selected_samples:
                picked = ", ".join(
                    f"{ds} {len(idxs)}/{len(sample_history.get(ds, []))}"
                    for ds, idxs in sorted(selected_samples.items())
                )
                print(f"  {_cyan('sample selector')} picked samples: {picked}")
            else:
                print(f"  {_cyan('sample selector')} full-pool evaluation")
        eval_datasets = selected_datasets or datasets

        bench_start = time.time()
        print(
            f"  {_ts()} {_cyan('benchmarking')} {len(valid_candidates)} system(s) x {len(eval_datasets)} datasets"
        )
        for ci, c in enumerate(valid_candidates):
            if _interrupted:
                break
            name = c["name"]
            print(
                f"    [{ci + 1}/{len(valid_candidates)}] {_bold(name)}...", flush=True
            )
            t0 = time.time()
            if selected_datasets:
                failed = False
                for ds in selected_datasets:
                    result = run_benchmark(
                        ["--memory", name, "--dataset", ds] + fraction_args
                    )
                    failed = failed or result.returncode != 0
            elif selected_samples:
                # One invocation for all datasets so their jobs run
                # concurrently; unlisted datasets get the full val.
                result = run_benchmark(
                    ["--memory", name, "--val-indices-json", json.dumps(selected_samples)]
                    + fraction_args
                )
                failed = result.returncode != 0
            else:
                result = run_benchmark(["--memory", name] + fraction_args)
                failed = result.returncode != 0
            elapsed = time.time() - t0
            if failed:
                print(f"      {_red('FAIL')} benchmark crashed ({_elapsed(elapsed)})")
            else:
                print(f"      {_green('OK')} ({_elapsed(elapsed)})")
        bench_time = time.time() - bench_start

        run_benchmark(["--frontier", "--model", model_short])

        # Compute scores and show results
        val_scores = {}
        results = load_results(LOGS_DIR, "val.json")
        # On a subset, compare against the per-dataset frontier restricted to
        # the same datasets; the full-pool pareto best is not comparable.
        if selected_datasets:
            ref_val = sum(
                frontier.get(ds, {}).get("accuracy", 0) for ds in selected_datasets
            ) / len(selected_datasets)
        elif selected_samples:
            # Sample subsets skew hard, so restrict the per-dataset frontier
            # best to the same samples via its per_sample record.
            ref_accs = []
            for ds in datasets:
                best_system = frontier.get(ds, {}).get("best_system")
                data = results.get((model_short, ds, best_system))
                per_sample = (data or {}).get("per_sample") or {}
                idxs = selected_samples.get(ds)
                if idxs:
                    outcomes = [
                        per_sample[str(i)] for i in idxs if str(i) in per_sample
                    ]
                    if outcomes:
                        ref_accs.append(
                            100.0 * sum(map(bool, outcomes)) / len(outcomes)
                        )
                        continue
                ref_accs.append(frontier.get(ds, {}).get("accuracy", 0))
            ref_val = sum(ref_accs) / len(ref_accs) if ref_accs else 0.0
        else:
            ref_val = best_val * 100 if best_val <= 1 else best_val

        anchored_scores = None
        if anchor and selected_samples:
            inclusion = None
            if getattr(anchor, "REQUIRES_INCLUSION", False) and hasattr(
                sample_selector, "inclusion_probs"
            ):
                # π must come from the same history the selection was drawn from
                inclusion = sample_selector.inclusion_probs(
                    sample_history, iteration
                )
            anchored_scores = {}
            for c in valid_candidates:
                est = anchor.estimate_full_scores(
                    datasets, frontier, results, selected_samples, model_short,
                    c["name"], inclusion=inclusion, history=sample_history,
                )
                if est:
                    anchored_scores[c["name"]] = est

        for c in valid_candidates:
            name = c["name"]
            accs = [
                results[k]["accuracy"] * 100
                for ds in eval_datasets
                for k in [(model_short, ds, name)]
                if k in results and results[k].get("accuracy") is not None
            ]
            missing_ds = [
                ds
                for ds in eval_datasets
                if (model_short, ds, name) not in results
                or results[(model_short, ds, name)].get("accuracy") is None
            ]
            if missing_ds:
                # Averaging a partial evaluation inflates the score, because it
                # averages only the datasets that succeeded. During a provider
                # outage this produced candidates with a fake 100.0 that then
                # contaminated the frontier and the proposer. A candidate missing
                # even one dataset is recorded as failed (0).
                val_scores[name] = 0
                if anchored_scores:
                    anchored_scores.pop(name, None)  # keep the estimate honest too
                print(
                    f"    {_red('PARTIAL')} {_bold(name)}: missing "
                    f"{', '.join(missing_ds)}; treated as failed"
                )
            else:
                val_scores[name] = sum(accs) / len(accs) if accs else 0
            delta = val_scores[name] - ref_val
            delta_str = f"{delta:+.1f}"
            delta_colored = (
                _green(delta_str)
                if delta > 0
                else (_red(delta_str) if delta < 0 else _dim(delta_str))
            )
            est_str = ""
            if anchored_scores and name in anchored_scores:
                est_str = f"  est_full={_pct(anchored_scores[name]['est_full_val'])}"
            print(
                f"    {_bold(name)}: avg_val={_pct(val_scores[name])}  delta={delta_colored}{est_str}"
            )

        # An iteration where every candidate failed is more likely a systemic
        # outage (exhausted credit, provider down) than a bug in all of them.
        # Continuing would burn the remaining iterations on nothing, so the run
        # stops without recording anything and can be resumed from the same
        # iteration.
        if valid_candidates and all(
            val_scores.get(c["name"], 0) == 0 for c in valid_candidates
        ):
            print(
                f"\n{_red('FATAL')} all {len(valid_candidates)} candidate(s) failed "
                f"evaluation this iteration — systemic API failure likely "
                f"(credit/provider). Aborting without recording; this iteration "
                f"will be retried on resume."
            )
            raise SystemExit(2)

        wall_time = time.time() - iter_start
        update_evolution_summary(
            iteration,
            valid_candidates,
            val_scores,
            propose_time=propose_time,
            bench_time=bench_time,
            wall_time=wall_time,
            selected_datasets=selected_datasets,
            selected_samples=selected_samples,
            subset_ref=ref_val if selected_samples else None,
            anchored_scores=anchored_scores,
        )

        # Show iteration summary
        if selected_datasets:
            status = _dim(
                f"subset iteration ({len(selected_datasets)}/{len(datasets)} datasets; pareto frozen)"
            )
        elif selected_samples:
            n_samples = sum(len(v) for v in selected_samples.values())
            status = _dim(
                f"sample-subset iteration ({n_samples} samples; pareto frozen)"
            )
        else:
            improved = any(
                v > (best_val * 100 if best_val <= 1 else best_val)
                for v in val_scores.values()
            )
            status = _green("NEW BEST") if improved else _dim("no improvement")
        print(f"  {_ts()} {status}")
        print(
            f"  {_dim(f'timing: propose={_elapsed(propose_time)} bench={_elapsed(bench_time)} total={_elapsed(wall_time)}')}"
        )

    if _interrupted:
        return

    print(f"\n{_ts()} {_bold('Validation evolution complete.')}")

    # Sample-selector runs select by est_full_val (paper Section 3.3, Phase 3),
    # which costs no extra evaluation. Only fraction runs re-evaluate the top-K
    # on the full validation set.
    if (
        args.search_fraction < 1.0 or args.search_train_fraction < 1.0
    ) and not args.skip_top3_full:
        run_top3_full(
            model_short,
            datasets,
            baselines,
            top_k=args.top_k,
            search_fraction=args.search_fraction,
            search_train_fraction=args.search_train_fraction,
        )

    # Held-out test of the selection winners + baselines (freezes the run)
    if args.auto_test:
        finalize_run(baselines, datasets, model_short)
    else:
        print(
            f"Finalize once with: uv run python meta_harness.py --run-name {run_name} --test"
        )


def main():
    parser = argparse.ArgumentParser(description="Evolution loop for memory systems")
    parser.add_argument("--iterations", type=int, default=20)
    with open(CONFIG_PATH) as f:
        _cfg = yaml.safe_load(f)
    _default_model = _cfg["models"][0]["model"] if _cfg.get("models") else None
    parser.add_argument(
        "--model",
        default=_default_model,
        help=f"Solver model (default: {_default_model})",
    )
    parser.add_argument(
        "--propose-timeout",
        type=int,
        default=2400,
        help="Timeout per propose step (default: 2400s)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for isolated output dirs. Auto-generated if not set.",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Clear proposed systems and reset logs"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip Phase 0 baseline eval"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Finalize an existing named run with one held-out test evaluation",
    )
    parser.add_argument(
        "--search-fraction",
        type=float,
        default=1.0,
        help="Fraction of the val split used during evolution search "
        "(e.g. 0.1 = eval on 1/10 of val; top candidates get a full-data val "
        "at the end). Training always uses the full train split unless "
        "--search-train-fraction is set.",
    )
    parser.add_argument(
        "--search-train-fraction",
        type=float,
        default=1.0,
        help="Fraction of the train split used during evolution search "
        "(default 1.0; training is cheap, so shrinking it mostly hurts fidelity)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top candidates to re-evaluate on full data (default: 3)",
    )
    parser.add_argument(
        "--top3-full",
        action="store_true",
        help="Run the full-data val of the top-K candidates for an existing run, then exit",
    )
    parser.add_argument(
        "--skip-top3-full",
        action="store_true",
        help="Skip the automatic top-K full-val re-eval at the end of a "
        "--search-fraction run (selection is then done externally, e.g. "
        "max subset val -> held-out test)",
    )
    parser.add_argument(
        "--auto-test",
        action="store_true",
        help="After the run, evaluate the held-out test "
        "for the selection winners + baselines and freeze the run. Test scores "
        "are required to discuss results; this runs it once, automatically.",
    )
    parser.add_argument(
        "--sample-selector",
        type=str,
        default=None,
        help="Extension per-sample selector for candidate benchmarking "
        "(see extensions/__init__.py SAMPLE_SELECTORS; currently 'task-coevolve')",
    )
    parser.add_argument(
        "--snapshot-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy the project code to logs/<run>/code at run start and execute "
        "benchmarks / proposer from the copy, so editing the repo while the "
        "run executes cannot change its evaluation conditions. ON by default; "
        "disable with --no-snapshot-code. Resumed runs reuse an existing "
        "snapshot automatically.",
    )
    parser.add_argument(
        "--anchored-summary",
        nargs="?",
        const="hajek",
        default=None,
        metavar="ESTIMATOR",
        help="Record est_full_val (full-val-equivalent estimate) in "
        "evolution_summary rows, making sample-subset iterations comparable. "
        "Optional value picks the estimator from SUMMARY_ESTIMATORS: "
        "'hajek' (Eq. 3, default) or 'difference' (Eq. 4). "
        "Requires --sample-selector.",
    )
    args = parser.parse_args()

    if args.test and not args.run_name:
        parser.error("--test requires --run-name")
    if args.test and args.fresh:
        parser.error("--test cannot be combined with --fresh")
    if args.top3_full and not args.run_name:
        parser.error("--top3-full requires --run-name")
    if args.top3_full and (args.test or args.fresh):
        parser.error("--top3-full cannot be combined with --test or --fresh")
    if args.auto_test and args.test:
        parser.error("--auto-test is implied by --test; use one or the other")
    if not 0.0 < args.search_fraction <= 1.0:
        parser.error("--search-fraction must be in (0, 1]")
    if not 0.0 < args.search_train_fraction <= 1.0:
        parser.error("--search-train-fraction must be in (0, 1]")
    if args.sample_selector and args.search_fraction < 1.0:
        parser.error(
            "--sample-selector requires --search-fraction 1.0; sample indices "
            "are defined on the full val split"
        )
    if args.anchored_summary and not args.sample_selector:
        parser.error("--anchored-summary requires --sample-selector")
    # Resolve extension names up front so typos fail before any run dir is made
    try:
        if args.sample_selector:
            from extensions import get_sample_selector

            get_sample_selector(args.sample_selector)
        if args.anchored_summary:
            from extensions import get_summary_estimator

            get_summary_estimator(args.anchored_summary)
    except (KeyError, AttributeError) as exc:
        parser.error(str(exc))

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        run_evolve(args)
    finally:
        # Park candidates in the run dir on every exit path (done,
        # interrupted, or failed); the next run_evolve restores them.
        archive_candidates()


if __name__ == "__main__":
    main()
