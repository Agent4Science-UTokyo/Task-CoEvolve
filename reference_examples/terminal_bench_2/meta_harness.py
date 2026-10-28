"""Autonomous evolution loop for agent scaffolds on Terminal-Bench 2.0.

Starts from the shipped KIRA baseline and evolves improvements on the full
official TB2 dataset used in the paper runs.

    uv run python meta_harness.py --iterations 5
    uv run python meta_harness.py --iterations 10 --trials 2 --fresh
"""

import argparse
import importlib
import inspect
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import claude_wrapper
from dotenv import load_dotenv

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
    """Short timestamp prefix."""
    return _dim(datetime.now().strftime("[%H:%M:%S]"))


def _elapsed(seconds):
    """Format seconds as 1m23s or 45s."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def _rate_str(rate):
    """Color-coded pass rate."""
    s = f"{rate:.0%}"
    if rate >= 0.75:
        return _green(s)
    elif rate >= 0.25:
        return _yellow(s)
    return _red(s)


# The repository-level .env fills in what the shell has not set; a value
# exported in the shell always wins. With override=True, an empty line copied
# from .env.example (e.g. `AZURE_OPENAI_API_KEY=`) replaced a valid key exported
# in the shell, and every trial failed with AuthenticationError.
load_dotenv(override=False)
# LiteLLM's Azure route reads its endpoint from AZURE_API_BASE. With only
# AZURE_OPENAI_ENDPOINT set, the request still reaches the right endpoint (the
# OpenAI client reads that variable itself), but LiteLLM's pre-call logging sees
# api_base=None and prints a TypeError traceback on every call. Mirror the
# endpoint so both read the same value. Subprocesses inherit it.
if os.environ.get("AZURE_OPENAI_ENDPOINT") and not os.environ.get("AZURE_API_BASE"):
    os.environ["AZURE_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"]

EVOLVE_DIR = Path(__file__).parent
LOGS_DIR = EVOLVE_DIR / "logs"
JOBS_DIR = EVOLVE_DIR / "jobs"
PENDING_EVAL = LOGS_DIR / "pending_eval.json"
FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
AGENTS_DIR = EVOLVE_DIR / "agents"

BASELINES = [
    ("kira-baseline", "agents.baseline_kira:AgentHarness"),
    ("terminus2-baseline", "agents.baseline_terminus2:AgentHarness"),
]
BASELINE_AGENT_NAME = BASELINES[0][0]  # primary baseline for frontier comparison
BASELINE_FILES = {"__init__.py", "baseline_kira.py", "baseline_terminus2.py"}
# A KIRA-style candidate is a pair: agents/<name>.py and
# prompt-templates/<name>.txt. Templates follow the same lifecycle as the
# candidate itself (archive, restore, exclude from the snapshot).
TEMPLATES_DIR = EVOLVE_DIR / "prompt-templates"
BASELINE_TEMPLATES = {"terminus-kira.txt"}
ARCHIVED_TEMPLATES = None
# Per-run home for proposed candidates while the run is not executing.
# Set in run_evolve once the run dir is known; None means "no run claimed
# agents/ yet" and archive_candidates() is a no-op.
ARCHIVED_AGENTS = None

# Per-run code snapshot (logs/<run>/code/terminal_bench_2). When set, harbor,
# the proposer, and import checks execute the snapshot's code, so editing the
# main repo mid-run cannot change the run's evaluation conditions. Set in
# run_evolve when --snapshot-code is passed (or an existing run already has a
# snapshot). The main .venv is reused via UV_PROJECT_ENVIRONMENT.
CODE_SNAPSHOT = None

EVAL_TASK_SET = os.environ.get(
    "EVAL_TASK_SET", "full"
)  # "full" | "hard" | any key of task_split.json
TASK_SPLIT_FILE = EVOLVE_DIR / "task_split.json"  # seed-fixed task subsets


def _split_tasks(name):
    """The task list under `name` in task_split.json, or None if absent."""
    try:
        return json.loads(TASK_SPLIT_FILE.read_text()).get(name)
    except FileNotFoundError:
        return None


def _task_set_size(name):
    if name in ("full", "hard"):
        # "full" is the same 89 tasks as the "full" key, but runs without -i filters
        return 30 if name == "hard" else 89
    tasks = _split_tasks(name)
    if tasks is None:
        raise SystemExit(f"EVAL_TASK_SET={name!r} is not a key of task_split.json")
    return len(tasks)


N_EVAL_TASKS = _task_set_size(EVAL_TASK_SET)


def _active_tasks():
    """Task names of the current EVAL_TASK_SET, or None when unrestricted."""
    if EVAL_TASK_SET in ("full", "hard"):
        return None
    return set(_split_tasks(EVAL_TASK_SET))


# Shared full-89 baseline rollouts. Phase 0 falls back to this job dir when
# jobs/<run>/<baseline>-t<trials> is absent, filtering per-task results to the
# active task set — val/val10pct runs reuse the same rollouts with no copying.
# Baselines are collected per dataset version; rollouts from an earlier version
# of the benchmark cannot be reused. If the directory is missing, Phase 0
# evaluates both baselines itself.
BASELINE_JOBS_SOURCE = Path(
    os.environ.get("BASELINE_JOBS_SOURCE", str(EVOLVE_DIR / "jobs" / "gpt_oss_tb21"))
)

SMOKE_TEST_TASK = "extract-elf"  # simple task, reliably fast

# Rollout model for harbor evaluations (a litellm model string). Defaults to a
# locally served model, which has no marginal cost and is unaffected by provider
# credit or empty-response incidents. run_eval.sh sets OPENAI_API_BASE for
# openai/* models; override it to point at your own endpoint.
# For a hosted provider, set HARBOR_MODEL=openrouter/openai/gpt-oss-120b; for
# Claude, HARBOR_MODEL=anthropic/... (which also needs ALLOW_ANTHROPIC_MODEL=1).
MODEL = os.environ.get("HARBOR_MODEL", "openai/gpt-oss-120b")
DEFAULT_SEARCH_TRIALS = 2
# At 50 concurrent trials the sandbox layer produced frequent write, block and
# setup-timeout errors; at 10 they disappeared entirely, and the measured score
# of the same harness rose from 15.7% to 21.9%. The errors were being scored as
# failures, so concurrency was silently costing accuracy.
DEFAULT_CONCURRENCY = 10
HARBOR_TIMEOUT_SECONDS = int(os.environ.get("HARBOR_TIMEOUT_SECONDS", 8 * 60 * 60))
# Watchdog for hung harbor jobs. "Complete": every trial reached a terminal
# state (result.json or exception.txt) but the harbor process keeps running —
# observed when a trial crashes without writing result.json and harbor spins
# forever waiting for it. "Stall": no trial file touched for this long.
HARBOR_COMPLETE_GRACE_SECONDS = int(os.environ.get("HARBOR_COMPLETE_GRACE_SECONDS", 300))
HARBOR_STALL_TIMEOUT_SECONDS = int(os.environ.get("HARBOR_STALL_TIMEOUT_SECONDS", 3600))

# The local model server occasionally (~2% of trials) fails to parse its own
# output and raises InternalServerError. Retrying the same context reproduces it,
# but re-rolling the whole trial changes the trajectory and almost always clears
# it: of 54 affected trials, 3 survived a re-roll. Retrying without bound would
# skew the distribution, so the number of passes is capped and whatever remains
# is accepted as a reward of 0.
TRIAL_RETRY_EXCEPTIONS = ("InternalServerError",)
# Two passes, the same cap as the environment-failure repair below. One pass
# already clears ~98% of these (54 -> 3); the second rarely finds anything and
# costs wall clock, but the cap is what bounds the retrying, so it is stated
# once and applied to both kinds of failure.
TRIAL_RETRY_PASSES = int(os.environ.get("TRIAL_RETRY_PASSES", 2))
TRIAL_RETRY_TIMEOUT_SECONDS = int(os.environ.get("TRIAL_RETRY_TIMEOUT_SECONDS", 4 * 3600))

# Under rootless docker each container start consumes a kernel keyring key from
# the invoking user's quota (kernel.keys.maxkeys, 200 by default). Keys of exited
# containers linger until garbage collection, so a long churn exhausts the quota
# and every later container start dies with "unable to create session key: disk
# quota exceeded"; one observed run lost 162 of 178 trials in minutes. This is
# unrelated to whether the task was solved, so the affected trials are deleted
# and re-run once the quota frees up. The permanent fix is to raise
# kernel.keys.maxkeys.
KEYRING_ERROR_SIGNATURE = ("session key", "disk quota exceeded")
# Signatures of transient environment failures. Each tuple is an AND over
# substrings; the list is an OR over tuples. Selection is by message rather than
# by exception type, because these arrive as RuntimeError, the same type as a
# genuine bug in a candidate.
INFRA_ERROR_SIGNATURES = (
    KEYRING_ERROR_SIGNATURE,                # kernel keyring quota exhausted
    ("Failed to start tmux session",),      # tmux session failed to start
    ("failed to send non-blocking keys",),  # tmux send timed out
    # The 120s setup timeout of the harness framework. It clusters when the host
    # is busy. Setup is framework code that candidates do not touch, so this is
    # safe to treat as an environment failure; if a candidate did break setup,
    # the pass cap still lands it on a reward of 0.
    ("Command timed out after 120 seconds",),
    # Docker's bridge-network address pool is exhausted. Orphan networks left by
    # interrupted runs fill the default pool (about 30 slots), after which
    # environment setup fails immediately and trials die in seconds; one job lost
    # 67 of 89 this way. Candidates play no part in creating networks, so this is
    # an environment failure. The permanent fix is a larger default-address-pools
    # setting in the docker daemon configuration.
    ("could not find an available, non-overlapping IPv4 address pool",),
)
KEYRING_RETRY_PASSES = int(os.environ.get("KEYRING_RETRY_PASSES", 2))
KEYRING_DRAIN_TIMEOUT_SECONDS = int(os.environ.get("KEYRING_DRAIN_TIMEOUT_SECONDS", 900))
# Keyring usage below which a repair may start a resume.
# Note: once the docker runtime was configured to stop creating session keys, the
# quota no longer moves in normal operation. Preventive measures based on the
# remaining quota only ever misfired, so they were removed; what is left is the
# after-the-fact repair that runs only when a keyring failure is actually seen.
KEYRING_HEADROOM_FRACTION = float(os.environ.get("KEYRING_HEADROOM_FRACTION", 0.5))

# The model that writes the candidates. Pinned by exact id rather than the
# "opus" alias: the alias resolved to claude-opus-4-6 under CLI 2.1.56 and to
# claude-opus-5 under 2.1.220. Changing the proposer changes the search itself,
# so this must not drift between runs.
PROPOSER_MODEL = "claude-opus-5"

# The Claude Code CLI version the proposer runs on. Its behaviour depends on the
# version (between 2.1.56 and 2.1.220 the output token count roughly doubled), so
# it is pinned. This is the version the runs reported in the paper used.
CLAUDE_CODE_VERSION = "2.1.220"

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


def _handle_signal(signum, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupted, finishing current step...", flush=True)


def run_cmd(cmd, timeout=HARBOR_TIMEOUT_SECONDS, cwd=None):
    """Run a subprocess; return CompletedProcess (returncode=124 on timeout)."""
    env = os.environ.copy()
    env["HARBOR_MODEL"] = MODEL
    # Ensure dotenv-loaded keys survive through uv run subprocess
    for key in ("ANTHROPIC_API_KEY",):
        val = os.environ.get(key)
        if val:
            env[key] = val
    try:
        return subprocess.run(
            cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True, env=env
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"Timed out after {timeout}s")


def validate_agent_class(import_path):
    """Load and return the exact Terminus2 subclass named by an import path."""
    module_name, separator, class_name = import_path.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("agent import path must be MODULE:CLASS")

    module = importlib.import_module(module_name)
    agent_class = getattr(module, class_name)
    if not inspect.isclass(agent_class):
        raise TypeError(f"{import_path} is not a class")

    from harbor.agents.terminus_2.terminus_2 import Terminus2

    if not issubclass(agent_class, Terminus2):
        raise TypeError(f"{import_path} must subclass Terminus2")
    return agent_class


def _job_progress(job_dir):
    """Scan trial dirs: return (n_terminal_trials, newest_activity_mtime).

    A trial is terminal once it has result.json or exception.txt (a crashed
    trial writes exception.txt but never result.json).
    """
    terminal = 0
    newest = 0.0
    if not job_dir.exists():
        return 0, 0.0
    candidates = [job_dir / "job.log", job_dir / "result.json"]
    for td in job_dir.iterdir():
        if not td.is_dir() or "__" not in td.name:
            continue
        if (td / "result.json").exists() or (td / "exception.txt").exists():
            terminal += 1
        candidates += [
            td / "trial.log",
            td / "result.json",
            td / "exception.txt",
            td / "agent" / "trajectory.json",
        ]
    for f in candidates:
        try:
            newest = max(newest, f.stat().st_mtime)
        except OSError:
            pass
    return terminal, newest


def _kill_process_group(proc):
    """SIGTERM the process group, escalate to SIGKILL if it lingers."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def harbor_run(import_path, job_name, n_trials=2, n_concurrent=10, tasks=None):
    """Run Harbor on the paper TB2 config using the selected environment.

    tasks: explicit task list (e.g. from a --task-selector extension). When set,
    it replaces the EVAL_TASK_SET named set via explicit -i flags.

    The result is None if Harbor fails, times out, or stalls mid-job.
    Callers do not parse partial results from timed-out jobs.

    A watchdog kills harbor if it keeps running after every trial has reached
    a terminal state; that case returns success since results are on disk.
    """
    base = _code_base()
    cmd = [
        str(base / "scripts" / "run_eval.sh"),
        import_path,
        "full" if tasks else EVAL_TASK_SET,  # "full" adds no task flags of its own
        str(n_trials),
        str(n_concurrent),
        "--job-name",
        job_name,
        "--jobs-dir",
        str(JOBS_DIR),
    ]
    for task in tasks or []:
        cmd += ["-i", task]

    env = os.environ.copy()
    env["HARBOR_MODEL"] = MODEL
    env["HARBOR_TIMEOUT_SECONDS"] = str(HARBOR_TIMEOUT_SECONDS)
    env["HARBOR_EXTERNAL_TIMEOUT"] = "1"

    job_dir = JOBS_DIR / job_name
    expected_trials = (len(tasks) if tasks else N_EVAL_TASKS) * n_trials

    proc = subprocess.Popen(
        cmd,
        cwd=str(base),
        stdout=None,
        stderr=None,
        env=env,
        start_new_session=True,
    )
    start = time.time()
    complete_since = None
    kill_reason = None
    while True:
        try:
            proc.wait(timeout=30)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.time()
        terminal, newest_mtime = _job_progress(job_dir)
        if now - start > HARBOR_TIMEOUT_SECONDS:
            kill_reason = "timed out"
        elif terminal >= expected_trials:
            complete_since = complete_since or now
            if now - complete_since > HARBOR_COMPLETE_GRACE_SECONDS:
                kill_reason = "complete but hung"
        else:
            complete_since = None
            if newest_mtime and now - newest_mtime > HARBOR_STALL_TIMEOUT_SECONDS:
                kill_reason = "stalled"
        if kill_reason:
            print(
                f"  {_yellow('watchdog')}: harbor {kill_reason} "
                f"({terminal}/{expected_trials} trials terminal), killing job={job_name}",
                flush=True,
            )
            _kill_process_group(proc)
            break

    if kill_reason == "complete but hung":
        repair_keyring_trials(job_dir)
        repair_retryable_trials(job_dir)
        return job_dir, True  # all trials terminal; results are on disk
    if kill_reason is not None:
        print(f"  {_yellow('harbor ' + kill_reason)} job={job_name}")
        return job_dir, None
    if proc.returncode != 0:
        print(f"  {_red('harbor failed')} exit={proc.returncode} job={job_name}")
        return job_dir, None

    repair_keyring_trials(job_dir)
    repair_retryable_trials(job_dir)
    return job_dir, True  # signal success; callers use parse_job_results directly


def _resume_env(base):
    """Environment for `harbor job resume`.

    This path calls harbor directly instead of going through run_eval.sh, so the
    PYTHONPATH used to import the agent and the endpoint of a locally served
    model have to be set here. Without them the client falls back to the public
    endpoint and fails to find the model.
    """
    env = os.environ.copy()
    env["HARBOR_MODEL"] = MODEL
    env["PYTHONPATH"] = str(base) + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    if MODEL.startswith("openai/"):
        env.setdefault(
            "OPENAI_API_BASE",
            os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
        )
        env.setdefault("OPENAI_API_KEY", "dummy")
    return env


def _keyring_usage():
    """Return (used, max) keyring quota for this UID, or (None, None).

    Line format of /proc/key-users:
        "uid: usage nkeys/nikeys qnkeys/maxkeys qnbytes/maxbytes"
    """
    try:
        for line in Path("/proc/key-users").read_text().splitlines():
            fields = line.split()
            if fields and fields[0].rstrip(":") == str(os.getuid()):
                used, quota = fields[3].split("/")
                return int(used), int(quota)
    except (OSError, ValueError, IndexError):
        pass
    return None, None


def _wait_for_keyring_headroom(context):
    """Wait for keyring usage to fall below the threshold, up to a timeout.

    Starting while the quota is exhausted kills every trial immediately, so this
    runs before a start and before a resume. On timeout it warns and continues:
    another job holding keys under the same uid can keep usage high indefinitely.
    """
    deadline = time.time() + KEYRING_DRAIN_TIMEOUT_SECONDS
    announced = False
    while True:
        used, quota = _keyring_usage()
        if used is None or used <= quota * KEYRING_HEADROOM_FRACTION:
            if announced:
                print(f"  keyring: recovered to {used}/{quota}, proceeding", flush=True)
            return
        if time.time() > deadline:
            print(
                f"  {_yellow('keyring')}: still {used}/{quota} after "
                f"{_elapsed(KEYRING_DRAIN_TIMEOUT_SECONDS)} ({context}), proceeding anyway",
                flush=True,
            )
            return
        if not announced:
            print(
                f"  {_yellow('keyring')}: {used}/{quota} keys in use, "
                f"waiting for GC ({context})",
                flush=True,
            )
            announced = True
        time.sleep(30)


def _keyring_failed_trials(job_dir):
    """Trials killed by a transient environment failure: [(trial_dir, message)].

    Keyring exhaustion and tmux startup failures raise RuntimeError, the same
    type as a bug in a candidate, so harbor's type filter cannot separate them.
    They are matched on the message instead, against INFRA_ERROR_SIGNATURES.
    """
    trials = []
    for rf in job_dir.glob("*/result.json"):
        try:
            r = json.loads(rf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        msg = (r.get("exception_info") or {}).get("exception_message") or ""
        for signature in INFRA_ERROR_SIGNATURES:
            if all(sig in msg for sig in signature):
                trials.append((rf.parent, msg))
                break
    return trials


def repair_keyring_trials(job_dir):
    """Delete and re-run trials killed by a transient environment failure.

    Only trials whose message matches INFRA_ERROR_SIGNATURES are touched; a
    RuntimeError from a bug in a candidate is left alone. Keyring keys are freed
    by garbage collection, so resuming after waiting for headroom goes through.
    tmux failures are contention during a burst of container starts and clear on
    a plain re-run. If the quota is exhausted again mid-resume, the next pass
    picks it up.

    `harbor job resume` can itself crash while cleaning up an unrelated trial
    directory: its empty-tree removal has a gap between the emptiness check and
    the rmdir, which can race with asynchronous cleanup after a watchdog kill. In
    one observed case the resume exited without re-running a single target trial.
    That crash is invisible to the message matching above, because the target
    directories are already gone and so look like they do not exist, leaving no
    residual warning either. So a non-zero exit is retried once after a short
    wait, and the deleted trials are verified directly by checking that their
    directories came back.
    """
    if not job_dir.exists():
        return
    base = _code_base()
    for i in range(KEYRING_RETRY_PASSES):
        trials = _keyring_failed_trials(job_dir)
        if not trials:
            return
        n_keyring = sum(
            1 for _, msg in trials
            if all(sig in msg for sig in KEYRING_ERROR_SIGNATURE)
        )
        print(
            f"  {_yellow('infra errors')}: {len(trials)} trials "
            f"({n_keyring} keyring, {len(trials) - n_keyring} tmux/other), "
            f"resume pass {i + 1}/{KEYRING_RETRY_PASSES} job={job_dir.name}",
            flush=True,
        )
        if n_keyring:
            _wait_for_keyring_headroom(f"resume {job_dir.name}")
        targeted_names = {td.name for td, _ in trials}
        for td, _ in trials:
            shutil.rmtree(td, ignore_errors=True)
        for attempt in range(2):  # crash retry: resume itself can OSError (TOCTOU)
            proc = subprocess.Popen(
                ["uv", "run", "harbor", "job", "resume", "-p", str(job_dir)],
                cwd=str(base),
                env=_resume_env(base),
                start_new_session=True,
            )
            try:
                proc.wait(timeout=HARBOR_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                print(f"  {_yellow('infra resume timed out')} job={job_dir.name}")
                _kill_process_group(proc)
                break
            if proc.returncode == 0:
                break
            if attempt == 0:
                print(
                    f"  {_yellow('resume crashed')} (exit {proc.returncode}) "
                    f"job={job_dir.name}, retrying once in 10s",
                    flush=True,
                )
                time.sleep(10)
            else:
                print(
                    f"  {_yellow('resume crashed again')} (exit {proc.returncode}) "
                    f"job={job_dir.name}, giving up this pass",
                    flush=True,
                )
        vanished = sorted(
            name for name in targeted_names
            if not (job_dir / name / "result.json").exists()
        )
        if vanished:
            print(
                f"  {_yellow('VANISHED')}: {len(vanished)} trial dir(s) did not "
                f"come back after resume ({', '.join(vanished[:5])}"
                f"{', ...' if len(vanished) > 5 else ''})",
                flush=True,
            )
    trials = _keyring_failed_trials(job_dir)
    if trials:
        print(
            f"  {_yellow('RESIDUAL')} infra failures: {len(trials)} trials "
            f"remain after {KEYRING_RETRY_PASSES} passes (scored as 0). "
            f"permanent fix: raise kernel.keys.maxkeys",
            flush=True,
        )


def _count_retryable_trials(job_dir):
    """Trials whose exception_type is in TRIAL_RETRY_EXCEPTIONS."""
    count = 0
    for rf in job_dir.glob("*/result.json"):
        try:
            r = json.loads(rf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if (r.get("exception_info") or {}).get("exception_type") in TRIAL_RETRY_EXCEPTIONS:
            count += 1
    return count


def repair_retryable_trials(job_dir):
    """Repair transient errors that a re-roll clears.

    `harbor job resume -f <type>` deletes the matching trials and re-runs them.
    The baselines get the same repair, so applying it to candidates too is what
    keeps the comparison fair; skipping it would penalise the candidate side.
    The smaller the subset, the more one contaminated trial distorts selection:
    about 1.7 points over 60 trials, about 17 points over 6. Only the failed
    trials (~2%) are re-run, so the cost is negligible.
    """
    if not job_dir.exists():
        return
    base = _code_base()
    n = _count_retryable_trials(job_dir)
    for i in range(TRIAL_RETRY_PASSES):
        if n == 0:
            return
        print(
            f"  retryable {'/'.join(TRIAL_RETRY_EXCEPTIONS)}: {n} trials, "
            f"resume pass {i + 1}/{TRIAL_RETRY_PASSES} job={job_dir.name}",
            flush=True,
        )
        env = _resume_env(base)
        cmd = ["uv", "run", "harbor", "job", "resume", "-p", str(job_dir)]
        for exc_type in TRIAL_RETRY_EXCEPTIONS:
            cmd += ["-f", exc_type]
        proc = subprocess.Popen(cmd, cwd=str(base), env=env, start_new_session=True)
        try:
            proc.wait(timeout=TRIAL_RETRY_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            print(f"  {_yellow('repair pass timed out')} job={job_dir.name}")
            _kill_process_group(proc)
        n = _count_retryable_trials(job_dir)
    if n:
        print(
            f"  {_yellow('RESIDUAL')} {'/'.join(TRIAL_RETRY_EXCEPTIONS)}: {n} trials "
            f"remain after {TRIAL_RETRY_PASSES} passes (scored as 0)",
            flush=True,
        )


def check_fatal_api_errors(job_dir):
    """Abort the whole run when trials died on unrecoverable API errors.

    When credit runs out every rollout fails, but the proposer authenticates
    separately and keeps going, so the loop would spend the night proposing
    against nothing. Any fatal error stops the run immediately.

    Errors are recorded in two places: exception.txt for crashes, and
    exception_info inside a trial's result.json for API errors. An exhausted
    credit balance is recorded only in the latter, and an earlier version of this
    check that scanned only exception.txt missed it and burned three iterations.
    """
    fatal_markers = (
        "credit balance is too low",   # Anthropic
        "insufficient credits",        # OpenRouter 402
        "invalid x-api-key",
        "authentication_error",
        "invalid api key",
        "no auth credentials",
    )
    if not job_dir.exists():
        return

    sources = []
    for exc in list(job_dir.glob("*/exception.txt"))[:5]:
        try:
            sources.append((exc, exc.read_text()[:4000].lower()))
        except OSError:
            continue
    for rf in list(job_dir.glob("*/result.json"))[:10]:
        try:
            r = json.loads(rf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        e = r.get("exception_info")
        if e:
            sources.append((rf, json.dumps(e)[:4000].lower()))

    for src, text in sources:
        for marker in fatal_markers:
            if marker in text:
                print(
                    f"\n{_red('FATAL')} unrecoverable API error in {src}:\n"
                    f"  '{marker}' — aborting the run (fix billing/auth, then resume)"
                )
                raise SystemExit(2)


def parse_job_results(job_dir, expected_trials=None):
    """Parse per-task rewards from a harbor job directory.

    Always reads individual trial dirs (ground truth). Ignores top-level result.json
    which may be a stale snapshot written before all trials complete.

    Every trial dir counts: no result.json, no verifier_result, corrupt JSON,
    errored trials — all count as reward=0. This matches harbor's metric
    (total_passes / total_trials).

    Returns dict: {task_name: [reward1, reward2, ...]}
    """
    task_rewards = {}

    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir() or "__" not in trial_dir.name:
            continue
        task = trial_dir.name.rsplit("__", 1)[0]

        rf = trial_dir / "result.json"
        if not rf.exists():
            task_rewards.setdefault(task, []).append(0.0)
            continue
        try:
            r = json.loads(rf.read_text())
        except (json.JSONDecodeError, OSError):
            task_rewards.setdefault(task, []).append(0.0)
            continue

        vr = r.get("verifier_result") or {}
        reward = (vr.get("rewards") or {}).get("reward")
        task_rewards.setdefault(task, []).append(
            float(reward) if reward is not None else 0.0
        )

    # Validate trial counts
    if expected_trials:
        for task, rewards in task_rewards.items():
            if len(rewards) != expected_trials:
                print(
                    f"  {_yellow('warning')}: {task} has {len(rewards)}/{expected_trials} trials"
                )

    return task_rewards


def compute_pass_rates(task_rewards):
    """Compute pass rate per task and flat average. Returns (per_task, avg).

    avg is total_passes / total_trials (flat, matching harbor's metric),
    NOT the mean of per-task rates.
    """
    per_task = {}
    total_passes = 0
    total_trials = 0
    for task, rewards in task_rewards.items():
        per_task[task] = sum(r > 0 for r in rewards) / len(rewards) if rewards else 0.0
        total_passes += sum(r > 0 for r in rewards)
        total_trials += len(rewards)

    avg = total_passes / total_trials if total_trials else 0.0
    return per_task, avg


def parse_trial_metrics(job_dir):
    """Extract per-trial rollout metrics from a harbor job directory.

    Returns dict: {task_name: [trial_metrics, ...]}
    where trial_metrics = {n_input_tokens, n_output_tokens, n_cache_tokens,
                           cost_usd, n_turns, n_api_calls, reward}
    """
    per_task = {}
    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir() or "__" not in trial_dir.name:
            continue
        task = trial_dir.name.rsplit("__", 1)[0]
        rf = trial_dir / "result.json"
        if not rf.exists():
            continue
        try:
            r = json.loads(rf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        ar = r.get("agent_result") or {}
        md = ar.get("metadata") or {}
        vr = r.get("verifier_result") or {}
        reward = (vr.get("rewards") or {}).get("reward")
        metrics = {
            "n_input_tokens": ar.get("n_input_tokens"),
            "n_output_tokens": ar.get("n_output_tokens"),
            "n_cache_tokens": ar.get("n_cache_tokens"),
            "cost_usd": ar.get("cost_usd"),
            "n_turns": md.get("n_episodes"),
            "n_api_calls": len(md.get("api_request_times_msec", [])),
            "reward": reward,
        }
        per_task.setdefault(task, []).append(metrics)
    return per_task


def summarize_trial_metrics(trial_metrics):
    """Aggregate per-trial metrics into a summary dict."""
    all_costs = []
    all_input = []
    all_output = []
    all_cache = []
    all_turns = []
    per_task_summary = {}

    for task, trials in trial_metrics.items():
        task_costs = [t["cost_usd"] for t in trials if t["cost_usd"] is not None]
        task_turns = [t["n_turns"] for t in trials if t["n_turns"] is not None]
        per_task_summary[task] = {
            "mean_cost": round(sum(task_costs) / len(task_costs), 3)
            if task_costs
            else None,
            "mean_turns": round(sum(task_turns) / len(task_turns), 1)
            if task_turns
            else None,
            "n_trials": len(trials),
        }
        all_costs.extend(task_costs)
        all_turns.extend(task_turns)
        all_input.extend(
            t["n_input_tokens"] for t in trials if t["n_input_tokens"] is not None
        )
        all_output.extend(
            t["n_output_tokens"] for t in trials if t["n_output_tokens"] is not None
        )
        all_cache.extend(
            t["n_cache_tokens"] for t in trials if t["n_cache_tokens"] is not None
        )

    n_trials = sum(len(v) for v in trial_metrics.values())
    return {
        "n_trials": n_trials,
        "total_cost_usd": round(sum(all_costs), 2) if all_costs else None,
        "mean_cost_usd": round(sum(all_costs) / len(all_costs), 3)
        if all_costs
        else None,
        "total_input_tokens": sum(all_input) if all_input else None,
        "total_output_tokens": sum(all_output) if all_output else None,
        "total_cache_tokens": sum(all_cache) if all_cache else None,
        "mean_turns": round(sum(all_turns) / len(all_turns), 1) if all_turns else None,
        "per_task": per_task_summary,
    }


def count_iterations():
    """Highest iteration in evolution_summary.jsonl (for resume)."""
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


def build_task_history(seed=None):
    """Per-task pass-rate observations across every evaluated agent.

    Combines the baseline seed ({task: [rate, ...]}) with all rows of
    evolution_summary.jsonl, so resumed runs recover full history.
    Returns {task: [rate, ...]} for task-selector extensions.
    """
    history = {task: list(rates) for task, rates in (seed or {}).items()}
    if EVOLUTION_SUMMARY.exists():
        for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for task, rate in (row.get("per_task") or {}).items():
                history.setdefault(task, []).append(rate)
    return history


def update_frontier(candidates_results, partial=False, est_scores=None, iteration=None):
    """Update frontier_val.json with best agent per task and overall best.

    partial=True marks a subset evaluation (task selector active): per-task
    entries still update, but _best does not — a subset average is not
    comparable to full-set averages. Instead _best_est tracks the highest
    score seen so far: est_full_rate when est_scores are available
    (--anchored-summary), or else the raw subset avg_pass_rate as a naive
    stand-in (task-selector runs with no estimator configured) — without this fallback _best_est never
    updates at all and every iteration proposes against the frozen Phase-0
    baseline instead of the run's actual best-so-far. Kept as a SEPARATE
    field from _best (never overwritten) so estimated/proxy and measured
    scores are never conflated: _best is real full-set (or Phase-0 baseline)
    measurement, _best_est is the search-time working estimate.
    """
    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    est_scores = est_scores or {}

    for agent_name, (per_task, avg) in candidates_results.items():
        for task, rate in per_task.items():
            current_best = frontier.get(task, {}).get("pass_rate", -1)
            if rate > current_best:
                frontier[task] = {
                    "best_agent": agent_name,
                    "pass_rate": rate,
                }

        if partial:
            est = est_scores.get(agent_name)
            if est is not None:
                new_est, new_se = est["est_full_rate"], est["est_full_se"]
            else:
                # A task-selector run without --anchored-summary has no
                # design-unbiased est_full_rate, so this iteration's raw subset
                # average stands in for it. Without that stand-in neither _best
                # nor _best_est ever updates, and every iteration forks from the
                # Phase 0 baseline instead of from the run's own best so far;
                # that was observed in practice, with all ten iterations forking
                # from the baseline. The subset changes each iteration, so this
                # is not a strict comparison, but it is the same quantity such a
                # run reports as its score.
                new_est, new_se = avg, 0.0
            # Compare against _best_est when it exists, otherwise against the
            # measured baseline. Starting from -1 instead would accept the first
            # estimate unconditionally, letting a candidate weaker than the
            # baseline claim the frontier just by being first.
            current_best_est = frontier.get("_best_est", {}).get(
                "est_full_rate",
                frontier.get("_best", {}).get("avg_pass_rate", -1),
            )
            # A plain > comparison. Requiring the new estimate to clear the
            # standard-error interval of the old one would make an early upward
            # fluctuation impossible to correct, so every later candidate would
            # descend from a parent that is actually weak: in one run an
            # iteration-1 estimate of 48.2% measured 30.3% and still remained the
            # parent for nine iterations. Ties between overlapping intervals are
            # not resolved here; that is left to the analysis step of the
            # proposer's instructions.
            if new_est > current_best_est:
                frontier["_best_est"] = {
                    "agent": agent_name,
                    "est_full_rate": new_est,
                    "est_full_se": new_se,
                    "iteration": iteration,
                }
            continue
        current_best_avg = frontier.get("_best", {}).get("avg_pass_rate", -1)
        # strict > keeps the FIRST agent on ties, matching text_classification's
        # frontier semantics. The latest-tie-break selection protocol lives in
        # select_winner() (analysis layer), not here.
        if avg > current_best_avg:
            frontier["_best"] = {
                "agent": agent_name,
                "avg_pass_rate": avg,
            }

    FRONTIER_VAL.write_text(json.dumps(frontier, indent=2))


def update_evolution_summary(
    iteration,
    candidates,
    results,
    propose_time=None,
    bench_time=None,
    metrics=None,
    selected_tasks=None,
    est_scores=None,
):
    """Append one JSONL row per candidate.

    selected_tasks: task subset used this iteration (task selector active).
    Subset averages are not comparable to the full-set frontier, so delta is
    omitted and the outcome is labeled with the subset size.

    est_scores: {name: {est_full_rate, est_full_se, meta}} from the summary
    estimator (--anchored-summary). est_full_rate is a full-set-equivalent
    estimate, comparable across iterations regardless of the sampled subset.
    """
    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    best_avg = frontier.get("_best", {}).get("avg_pass_rate", 0)
    metrics = metrics or {}
    est_scores = est_scores or {}

    with open(EVOLUTION_SUMMARY, "a") as f:
        for i, c in enumerate(candidates):
            name = c["name"]
            per_task, avg = results.get(name, ({}, 0))
            if selected_tasks:
                delta = None
                outcome = (
                    f"{avg:.1%} on {len(selected_tasks)}-task subset"
                    if avg > 0
                    else "failed"
                )
                if avg > 0 and name in est_scores:
                    e = est_scores[name]
                    outcome = (
                        f"est_full {e['est_full_rate']:.1%}"
                        f"±{e['est_full_se']:.1%} | {outcome}"
                    )
            else:
                delta = round(avg - best_avg, 3) if best_avg else None
                outcome = f"{avg:.1%} ({avg - best_avg:+.1%})" if avg > 0 else "failed"
            row = {
                "iteration": iteration,
                "agent": name,
                "import_path": c.get("import_path", ""),
                "avg_pass_rate": round(avg, 3),
                "per_task": {k: round(v, 3) for k, v in per_task.items()},
                "hypothesis": c.get("hypothesis", ""),
                "changes": c.get("changes", ""),
                "delta": delta,
                "outcome": outcome,
            }
            if i == 0 and propose_time is not None:
                row["timing_s"] = {
                    "propose": round(propose_time, 1),
                    "bench": round(bench_time, 1) if bench_time else None,
                }
            if i == 0 and selected_tasks:
                row["selected_tasks"] = selected_tasks
            if name in est_scores:
                row["est_full_rate"] = est_scores[name]["est_full_rate"]
                row["est_full_se"] = est_scores[name]["est_full_se"]
            if name in metrics:
                row["rollout_metrics"] = metrics[name]
            f.write(json.dumps(row) + "\n")


def propose_claude(
    task_prompt, iteration, timeout=2400, skill_name="meta-harness-terminal-bench-2"
):
    """Call Claude Code to propose new agent candidates. Returns True if pending_eval.json exists."""
    os.environ.pop("CLAUDECODE", None)
    # tb2 proposer uses a dedicated Claude account, separate from text_classification.
    # One-time setup: CLAUDE_CONFIG_DIR=~/.claude-tb2 claude → /login
    os.environ["CLAUDE_CONFIG_DIR"] = os.environ.get(
        "TB2_CLAUDE_CONFIG_DIR", str(Path.home() / ".claude-tb2")
    )
    # Strip API key so claude CLI uses subscription auth, not API (avoids rate limits)
    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    skill_dir = _code_base() / ".claude/skills" / skill_name
    if not skill_dir.is_dir():
        # Snapshots frozen before the skill split only ship the selector skill;
        # resumed runs keep using it so the run stays internally consistent.
        skill_dir = _code_base() / ".claude/skills/meta-harness-terminal-bench-2"
    claude_wrapper.ensure_version(CLAUDE_CODE_VERSION)
    result = claude_wrapper.run(
        prompt=task_prompt,
        model=PROPOSER_MODEL,
        allowed_tools=PROPOSER_ALLOWED_TOOLS,
        skills=[str(skill_dir)],
        cwd=str(_code_base()),
        log_dir=str(LOGS_DIR / "claude_sessions"),
        name=f"iter{iteration}",
        timeout_seconds=timeout,
        effort="high",  # "max" requires API billing; not available on Claude.ai subscription auth
    )
    # Restore API key for harbor eval runs
    if saved_key:
        os.environ["ANTHROPIC_API_KEY"] = saved_key
    if result.exit_code != 0:
        print(f"  {_red('proposer failed')} exit={result.exit_code}")
        if result.stderr:
            print(f"  {_dim(result.stderr[:500])}")
        return False
    result.show()
    return PENDING_EVAL.exists()


def validate_candidate(name, import_path):
    """Import-check a candidate agent. Returns True if valid."""
    result = run_cmd(
        [
            "uv",
            "run",
            "python",
            str(_code_base() / "meta_harness.py"),
            "--validate-agent",
            import_path,
        ],
        cwd=str(_code_base()),
        timeout=30,
    )
    if result.returncode == 0 and "OK" in result.stdout:
        return True
    print(f"  {_red('import FAIL')}: {name}")
    if result.stderr:
        print(f"    {_dim(result.stderr[:300])}")
    return False


def smoke_test(name, import_path, timeout=1800):
    """Run 1 trial on 1 task to check for runtime crashes. Returns True if passed.

    With only one trial, hitting one of the ~2% transient errors would reject a
    healthy candidate. In that case only, the check is run once more.
    """
    job_name = f"smoke-{name}"
    job_dir = JOBS_DIR / job_name

    cmd = [
        str(_code_base() / "scripts" / "run_eval.sh"),
        import_path,
        "full",
        "1",
        "1",
        "-i",
        SMOKE_TEST_TASK,
        "--job-name",
        job_name,
        "--jobs-dir",
        str(JOBS_DIR),
    ]
    for attempt in range(2):
        if job_dir.exists():
            run_cmd(["rm", "-rf", str(job_dir)])
        t0 = time.time()
        result = run_cmd(cmd, timeout=timeout, cwd=str(_code_base()))
        elapsed = time.time() - t0

        failure = None
        if result.returncode != 0:
            failure = f"exit={result.returncode} ({_elapsed(elapsed)})"
        elif not (job_dir / "result.json").exists():
            failure = f"no result.json ({_elapsed(elapsed)})"
        else:
            data = json.loads((job_dir / "result.json").read_text())
            n_errors = data.get("stats", {}).get("n_errors", 0)
            if n_errors > 0:
                failure = f"{n_errors} errors ({_elapsed(elapsed)})"

        if failure is None:
            print(f"  {_green('smoke OK')}: {name} ({_elapsed(elapsed)})")
            return True
        if attempt == 0 and _count_retryable_trials(job_dir):
            print(
                f"  {_yellow('smoke retry')}: {name} hit a retryable "
                f"{'/'.join(TRIAL_RETRY_EXCEPTIONS)}, re-rolling once"
            )
            continue
        print(f"  {_red('smoke FAIL')}: {name} {failure}")
        if result.stderr:
            print(f"    {_dim(result.stderr[:300])}")
        return False
    return False


def render_task_prompt(iteration, n_trials):
    """Build the prompt for the proposer Claude session."""
    return (
        f"Run iteration {iteration} of the scaffold evolution loop (KIRA track). "
        f"Model: {MODEL} (Opus). "
        f"Start from the current top-performing base system (see the `_best` "
        f"entry in frontier_val.json below) as the parent.\n\n"
        f"## Eval split: {N_EVAL_TASKS} official TB2 tasks x {n_trials} trials\n\n"
        f"This reference example uses the full TB2 dataset from the paper runs. "
        f"Focus on scaffold changes that help the agent solve complex, long-horizon tasks.\n\n"
        f"## Run directories\n"
        f"All logs and results for this run are under `{LOGS_DIR}/`.\n"
        f"- `{LOGS_DIR / 'evolution_summary.jsonl'}` — past results\n"
        f"- `{LOGS_DIR / 'frontier_val.json'}` — frontier\n"
        f"- `{LOGS_DIR / 'reports'}/` — post-eval reports\n"
        f"- Write pending_eval.json to: `{PENDING_EVAL}`\n"
        f"- Project root for THIS run's code (read agents here): `{_code_base()}`\n"
        f"- Write candidate files ONLY to: `{_code_base() / 'agents'}/<name>.py`"
    )


def snapshot_code(snap_dir):
    """Copy the project code into logs/<run>/code/terminal_bench_2.

    Freezes everything subprocesses re-read from disk (agents baselines,
    scripts/run_eval.sh, the proposer skill, extensions). logs/, jobs/, .venv
    and stray candidate files in agents/ are excluded; the run proposes its
    own candidates into the copy. ~2 MB per run.
    """

    def _ignore(src, names):
        drop = {
            n
            for n in names
            if n
            in {
                "logs",
                "jobs",
                "__pycache__",
                ".pytest_cache",
                "offline_experience",
            }
            or n.startswith(".venv")
        }
        if Path(src) == AGENTS_SOURCE_DIR:
            drop |= {
                n for n in names if n.endswith(".py") and n not in BASELINE_FILES
            }
        if Path(src) == TEMPLATES_SOURCE_DIR:
            drop |= {
                n
                for n in names
                if n.endswith(".txt") and n not in BASELINE_TEMPLATES
            }
        return drop

    shutil.copytree(EVOLVE_DIR, snap_dir, ignore=_ignore, symlinks=True)


# snapshot_code needs the pristine source dirs even after AGENTS_DIR /
# TEMPLATES_DIR are switched to the snapshot copies.
AGENTS_SOURCE_DIR = AGENTS_DIR
TEMPLATES_SOURCE_DIR = TEMPLATES_DIR


def _code_base():
    """Directory whose code subprocesses should execute (snapshot when set)."""
    return CODE_SNAPSHOT or EVOLVE_DIR


def rescue_misplaced_candidates(candidate_names):
    """Move THIS iteration's candidates the proposer misplaced into main agents/.

    With a code snapshot the proposer works in logs/<run>/code, but it may
    still resolve absolute paths to the main repo and write candidates there.
    Only files named in the current pending_eval are rescued.
    """
    if not CODE_SNAPSHOT:
        return
    moved = []
    for name in candidate_names:
        f = AGENTS_SOURCE_DIR / f"{name}.py"
        if f.name not in BASELINE_FILES and f.exists():
            f.replace(AGENTS_DIR / f.name)
            moved.append(f.name)
        # KIRA candidates pair with a prompt template (underscores or hyphens)
        for tname in {f"{name}.txt", f"{name.replace('_', '-')}.txt"}:
            t = TEMPLATES_SOURCE_DIR / tname
            if tname not in BASELINE_TEMPLATES and t.exists():
                TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
                t.replace(TEMPLATES_DIR / tname)
                moved.append(tname)
    if moved:
        print(
            f"  {_yellow('rescued')} {len(moved)} misplaced file(s) from the "
            f"main repo: {', '.join(moved)}"
        )


def _candidate_files():
    return [f for f in AGENTS_DIR.glob("*.py") if f.name not in BASELINE_FILES]


def _candidate_templates():
    return [
        f for f in TEMPLATES_DIR.glob("*.txt") if f.name not in BASELINE_TEMPLATES
    ]


def restore_candidates():
    """Move this run's archived candidates back into agents/ (resume).

    Candidate prompt templates (prompt-templates/<name>.txt) travel with
    their agents.
    """
    if ARCHIVED_AGENTS.exists():
        files = list(ARCHIVED_AGENTS.glob("*.py"))
        for f in files:
            shutil.move(str(f), str(AGENTS_DIR / f.name))
        if files:
            print(f"  Restored {len(files)} candidate file(s) from {ARCHIVED_AGENTS}")
    if ARCHIVED_TEMPLATES and ARCHIVED_TEMPLATES.exists():
        files = list(ARCHIVED_TEMPLATES.glob("*.txt"))
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.move(str(f), str(TEMPLATES_DIR / f.name))
        if files:
            print(
                f"  Restored {len(files)} prompt template(s) from {ARCHIVED_TEMPLATES}"
            )


def archive_candidates():
    """Move proposed candidates from agents/ into this run's dir.

    Keeps agents/ and prompt-templates/ at baselines-only between runs, so
    runs cannot clobber each other's files and each run dir snapshots its own
    candidate code + templates.
    """
    if ARCHIVED_AGENTS is None:
        return
    files = _candidate_files()
    if files:
        ARCHIVED_AGENTS.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.move(str(f), str(ARCHIVED_AGENTS / f.name))
        print(f"  Archived {len(files)} candidate file(s) to {ARCHIVED_AGENTS}")
    if ARCHIVED_TEMPLATES is not None:
        templates = _candidate_templates()
        if templates:
            ARCHIVED_TEMPLATES.mkdir(parents=True, exist_ok=True)
            for f in templates:
                shutil.move(str(f), str(ARCHIVED_TEMPLATES / f.name))
            print(
                f"  Archived {len(templates)} prompt template(s) to {ARCHIVED_TEMPLATES}"
            )


def fresh_start():
    """Clear proposed agents (keep kira_baseline) and logs for a fresh run."""
    if AGENTS_DIR.exists():
        for f in AGENTS_DIR.iterdir():
            if f.name == "__pycache__" or f.name in BASELINE_FILES:
                continue
            if f.is_dir():
                run_cmd(["rm", "-rf", str(f)])
                print(f"  Cleared {f.name}/")
            elif f.suffix == ".py":
                f.unlink()
                print(f"  Cleared {f.name}")
    for f in _candidate_templates():
        f.unlink()
        print(f"  Cleared prompt-templates/{f.name}")

    for f in [EVOLUTION_SUMMARY, FRONTIER_VAL, PENDING_EVAL]:
        if f.exists():
            f.unlink()

    print("  Fresh start: cleared generated agents/ files and log files")


def load_baseline_history(trials):
    """Rebuild the Phase 0 baseline per-task history from cached job dirs."""
    history = {}
    for bl_name, _ in BASELINES:
        bl_dir = JOBS_DIR / f"{bl_name}-t{trials}"
        if bl_dir.exists():
            per_task, _ = compute_pass_rates(parse_job_results(bl_dir))
            for t, r in per_task.items():
                history.setdefault(t, []).append(r)
    return history


def select_winner():
    """Zero-cost final selection: pick the winner straight from the search log.

    Ranks by est_full_rate (the full-set estimate of §3.3 Phase 2) when
    present, at zero extra rollouts. Fixed-set runs have no
    est_full_rate; there raw avg_pass_rate is comparable across iterations,
    so it is the fallback.

    Final selection takes the MAX point estimate directly, with no SE-overlap
    tie-break: reliability/variety concerns are handled earlier, during search,
    by update_frontier()'s SE-aware _best_est update (a candidate only ever
    displaces the frontier there once it clearly clears the incumbent's SE
    band). By the time a run is finished, the top est_full_rate on record is
    the one to report; ties (exact score match) break to the EARLIEST
    iteration (the simpler/first-found candidate, not whichever happened to
    be proposed last). avg_pass_rate rows (no SE, i.e. naive fixed-subset
    runs) are where exact ties actually occur in practice — task-level scores
    there only take a handful of discrete values — and use the same
    earliest-iteration tie-break.
    Writes logs/<run>/winner.json.
    """
    rows = []
    if EVOLUTION_SUMMARY.exists():
        for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    rows = [r for r in rows if r.get("avg_pass_rate")]  # drop failed (0) candidates
    if not rows:
        print(f"  {_red('no evaluated candidates in evolution_summary')}; nothing to select")
        return None

    est_rows = [r for r in rows if r.get("est_full_rate") is not None]
    key = "est_full_rate" if est_rows else "avg_pass_rate"
    pool = est_rows or rows

    if key == "est_full_rate":
        best = max(pool, key=lambda r: (r["est_full_rate"], -r.get("iteration", 0)))
        tie_break = "max est_full_rate (point estimate), then earliest iteration on exact ties"
    else:
        best = max(pool, key=lambda r: (r[key], -r.get("iteration", 0)))
        tie_break = "earliest iteration"

    summary = {
        "method": key,
        "tie_break": tie_break,
        "winner": best["agent"],
        "iteration": best.get("iteration"),
        "score": best[key],
        "import_path": best.get("import_path", ""),
        "selected_at": datetime.now().isoformat(),
    }
    (LOGS_DIR / "winner.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\n{_ts()} {_bold('Winner selection (zero-cost)')}: {_bold(best['agent'])}  "
        f"iter{best.get('iteration')}  {key}={best[key]}"
    )
    print(f"  Wrote {LOGS_DIR / 'winner.json'}")
    print(f"  evaluate it with: bash scripts/eval_winner.sh {LOGS_DIR.name}")
    return summary


def run_evolve(args):
    global JOBS_DIR, LOGS_DIR, PENDING_EVAL, FRONTIER_VAL, EVOLUTION_SUMMARY
    global ARCHIVED_AGENTS, AGENTS_DIR, CODE_SNAPSHOT
    global ARCHIVED_TEMPLATES, TEMPLATES_DIR

    # Isolate run outputs under run-name subdirs
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    JOBS_DIR = EVOLVE_DIR / "jobs" / run_name
    LOGS_DIR = EVOLVE_DIR / "logs" / run_name
    PENDING_EVAL = LOGS_DIR / "pending_eval.json"
    FRONTIER_VAL = LOGS_DIR / "frontier_val.json"
    EVOLUTION_SUMMARY = LOGS_DIR / "evolution_summary.jsonl"
    ARCHIVED_AGENTS = LOGS_DIR / "agents"
    ARCHIVED_TEMPLATES = LOGS_DIR / "prompt-templates"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    # Code snapshot: freeze this run's code under logs/<run>/code. An existing
    # snapshot (resume) is always reused so the run keeps
    # executing the code it started with. --fresh re-snapshots.
    snap_parent = LOGS_DIR / "code"
    snap_dir = snap_parent / "terminal_bench_2"
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
        TEMPLATES_DIR = snap_dir / "prompt-templates"
        # run_eval.sh / harbor / validate reuse the main project's venv.
        # A caller-set UV_PROJECT_ENVIRONMENT wins, for an environment built
        # somewhere other than .venv.
        os.environ.setdefault("UV_PROJECT_ENVIRONMENT", str(EVOLVE_DIR / ".venv"))

    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    restore_candidates()

    if args.select_winner:
        select_winner()
        return

    if args.fresh:
        fresh_start()

    n_tasks = N_EVAL_TASKS
    print(
        f"{_ts()} {_bold('Evolution (KIRA track)')}  run={_cyan(run_name)}  model={_cyan(MODEL)}  iters={args.iterations}  trials={args.trials}  tasks={n_tasks}"
    )

    selector = None
    if args.task_selector:
        from extensions import get_task_selector

        selector = get_task_selector(args.task_selector)
        print(f"  task selector: {_cyan(args.task_selector)} ({selector.__name__})")

    estimator = None
    if args.anchored_summary:
        from extensions import get_summary_estimator

        estimator = get_summary_estimator(args.anchored_summary)
        print(
            f"  summary estimator: {_cyan(args.anchored_summary)} ({estimator.__name__})"
        )
        if getattr(estimator, "REQUIRES_INCLUSION", False) and not hasattr(
            selector, "inclusion_probs"
        ):
            print(
                f"  {_yellow('WARN')} estimator requires inclusion probabilities but "
                f"the task selector does not provide them; est_full will be skipped"
            )

    # ── Phase 0: Baselines ─────────────────────────────────────
    baseline_dirs = {}
    if not args.skip_baseline:
        print(
            f"\n{_ts()} {_bold('Phase 0: Baselines')}  agents={len(BASELINES)}  trials={args.trials}"
        )

    def _baseline_cache_ok(d, bl_name):
        """True if job dir d holds usable baseline results for MODEL/trials."""
        if not d.exists():
            return False
        cached_results = parse_job_results(d)
        if not cached_results:
            return False
        # harbor does not record n_attempts in config.json, so count actual
        # trial dirs per task instead.
        cached_counts = sorted({len(v) for v in cached_results.values()})
        if cached_counts != [args.trials]:
            print(
                f"  {_yellow('stale')} {bl_name} at {d}: trials mismatch "
                f"(cached={cached_counts}, want={args.trials})"
            )
            return False
        cfg_file = d / "config.json"
        if not cfg_file.exists():
            return True  # no config to check, trust it
        try:
            cfg = json.loads(cfg_file.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        # harbor writes the model under agents[0].model_name
        cfg_agents = cfg.get("agents") or [{}]
        cfg_model = (
            cfg.get("model")
            or cfg.get("agent", {}).get("model")
            or (
                cfg_agents[0].get("model_name", "")
                if isinstance(cfg_agents[0], dict)
                else ""
            )
        )
        if MODEL not in str(cfg_model):
            print(
                f"  {_yellow('stale')} {bl_name} at {d}: model mismatch "
                f"(cached={cfg_model}, want={MODEL})"
            )
            return False
        return True

    for bl_name, bl_import in BASELINES:
        bl_job = f"{bl_name}-t{args.trials}"
        bl_dir = JOBS_DIR / bl_job
        baseline_dirs[bl_name] = bl_dir

        if not args.skip_baseline:
            # Per-run job dir first, then the shared full-89 baseline rollouts
            # (per-task results are filtered to the active task set when read).
            cached_ok = False
            for cand_dir in (JOBS_DIR / bl_job, BASELINE_JOBS_SOURCE / bl_job):
                if _baseline_cache_ok(cand_dir, bl_name):
                    bl_dir = cand_dir
                    baseline_dirs[bl_name] = cand_dir
                    cached_ok = True
                    break
            if cached_ok:
                print(f"  {_dim('cached')} {bl_name}: {bl_dir}")
            else:
                print(
                    f"  {_ts()} running {_bold(bl_name)}: {n_tasks} tasks x {args.trials} trials...",
                    flush=True,
                )
                t0 = time.time()
                bl_dir, ok = harbor_run(
                    bl_import,
                    bl_job,
                    n_trials=args.trials,
                    n_concurrent=args.concurrent,
                )
                baseline_dirs[bl_name] = bl_dir
                elapsed = time.time() - t0
                if not ok:
                    print(
                        f"  {_red('FAIL')} {bl_name} crashed after {_elapsed(elapsed)}"
                    )
                else:
                    print(f"  {_ts()} {bl_name} completed in {_elapsed(elapsed)}")

    # Always seed frontier from baselines if results exist (even with --skip-baseline)
    active_tasks = _active_tasks()
    baseline_history = {}
    for bl_name, bl_dir in baseline_dirs.items():
        if bl_dir.exists():
            task_rewards = parse_job_results(bl_dir, expected_trials=args.trials)
            if active_tasks is not None:
                # Shared baseline jobs cover all 89 tasks; keep only the active
                # set so frontier/history/averages match this run's task set
                # (and test tasks never leak into the search state).
                task_rewards = {
                    t: r for t, r in task_rewards.items() if t in active_tasks
                }
            if task_rewards:
                per_task, avg = compute_pass_rates(task_rewards)
                update_frontier({bl_name: (per_task, avg)})
                for task, rate in per_task.items():
                    baseline_history.setdefault(task, []).append(rate)
                if not args.skip_baseline:
                    print(f"  {_bold(bl_name)}: avg={_rate_str(avg)}")
                    max_name = max(len(t) for t in per_task) if per_task else 0
                    for task, rate in sorted(per_task.items()):
                        print(f"    {task:<{max_name}}  {_rate_str(rate)}")

    # ── Phase 1..N: Evolution ──────────────────────────────────
    start_iteration = count_iterations() + 1
    for i in range(args.iterations):
        if _interrupted:
            print("Interrupted.")
            break

        iteration = start_iteration + i
        iter_start = time.time()
        frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
        best_avg = frontier.get("_best", {}).get("avg_pass_rate", 0)
        best_agent = frontier.get("_best", {}).get("agent", "none")
        print(
            f"\n{_ts()} {_bold(f'Iteration {iteration}')} ({i + 1}/{args.iterations})  frontier={best_agent} @ {best_avg:.1%}"
        )
        print(f"{'─' * 60}")

        # Propose
        if PENDING_EVAL.exists():
            PENDING_EVAL.unlink()

        propose_start = time.time()
        task_prompt = render_task_prompt(iteration, args.trials)
        print(f"  {_ts()} {_cyan('proposing')} new candidates...", flush=True)
        # Task-selector runs need the est_full_rate/delta ranking caveat
        # (avg_pass_rate is measured on a different task subset each iteration).
        # Fixed-set runs (full/hard/val) evaluate the same tasks every
        # iteration, so they get the naive skill without that instruction.
        proposer_skill = (
            "meta-harness-terminal-bench-2"
            if args.task_selector
            else "meta-harness-terminal-bench-2-naive"
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
        # Normalize class name: always AgentHarness regardless of what proposer wrote
        for c in candidates:
            if "import_path" in c and ":" in c["import_path"]:
                module, _ = c["import_path"].rsplit(":", 1)
                c["import_path"] = f"{module}:AgentHarness"
        print(
            f"  {_ts()} proposed {len(candidates)} candidate(s) in {_elapsed(propose_time)}"
        )
        for ci, c in enumerate(candidates):
            hyp = c.get("hypothesis", "")
            print(f"    {ci + 1}. {_bold(c['name'])}: {hyp[:80]}")

        # Validate
        valid = []
        print(f"  {_ts()} {_cyan('validating')} {len(candidates)} candidate(s)...")
        for ci, c in enumerate(candidates):
            name = c["name"]
            import_path = c["import_path"]
            prefix = f"    [{ci + 1}/{len(candidates)}] {name}:"
            if validate_candidate(name, import_path):
                if args.skip_smoke:
                    print(f"{prefix} {_green('import OK')} (smoke skipped)")
                    valid.append(c)
                elif smoke_test(name, import_path):
                    print(f"{prefix} {_green('import OK + smoke OK')}")
                    valid.append(c)
                else:
                    print(f"{prefix} {_red('smoke FAIL')}")
            else:
                print(f"{prefix} {_red('import FAIL')}")
            if _interrupted:
                break

        if not valid:
            print(
                f"  {_red('0 valid')} out of {len(candidates)} candidates, skipping iteration"
            )
            update_evolution_summary(
                iteration, candidates, {}, propose_time=propose_time
            )
            continue
        print(f"  {_green(f'{len(valid)} valid')} out of {len(candidates)} candidates")

        # Benchmark
        selected_tasks = None
        task_history = None
        inclusion = None
        if selector:
            task_history = build_task_history(baseline_history)
            selected_tasks = selector.select_tasks(task_history, iteration)
            if selected_tasks:
                print(
                    f"  {_cyan('task selector')} picked {len(selected_tasks)}/{n_tasks} tasks: "
                    + ", ".join(selected_tasks)
                )
                if (
                    estimator
                    and getattr(estimator, "REQUIRES_INCLUSION", False)
                    and hasattr(selector, "inclusion_probs")
                ):
                    # π must come from the same history the selection used
                    inclusion = selector.inclusion_probs(task_history, iteration)
            else:
                print(f"  {_cyan('task selector')} full-pool evaluation (exploration)")
        n_tasks_iter = len(selected_tasks) if selected_tasks else n_tasks

        bench_start = time.time()
        results = {}
        all_metrics = {}
        est_scores = {}
        n_evals = len(valid) * n_tasks_iter * args.trials
        print(
            f"  {_ts()} {_cyan('benchmarking')} {len(valid)} agent(s) x {n_tasks_iter} tasks x {args.trials} trials = {n_evals} evals"
        )
        for ci, c in enumerate(valid):
            if _interrupted:
                break
            name = c["name"]
            import_path = c["import_path"]
            job_name = f"evolve-{name}-t{args.trials}"

            print(f"    [{ci + 1}/{len(valid)}] {_bold(name)}...", flush=True)
            t0 = time.time()
            job_dir, job_result = harbor_run(
                import_path,
                job_name,
                n_trials=args.trials,
                n_concurrent=args.concurrent,
                tasks=selected_tasks,
            )
            elapsed = time.time() - t0
            if job_result:
                task_rewards = parse_job_results(job_dir, expected_trials=args.trials)
                per_task, avg = compute_pass_rates(task_rewards)
                if avg == 0:
                    check_fatal_api_errors(job_dir)
                results[name] = (per_task, avg)
                # On a subset, compare against the frontier restricted to the
                # same tasks; the full-set best_avg is not comparable.
                ref_avg = best_avg
                if selected_tasks and FRONTIER_VAL.exists():
                    fr = json.loads(FRONTIER_VAL.read_text())
                    rates = [
                        fr.get(t, {}).get("pass_rate", 0) for t in selected_tasks
                    ]
                    ref_avg = sum(rates) / len(rates)
                delta = avg - ref_avg
                delta_str = f"{delta:+.1%}"
                delta_colored = (
                    _green(delta_str)
                    if delta > 0
                    else (_red(delta_str) if delta < 0 else _dim(delta_str))
                )

                est_str = ""
                if estimator and selected_tasks and inclusion:
                    est = estimator.estimate_full_rate(
                        task_history, selected_tasks, per_task, inclusion
                    )
                    if est:
                        est_scores[name] = est
                        est_str = (
                            f"  est_full={_rate_str(est['est_full_rate'])}"
                            f"±{est['est_full_se']:.0%}"
                        )

                trial_metrics = parse_trial_metrics(job_dir)
                metrics_summary = summarize_trial_metrics(trial_metrics)
                all_metrics[name] = metrics_summary
                cost_str = (
                    f"${metrics_summary['total_cost_usd']:.2f}"
                    if metrics_summary["total_cost_usd"]
                    else "?"
                )
                print(
                    f"         avg={_rate_str(avg)}  delta={delta_colored}{est_str}  cost={cost_str}  ({_elapsed(elapsed)})"
                )
                max_name_len = max(len(t) for t in per_task) if per_task else 0
                baseline_per_task = {}
                if FRONTIER_VAL.exists():
                    fr = json.loads(FRONTIER_VAL.read_text())
                    for tk in per_task:
                        baseline_per_task[tk] = fr.get(tk, {}).get("pass_rate", 0)
                for task, rate in sorted(per_task.items()):
                    bl = baseline_per_task.get(task, 0)
                    td = rate - bl
                    td_s = f"{td:+.0%}" if td != 0 else "  ="
                    td_c = (
                        _green(td_s)
                        if td > 0
                        else (_red(td_s) if td < 0 else _dim(td_s))
                    )
                    tm = metrics_summary.get("per_task", {}).get(task, {})
                    tc = (
                        f"${tm['mean_cost']:.2f}"
                        if tm.get("mean_cost") is not None
                        else ""
                    )
                    tt = (
                        f"{tm['mean_turns']:.0f}t"
                        if tm.get("mean_turns") is not None
                        else ""
                    )
                    suffix = f"  {_dim(tc)} {_dim(tt)}" if tc else ""
                    print(
                        f"         {task:<{max_name_len}}  {_rate_str(rate)}  {td_c}{suffix}"
                    )
            else:
                results[name] = ({}, 0)
                print(
                    f"         {_red('FAIL')} benchmark crashed ({_elapsed(elapsed)})"
                )

        bench_time = time.time() - bench_start

        # Write the summary first: delta is measured against the frontier as it
        # stood before this iteration. Updating the frontier first would make the
        # delta of whichever candidate just became the frontier always 0.0.
        update_evolution_summary(
            iteration,
            valid,
            results,
            propose_time=propose_time,
            bench_time=bench_time,
            metrics=all_metrics,
            selected_tasks=selected_tasks,
            est_scores=est_scores,
        )
        update_frontier(
            results,
            partial=bool(selected_tasks),
            est_scores=est_scores,
            iteration=iteration,
        )

        wall_time = time.time() - iter_start
        frontier_now = (
            json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
        )
        new_best_avg = frontier_now.get("_best", {}).get("avg_pass_rate", 0)
        new_best_agent = frontier_now.get("_best", {}).get("agent", "none")
        improved = new_best_avg > best_avg
        status = _green("NEW BEST") if improved else _dim("no improvement")
        print(f"  {_ts()} {status}  frontier={new_best_agent} @ {new_best_avg:.1%}")
        print(
            f"  {_dim(f'timing: propose={_elapsed(propose_time)} bench={_elapsed(bench_time)} total={_elapsed(wall_time)}')}"
        )

    # ── Phase Final: zero-cost winner selection (est-max, tc-validated) ──────
    if not _interrupted:
        select_winner()

    # ── Optional: winners get 5-trial eval on the full dataset ───────────────
    if _interrupted or not args.full_eval:
        return

    print(f"\n{_ts()} {_bold('Phase Final: 5-trial eval for frontier agents')}")
    frontier = json.loads(FRONTIER_VAL.read_text()) if FRONTIER_VAL.exists() else {}
    best_agent = frontier.get("_best", {}).get("agent")
    if best_agent and best_agent != BASELINE_AGENT_NAME:
        import_path = None
        for line in EVOLUTION_SUMMARY.read_text().strip().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("agent") == best_agent and row.get("import_path"):
                import_path = row["import_path"]
                break

        if import_path:
            job_name = f"final-{best_agent}-t5"
            print(f"  {_ts()} running {_bold(best_agent)} x 5 trials...", flush=True)
            t0 = time.time()
            job_dir, _ = harbor_run(
                import_path,
                job_name,
                n_trials=5,
                n_concurrent=args.concurrent,
            )
            if job_dir.exists():
                task_rewards = parse_job_results(job_dir, expected_trials=5)
                per_task, avg = compute_pass_rates(task_rewards)
                print(
                    f"  {_bold(best_agent)} (5-trial): avg={_rate_str(avg)}  ({_elapsed(time.time() - t0)})"
                )
                max_name_len = max(len(t) for t in per_task) if per_task else 0
                for task, rate in sorted(per_task.items()):
                    print(f"    {task:<{max_name_len}}  {_rate_str(rate)}")

    print(f"\n{_ts()} {_bold('Evolution complete.')}")


def main():
    parser = argparse.ArgumentParser(
        description="Scaffold evolution loop for Terminal-Bench (KIRA track)"
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of evolution iterations"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_SEARCH_TRIALS,
        help="Trials per task during evolution (default: 2)",
    )
    parser.add_argument(
        "--propose-timeout",
        type=int,
        default=2400,
        help="Timeout for proposer (seconds)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for isolated output dirs (jobs/<run>/*, logs/<run>/*). Auto-generated if not set.",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Clear proposed agents and reset logs"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip Phase 0 baseline eval"
    )
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke tests")
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="Run the optional 5-trial winner eval on the full dataset",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent trials (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--task-selector",
        type=str,
        default=None,
        help="Extension task selector for candidate benchmarking "
        "(see extensions/__init__.py TASK_SELECTORS; currently 'task-coevolve')",
    )
    parser.add_argument(
        "--anchored-summary",
        nargs="?",
        const="hajek",
        default=None,
        metavar="ESTIMATOR",
        help="Record est_full_rate (full-set-equivalent estimate with design "
        "standard error) in evolution_summary rows, making task-subset "
        "iterations comparable. Estimators: 'hajek' (Eq. 3) or 'difference' "
        "(Eq. 4). Requires --task-selector.",
    )
    parser.add_argument(
        "--select-winner",
        action="store_true",
        help="Zero-cost winner selection for an existing run: rank by "
        "est_full_rate (fallback: avg_pass_rate), tie-break to the earliest "
        "iteration, write logs/<run>/winner.json, then exit. Requires "
        "--run-name. Runs automatically at the end of every evolve run.",
    )
    parser.add_argument(
        "--snapshot-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy the project code to logs/<run>/code at run start and execute "
        "harbor / proposer / import checks from the copy, so editing the repo "
        "while the run executes cannot change its evaluation conditions. "
        "ON by default; disable with --no-snapshot-code. "
        "Resumed runs reuse an existing snapshot automatically.",
    )
    parser.add_argument(
        "--validate-agent",
        metavar="MODULE:CLASS",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--restore-candidates",
        action="store_true",
        help="Copy a run's archived candidates from logs/<run>/agents/ back "
        "into agents/ (for scripts/run_eval.sh), then exit. Requires --run-name.",
    )
    args = parser.parse_args()

    if args.anchored_summary and not args.task_selector:
        parser.error("--anchored-summary requires --task-selector")
    if args.select_winner and not args.run_name:
        parser.error("--select-winner requires --run-name")
    # Resolve extension names up front so typos fail before any run dir is made
    try:
        if args.task_selector:
            from extensions import get_task_selector

            get_task_selector(args.task_selector)
        if args.anchored_summary:
            from extensions import get_summary_estimator

            get_summary_estimator(args.anchored_summary)
    except (KeyError, AttributeError) as exc:
        parser.error(str(exc))

    if args.restore_candidates:
        if not args.run_name:
            parser.error("--restore-candidates requires --run-name")
        archived = EVOLVE_DIR / "logs" / args.run_name / "agents"
        files = sorted(archived.glob("*.py")) if archived.exists() else []
        for f in files:
            shutil.copy2(str(f), str(AGENTS_DIR / f.name))
        print(f"Restored (copied) {len(files)} candidate file(s) from {archived}")
        return 0

    if args.validate_agent:
        try:
            agent_class = validate_agent_class(args.validate_agent)
        except Exception as exc:
            print(f"Invalid agent {args.validate_agent}: {exc}", file=sys.stderr)
            return 1
        print(f"OK: {agent_class.__module__}:{agent_class.__name__}")
        return 0

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        run_evolve(args)
    finally:
        # Park candidates in the run dir on every exit path (done,
        # interrupted, or failed); the next run_evolve restores them.
        archive_candidates()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
