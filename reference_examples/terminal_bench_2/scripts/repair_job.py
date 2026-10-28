"""Re-run the transient failures of a job, using the search's own retry rule.

meta_harness.py applies two repairs inside harbor_run() on every evaluation
during the search:

    repair_keyring_trials(job_dir)    # INFRA_ERROR_SIGNATURES (message match)
    repair_retryable_trials(job_dir)  # TRIAL_RETRY_EXCEPTIONS (exception type)

The final-evaluation path of scripts/baseline_eval.sh calls run_eval.sh directly
and does not go through it, so a tmux startup failure or a transient 500 from
the model endpoint would be recorded as a reward of 0. Those are environment
failures, not bugs in the candidate, and applying a different rule during the
search than at the final evaluation would penalise the candidate.

What counts as retryable is imported from meta_harness.py rather than copied, so
the two paths cannot drift apart. Only the resume itself is reimplemented here:
the repair functions in meta_harness set the working directory to the run's code
snapshot, which has no .venv, so `uv run` would trigger a full sync. This script
keeps the working directory at the repository and points only the candidate
import path at the snapshot, through PYTHONPATH.

Usage:
    uv run python scripts/repair_job.py jobs/<run>/<job> [--complete] [--dry-run]

--complete first runs a plain `harbor job resume` to finish the trials that were
never run (pending or cancelled). Use it to carry an interrupted evaluation up
to the full task set.
--dry-run only reports what it would act on; nothing is deleted or resumed.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import meta_harness as mh  # noqa: E402


def pending_trials(job):
    """Trials the job's own result.json reports as unfinished (pending + cancelled).

    A job that was killed part way has fewer trial directories than its
    config.json lists. A plain `harbor job resume` runs those and completes it.
    """
    try:
        stats = json.loads((job / "result.json").read_text()).get("stats") or {}
    except (json.JSONDecodeError, OSError):
        return 0
    return int(stats.get("n_pending_trials", 0)) + int(stats.get("n_cancelled_trials", 0))


def apply_registry_fix():
    """Apply the registry workaround (the same idempotent patch as run_eval.sh).

    Calling harbor directly bypasses run_eval.sh, and without the patch the
    dataset metadata lookup fails outright.
    """
    fix = REPO / "scripts" / "apply_harbor_registry_fix.py"
    if fix.exists():
        subprocess.run(["uv", "run", "python", str(fix)], cwd=str(REPO), check=False)


def infra_trials(job):
    """Trials that failed on a transient environment fault.

    These share the exception type (RuntimeError) with genuine candidate bugs, so
    they are selected by message rather than by type.
    """
    return mh._keyring_failed_trials(job)


def retryable_trials(job):
    """Trials that failed on an exception type that a re-roll usually clears."""
    out = []
    for rf in job.glob("*/result.json"):
        try:
            r = json.loads(rf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        exc = (r.get("exception_info") or {}).get("exception_type")
        if exc in mh.TRIAL_RETRY_EXCEPTIONS:
            out.append((rf.parent, exc))
    return out


def resume_env(snapshot):
    env = os.environ.copy()
    model = env.get("HARBOR_MODEL", mh.MODEL)
    env["HARBOR_MODEL"] = model
    base = str(snapshot) if snapshot else str(REPO)
    env["PYTHONPATH"] = base + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if model.startswith("openai/"):
        # Calling harbor directly skips run_eval.sh, and without these the client
        # would fall back to the public endpoint and fail to find the model.
        env.setdefault(
            "OPENAI_API_BASE",
            os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
        )
        env.setdefault("OPENAI_API_KEY", "dummy")
    return env


def resume(job, env, filters=()):
    cmd = ["uv", "run", "harbor", "job", "resume", "-p", str(job)]
    for f in filters:
        cmd += ["-f", f]
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, start_new_session=True)
    try:
        proc.wait(timeout=mh.HARBOR_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        print(f"  resume timed out job={job.name}", flush=True)
        mh._kill_process_group(proc)
        return False
    if proc.returncode != 0:
        print(f"  resume exit={proc.returncode} job={job.name}", flush=True)
    return proc.returncode == 0


def verify(job, task_names):
    """resume recreates a trial under a new random id, so match on the task name."""
    gone = []
    for task in task_names:
        hits = [d for d in job.glob(f"{task}__*") if (d / "result.json").exists()]
        if not hits:
            gone.append(task)
    return gone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--snapshot", default=None,
                    help="where to import the candidate from (default: inferred from the run)")
    ap.add_argument("--complete", action="store_true",
                    help="first run a plain harbor job resume to finish unfinished trials")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    job = Path(a.job).resolve()
    if not job.is_dir():
        sys.exit(f"ERROR: job dir not found: {job}")

    snapshot = Path(a.snapshot).resolve() if a.snapshot else None
    if snapshot is None:
        guess = REPO / "logs" / job.parent.name / "code" / "terminal_bench_2"
        snapshot = guess if guess.is_dir() else None
    print(f"job:      {job.relative_to(REPO) if REPO in job.parents else job}")
    print(f"import:   {snapshot or REPO}" + ("" if snapshot else "  (no snapshot: using agents/)"))

    n_pending = pending_trials(job)
    if n_pending:
        print(f"unfinished trials (pending+cancelled): {n_pending}"
              + ("" if a.complete else "  <- run with --complete to finish them"))
    infra = infra_trials(job)
    retry = retryable_trials(job)
    print(f"environment faults ({'/'.join(s[0] for s in mh.INFRA_ERROR_SIGNATURES)}): "
          f"{len(infra)} trials")
    print(f"transient exceptions ({'/'.join(mh.TRIAL_RETRY_EXCEPTIONS)}): {len(retry)} trials")
    for d, msg in infra:
        print(f"    [infra] {d.name}: {msg.splitlines()[0][:90]}")
    for d, exc in retry:
        print(f"    [retry] {d.name}: {exc}")
    if not infra and not retry and not (a.complete and n_pending):
        print("nothing to repair.")
        return
    if a.dry_run:
        print("--dry-run: nothing was done.")
        return

    env = resume_env(snapshot)
    apply_registry_fix()

    # 0) Finish the trials that were never run. A plain resume deletes nothing.
    if a.complete and n_pending:
        print(f"\n[complete] running {n_pending} unfinished trials via harbor job resume")
        resume(job, env)

    # 1) Environment faults: the exception type is RuntimeError, the same as a
    #    candidate bug, so harbor's type filter cannot select them. Delete the
    #    matching trial directories and resume, as meta_harness does.
    for i in range(mh.KEYRING_RETRY_PASSES):
        infra = infra_trials(job)
        if not infra:
            break
        names = [d.name for d, _ in infra]
        print(f"\n[environment pass {i + 1}/{mh.KEYRING_RETRY_PASSES}] "
              f"deleting {len(infra)} trials and resuming")
        if any(all(s in msg for s in mh.KEYRING_ERROR_SIGNATURE) for _, msg in infra):
            mh._wait_for_keyring_headroom(f"repair {job.name}")
        for d, _ in infra:
            shutil.rmtree(d, ignore_errors=True)
        resume(job, env)
        gone = verify(job, [n.rsplit("__", 1)[0] for n in names])
        if gone:
            print(f"  VANISHED: {len(gone)} tasks did not come back "
                  f"({', '.join(gone[:5])})")

    # 2) Transient exceptions: harbor's type filter can select these directly.
    for i in range(mh.TRIAL_RETRY_PASSES):
        n = mh._count_retryable_trials(job)
        if not n:
            break
        print(f"\n[retry pass {i + 1}/{mh.TRIAL_RETRY_PASSES}] resuming {n} trials with "
              f"resume -f {'/'.join(mh.TRIAL_RETRY_EXCEPTIONS)}")
        resume(job, env, mh.TRIAL_RETRY_EXCEPTIONS)

    n_infra, n_retry = len(infra_trials(job)), mh._count_retryable_trials(job)
    print(f"\n=== after repair === {n_infra} environment / {n_retry} transient "
          f"remaining (these count as a reward of 0)")


if __name__ == "__main__":
    main()
