"""Task-CoEvolve, implemented as optional extensions to the search loop.

meta_harness.py is not forked; the method lives here and is switched on by CLI
flags. Without those flags the loop evaluates every candidate on the full task
set, which is the reference arm in the paper.

Hooks:
- Task selection (paper Section 3.3, Phase 1):
      meta_harness.py --task-selector task-coevolve
  The module exposes
      select_tasks(history, iteration)    -> list[str] | None
      inclusion_probs(history, iteration) -> {task: inclusion probability pi}
  where history is {task_name: [pass_rate, ...]}, one entry per harness already
  evaluated. Returning None makes that iteration evaluate the full task set.

- Full-set estimation (paper Section 3.3, Phase 2):
      --anchored-summary {hajek,difference}
  The module exposes
      estimate_full_rate(history, selected_tasks, per_task, inclusion)
        -> dict | None
  When REQUIRES_INCLUSION is True the selector's inclusion_probs() is passed in.

  Which estimator to use follows from the shape of the task pool. On a bimodal
  pool, where almost every task is either always or never solved, use
  'difference'; otherwise use 'hajek'. Terminal-Bench 2.1 is bimodal, so the
  runs reported in the paper use 'difference' here.
"""

import importlib

# Paper Section 3.3, Phase 1: variance-weighted task selection
TASK_SELECTORS = {
    "task-coevolve": "extensions.task_coevolve",
}

# Paper Section 3.3, Phase 2: sampling-aware full-set estimation
SUMMARY_ESTIMATORS = {
    "hajek": "extensions.estimator_hajek",  # Eq. 3
    "difference": "extensions.estimator_difference",  # Eq. 4
}


def _resolve(name, registry, kind):
    if name in registry:
        return registry[name]
    raise KeyError(
        f"unknown {kind} '{name}' (available: {', '.join(sorted(registry))})"
    )


def get_task_selector(name):
    """Return the task selector registered under `name`.

    The module must expose select_tasks(history, iteration) -> list[str] | None.
    """
    path = _resolve(name, TASK_SELECTORS, "task selector")
    module = importlib.import_module(path)
    if not hasattr(module, "select_tasks"):
        raise AttributeError(f"{path} does not define select_tasks()")
    return module


def get_summary_estimator(name):
    """Return the full-set estimator registered under `name`."""
    path = _resolve(name, SUMMARY_ESTIMATORS, "summary estimator")
    module = importlib.import_module(path)
    if not hasattr(module, "estimate_full_rate"):
        raise AttributeError(f"{path} does not define estimate_full_rate()")
    return module
