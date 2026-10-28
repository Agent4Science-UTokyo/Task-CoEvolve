"""Task-CoEvolve for the text classification setting.

Same contract as terminal_bench_2/extensions/, except that the unit of
evaluation is a sample inside a dataset rather than a whole task.

meta_harness.py is not forked; the method lives here and is switched on by CLI
flags. Without those flags the loop evaluates every candidate on the full
validation set, which is the reference arm in the paper.

Hooks:
- Sample selection (paper Section 3.3, Phase 1):
      meta_harness.py --sample-selector task-coevolve
  The module exposes
      select_samples(history, iteration)  -> {dataset: [sample_index, ...]} | None
      inclusion_probs(history, iteration) -> {dataset: {sample_index: pi}}
  where history is {dataset: {sample_index: [was_correct, ...]}}. A dataset
  absent from the returned dict, or a None return, is evaluated in full.

- Full-set estimation (paper Section 3.3, Phase 2):
      --anchored-summary {hajek,difference}
  The module exposes
      estimate_full_scores(datasets, frontier, results, selected_samples,
                           model_short, candidate, inclusion=None, history=None)
        -> dict | None
  When REQUIRES_INCLUSION is True the selector's inclusion_probs() is passed in.

  Which estimator to use follows from the shape of the pool. Per-sample outcomes
  here spread across the range rather than polarising, so the runs reported in
  the paper use 'hajek'.
"""

import importlib

# Paper Section 3.3, Phase 1: variance-weighted sample selection
SAMPLE_SELECTORS = {
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


def selector_env_prefix(name, kind="sample"):
    """Env-var prefix for a selector's tuning knobs (e.g. 'VWS_FRACTION').

    Taken from the module's ENV_PREFIX (default 'VWS_') so that renaming the
    module does not change the variable names. Used to build the run name.
    """
    path = _resolve(name, SAMPLE_SELECTORS, "sample selector")
    return getattr(importlib.import_module(path), "ENV_PREFIX", "VWS_")


def get_sample_selector(name):
    """Return the sample selector registered under `name`.

    The module must expose
    select_samples(history, iteration) -> dict[str, list[int]] | None.
    """
    path = _resolve(name, SAMPLE_SELECTORS, "sample selector")
    module = importlib.import_module(path)
    if not hasattr(module, "select_samples"):
        raise AttributeError(f"{path} does not define select_samples()")
    return module


def get_summary_estimator(name):
    """Return the full-set estimator registered under `name`."""
    path = _resolve(name, SUMMARY_ESTIMATORS, "summary estimator")
    module = importlib.import_module(path)
    if not hasattr(module, "estimate_full_scores"):
        raise AttributeError(f"{path} does not define estimate_full_scores()")
    return module
