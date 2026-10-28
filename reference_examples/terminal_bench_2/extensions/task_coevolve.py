"""Phase 1 of Task-CoEvolve: variance-weighted task selection.

Each iteration draws the evaluation subset without replacement, giving more
weight to tasks whose outcomes have been inconsistent across the harnesses seen
so far. A task that every harness solves, or that none solves, separates
candidates poorly, so it is worth little; a task near a 50% pass rate separates
them best.

Weight of task i, where p is the mean of its observed pass rates and n is the
number of observations:

    w(i) = max( W_PARTIAL * 4 * p * (1 - p),   # discriminative power, max at p=0.5
                floor )                        # never solved -> W_UNSOLVED (+growth)
                                               # always solved -> W_SOLVED
           + UNCERTAINTY_WEIGHT / sqrt(n)      # re-exploration bonus

The re-exploration bonus keeps a task from freezing out of the pool once its
early observations happen to agree.

"Always solved" means every observation >= SOLVED_MIN, "never solved" means
every observation <= UNSOLVED_MAX. With trials > 1 an intermediate pass rate is
itself evidence of partial solvability and feeds the variance term directly.

The paper states the weight on a 1/4 scale, so the values there are a quarter of
the code defaults; the code carries the factor of 4 above.

Enable with:
    meta_harness.py --task-selector task-coevolve

Tuning knobs (environment variables): VWS_FRACTION, VWS_W_SOLVED,
VWS_W_UNSOLVED, VWS_W_UNSOLVED_GROWTH, VWS_W_PARTIAL, VWS_SOLVED_MIN,
VWS_UNSOLVED_MAX, VWS_UNCERTAINTY_WEIGHT, VWS_EXPLORE_ITERATIONS,
VWS_RECENT_WINDOW, VWS_SEED, VWS_PI_REPS (Monte-Carlo draws for the inclusion
probabilities).
"""

import math
import os
import random

# Environment-variable prefix for the tuning knobs. Pinned here so that
# renaming the module does not change the variable names.
ENV_PREFIX = "VWS_"

FRACTION = float(os.environ.get("VWS_FRACTION", 0.2))
W_SOLVED = float(os.environ.get("VWS_W_SOLVED", 0.0))
W_UNSOLVED = float(os.environ.get("VWS_W_UNSOLVED", 0.5))
W_UNSOLVED_GROWTH = float(os.environ.get("VWS_W_UNSOLVED_GROWTH", 0.0))
W_PARTIAL = float(os.environ.get("VWS_W_PARTIAL", 1.0))
SOLVED_MIN = float(os.environ.get("VWS_SOLVED_MIN", 1.0))
UNSOLVED_MAX = float(os.environ.get("VWS_UNSOLVED_MAX", 0.0))
UNCERTAINTY_WEIGHT = float(os.environ.get("VWS_UNCERTAINTY_WEIGHT", 0.1))
EXPLORE_ITERATIONS = int(os.environ.get("VWS_EXPLORE_ITERATIONS", 0))
RECENT_WINDOW = int(os.environ.get("VWS_RECENT_WINDOW", 0))
SEED = int(os.environ.get("VWS_SEED", 42))
PI_REPS = int(os.environ.get("VWS_PI_REPS", 4000))


def _weight(rates, iteration):
    """Continuous sampling weight from one task's pass-rate history."""
    if RECENT_WINDOW > 0:
        rates = rates[-RECENT_WINDOW:]
    n = len(rates)
    p = sum(rates) / n
    w = W_PARTIAL * 4.0 * p * (1.0 - p)
    if all(r <= UNSOLVED_MAX for r in rates):
        floor = min(W_PARTIAL, W_UNSOLVED + W_UNSOLVED_GROWTH * max(0, iteration - 1))
        w = max(w, floor)
    elif all(r >= SOLVED_MIN for r in rates):
        w = max(w, W_SOLVED)
    return w + UNCERTAINTY_WEIGHT / math.sqrt(n)


def _weighted_sample(rng, weights, n):
    """Weighted sampling without replacement; never picks zero-weight items."""
    pool = {t: w for t, w in weights.items() if w > 0}
    picked = []
    while pool and len(picked) < n:
        total = sum(pool.values())
        r = rng.random() * total
        acc = 0.0
        for t in sorted(pool):
            acc += pool[t]
            if r <= acc:
                picked.append(t)
                del pool[t]
                break
    return picked


def _design(history, iteration):
    """(n_select, weights): the sampling design of this iteration.

    Shared by select_tasks() and inclusion_probs() so that the probabilities
    describe the design actually used.
    """
    # round, not truncate: with the 89-task pool this matches the budget of the
    # fixed-subset baseline exactly (8.9 -> 9 and 26.7 -> 27, where truncation
    # would be one task short).
    n_select = max(1, round(len(history) * FRACTION))
    weights = {t: _weight(history[t], iteration) for t in history}
    if all(w <= 0 for w in weights.values()):
        weights = {t: 1.0 for t in history}  # degenerate: uniform fallback
    return n_select, weights


def select_tasks(history, iteration):
    """Pick the evaluation subset for this iteration.

    Returns a sorted task list, or None to evaluate the full task set.
    """
    if iteration <= EXPLORE_ITERATIONS:
        return None
    pool = {t: r for t, r in history.items() if r}
    if not pool:
        return None
    n_select, weights = _design(pool, iteration)
    if n_select >= len(pool):
        return None
    rng = random.Random(f"vws:{SEED}:{iteration}")
    picked = _weighted_sample(rng, weights, n_select)
    return sorted(picked) or None


def inclusion_probs(history, iteration):
    """Per-task inclusion probabilities π of this iteration's design.

    Used by the Phase 2 estimators. Sampling without replacement with unequal
    weights has no closed form here, so the same design is simulated PI_REPS
    times with independent randomness and the probabilities are estimated by
    Monte Carlo. Only zero-weight tasks get pi=0; a positive-weight task that
    never came up is floored at 0.5/PI_REPS so that 1/pi cannot diverge.

    Returns {task: pi} covering every task in the pool.
    """
    pool = {t: r for t, r in history.items() if r}
    if not pool:
        return {}
    n_select, weights = _design(pool, iteration)
    if iteration <= EXPLORE_ITERATIONS or n_select >= len(pool):
        return {t: 1.0 for t in pool}
    counts = dict.fromkeys(pool, 0)
    for r in range(PI_REPS):
        rng = random.Random(f"vws-pi:{SEED}:{iteration}:{r}")
        for t in _weighted_sample(rng, weights, n_select):
            counts[t] += 1
    return {
        t: (max(counts[t], 0.5) / PI_REPS if weights[t] > 0 else 0.0) for t in pool
    }
