"""Phase 1 of Task-CoEvolve: variance-weighted sample selection.

Same method as terminal_bench_2/extensions/task_coevolve.py; here the unit of
evaluation is a sample inside a dataset rather than a whole task.

Each iteration draws the evaluation subset without replacement, giving more
weight to samples whose outcomes have been inconsistent across the memory
systems seen so far. A sample that every system gets right, or that none gets
right, separates candidates poorly; a sample near a 50% accuracy separates them
best.

Weight of sample i, where p is its accuracy over the observations so far and n
is the number of observations:

    w(i) = max( W_PARTIAL * 4 * p * (1 - p),   # discriminative power, max at p=0.5
                floor )                        # p = 0 -> W_UNSOLVED (+growth)
                                               # p = 1 -> W_SOLVED
           + UNCERTAINTY_WEIGHT / sqrt(n)      # re-exploration bonus

4 * p * (1 - p) is the Bernoulli variance normalised to [0, 1]: 1.0 at p = 0.5,
0 as p approaches 0 or 1. The re-exploration bonus keeps a sample from freezing
out of the pool once its early observations happen to agree; without it, a
sample that is solved by everyone early on would never be revisited even if
later candidates start failing it.

The paper states the weight on a 1/4 scale, so the values there are a quarter of
the code defaults; the code carries the factor of 4 above.

Enable with:
    meta_harness.py --iterations 20 --sample-selector task-coevolve \
        --anchored-summary hajek --snapshot-code

Requires --search-fraction 1.0: sample indices are defined on the full
validation split.

Tuning knobs (environment variables): VWS_FRACTION, VWS_W_SOLVED,
VWS_W_UNSOLVED, VWS_W_UNSOLVED_GROWTH, VWS_W_PARTIAL, VWS_UNCERTAINTY_WEIGHT,
VWS_RECENT_WINDOW, VWS_SEED
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
UNCERTAINTY_WEIGHT = float(os.environ.get("VWS_UNCERTAINTY_WEIGHT", 0.1))
RECENT_WINDOW = int(os.environ.get("VWS_RECENT_WINDOW", 0))
SEED = int(os.environ.get("VWS_SEED", 42))
PI_REPS = int(os.environ.get("VWS_PI_REPS", 4000))


def _weight(outcomes, iteration):
    """Continuous sampling weight from one sample's outcome history."""
    if RECENT_WINDOW > 0:
        outcomes = outcomes[-RECENT_WINDOW:]
    n = len(outcomes)
    p = sum(outcomes) / n
    w = W_PARTIAL * 4.0 * p * (1.0 - p)
    if p == 0.0:
        floor = min(W_PARTIAL, W_UNSOLVED + W_UNSOLVED_GROWTH * max(0, iteration - 1))
        w = max(w, floor)
    elif p == 1.0:
        w = max(w, W_SOLVED)
    return w + UNCERTAINTY_WEIGHT / math.sqrt(n)


def _weighted_sample(rng, weights, n):
    """Weighted sampling without replacement; never picks zero-weight items."""
    pool = {i: w for i, w in weights.items() if w > 0}
    picked = []
    while pool and len(picked) < n:
        total = sum(pool.values())
        r = rng.random() * total
        acc = 0.0
        for i in sorted(pool):
            acc += pool[i]
            if r <= acc:
                picked.append(i)
                del pool[i]
                break
    return picked


def _design(samples, iteration):
    """(n_select, weights): the sampling design of this iteration.

    Shared by select_samples() and inclusion_probs() so that the
    probabilities describe the design actually used.
    """
    n_select = max(1, int(len(samples) * FRACTION))
    weights = {i: _weight(samples[i], iteration) for i in samples}
    if all(w <= 0 for w in weights.values()):
        weights = {i: 1.0 for i in samples}  # degenerate: uniform fallback
    return n_select, weights


def select_samples(history, iteration):
    """Pick the per-dataset val sample subset for this iteration.

    Returns {dataset: sorted indices} or None for full-pool evaluation.
    Datasets absent from the returned dict are evaluated on all samples.
    """
    selected = {}
    for ds in sorted(history):
        samples = history[ds]
        n_select, weights = _design(samples, iteration)
        if n_select >= len(samples):
            continue  # nothing to prune for this dataset
        rng = random.Random(f"vws:{SEED}:{iteration}:{ds}")
        picked = _weighted_sample(rng, weights, n_select)
        if picked:
            selected[ds] = sorted(picked)
    return selected or None


def inclusion_probs(history, iteration):
    """Per-sample inclusion probabilities π of this iteration's design.

    Used by the Phase 2 estimators. Sequential weighted sampling without
    replacement has no closed form, so the same design is simulated PI_REPS
    times with independent randomness and the probabilities are estimated by
    Monte Carlo. Only zero-weight samples get pi = 0; a positive-weight sample
    that never came up is floored at 0.5/PI_REPS so that 1/pi cannot diverge.
    Returns {dataset: {index: π}} covering every sample in the pool.
    """
    out = {}
    for ds in sorted(history):
        samples = history[ds]
        n_select, weights = _design(samples, iteration)
        if n_select >= len(samples):
            out[ds] = {i: 1.0 for i in samples}
            continue
        counts = dict.fromkeys(samples, 0)
        for r in range(PI_REPS):
            rng = random.Random(f"vws-pi:{SEED}:{iteration}:{ds}:{r}")
            for i in _weighted_sample(rng, weights, n_select):
                counts[i] += 1
        out[ds] = {
            i: (max(counts[i], 0.5) / PI_REPS if weights[i] > 0 else 0.0)
            for i in samples
        }
    return out
