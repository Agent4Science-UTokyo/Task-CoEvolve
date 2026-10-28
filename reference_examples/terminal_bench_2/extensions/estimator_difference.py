"""Phase 2 of Task-CoEvolve: anchored difference estimator (Eq. 4).

Reweights the residual x_t - p_t instead of the value x_t:

    S_hat = (1/N) * sum_t p_t  +  (1/N) * sum_{t in S, pi > 0} (x_t - p_t) / pi_t

The anchor p_t is the mean of task t over the history, fixed before the subset
is drawn, so design-unbiasedness is preserved exactly as in the Hajek form.

The point of anchoring is variance. Each task contributes in proportion to
(x_t - p_t)^2 rather than x_t^2. On a bimodal pool, where almost every task is
either always or never solved and the pool mean sits in the empty middle, the
unanchored estimator has a failure mode: a rarely sampled task that everyone
solves carries an outcome far from the pool mean, so if it happens to be drawn,
1/pi * x_t dominates the estimate. One observed instance on Terminal-Bench 2.1
turned a raw subset score of 33.3% into an estimate of 85.9%. With the anchor
the residual of such a task is near zero, which removes the pathology
structurally rather than by clipping.

The Hajek form's imputation of pi = 0 tasks by their history mean is the special
case of this estimator where the anchor is applied to those tasks only. On a
unimodal pool the two are effectively equivalent.

Contract (identical to the Hajek estimator):
    estimate_full_rate(history, selected_tasks, per_task, inclusion)
      -> {"est_full_rate", "est_full_se", "meta"} | None

Enable with:
    meta_harness.py --task-selector task-coevolve --anchored-summary difference
"""

import math

REQUIRES_INCLUSION = True


def estimate_full_rate(history, selected_tasks, per_task, inclusion):
    """Anchored difference estimate for one subset-evaluated candidate."""
    if not inclusion or not selected_tasks or not per_task:
        return None

    # Anchor p_t: the task's mean over the history as of selection time.
    # A task with no history falls back to the mean over the tasks that have one.
    anchors = {}
    known = []
    for t in inclusion:
        rates = history.get(t, [])
        if rates:
            anchors[t] = sum(rates) / len(rates)
            known.append(anchors[t])
    fallback = sum(known) / len(known) if known else 0.0
    for t in inclusion:
        anchors.setdefault(t, fallback)

    n_total = len(inclusion)
    anchor_term = sum(anchors.values()) / n_total

    # Correction term: Horvitz-Thompson sum of the residuals over the sampled
    # tasks with pi > 0.
    terms = []
    for t in selected_tasks:
        if t in per_task and inclusion.get(t, 0) > 0:
            w = 1.0 / inclusion[t]
            terms.append(w * (per_task[t] - anchors[t]))
    if not terms:
        return None
    correction = sum(terms) / n_total
    est = anchor_term + correction

    # Design standard error (conservative with-replacement approximation).
    # The variance is set by the scale of the residuals, not of the values.
    se = math.sqrt(sum(v * v for v in terms)) / n_total

    return {
        "est_full_rate": round(max(0.0, min(1.0, est)), 4),
        "est_full_se": round(se, 4),
        "meta": {
            "mode": "diff",
            "n_sampled": len(terms),
            "anchor_term": round(anchor_term, 4),
            "correction": round(correction, 4),
            "n_anchor_fallback": sum(1 for t in inclusion if not history.get(t)),
        },
    }
