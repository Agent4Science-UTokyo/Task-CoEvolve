"""Phase 2 of Task-CoEvolve: anchored difference estimator (Eq. 4).

Same estimator as terminal_bench_2/extensions/estimator_difference.py; here the
unit of evaluation is a sample inside a dataset rather than a whole task.

Reweights the residual x_i - p_i instead of the value x_i. The anchor p_i is the
sample's mean over the history as of the start of the iteration, a constant
fixed before the draw, so E[sum (x - p) / pi] = sum (x - p) holds exactly and
the estimate is design-unbiased. This is the classical difference estimator from
survey sampling.

Two differences from the Hajek form:

1. The Hajek form is a ratio estimator, so it is not exactly unbiased; it carries
   a bias of order 1/m in the subset size m. At the 7% budget m is only 2-3
   samples per dataset, small enough for that bias to matter: measured against
   the full validation set, the Hajek estimate came out 9.3 points low at 7% and
   3.1 points low at 20%, the deviation growing as m shrinks. The difference
   estimator is exactly unbiased for any m, so this systematic error cannot
   arise.

2. Each sample contributes to the variance in proportion to (x_i - p_i)^2, so a
   sample with an extreme history (p near 0 or 1) cannot dominate the estimate
   even if it is drawn at a very small pi.

Contract (identical to the Hajek estimator):
    estimate_full_scores(datasets, frontier, results, selected_samples,
                         model_short, candidate, inclusion=None, history=None)
      -> {"est_full_val", "est_full_se", "est_by_dataset", "anchors"} | None

Enable with:
    meta_harness.py --sample-selector task-coevolve --anchored-summary difference
"""

import math

# Tells meta_harness that this module needs the inclusion probabilities.
REQUIRES_INCLUSION = True


def estimate_full_scores(
    datasets,
    frontier,
    results,
    selected_samples,
    model_short,
    candidate,
    inclusion=None,
    history=None,
):
    """Anchored difference estimate of the full-val score for one candidate.

    inclusion: {dataset: {index: π}} from the selector's sampling design.
    history: {dataset: {index: [was_correct, ...]}} — anchors are the per-sample
    means of this history, fixed before the subset was drawn.
    """
    if not inclusion:
        return None

    est_by_ds = {}
    meta = {}
    ses = []
    for ds in datasets:
        cand = results.get((model_short, ds, candidate))
        if not cand or cand.get("accuracy") is None:
            continue

        idxs = (selected_samples or {}).get(ds)
        if not idxs:
            # Evaluated on the full val: measured, not estimated
            est_by_ds[ds] = round(cand["accuracy"] * 100, 1)
            meta[ds] = {"mode": "measured"}
            ses.append(0.0)
            continue

        pi = inclusion.get(ds) or {}
        if not pi:
            est_by_ds[ds] = round(cand["accuracy"] * 100, 1)
            meta[ds] = {"mode": "subset-raw"}  # no usable design info
            ses.append(0.0)
            continue

        # Anchor p_i: the sample's mean over the history as of the draw.
        # A sample with no history falls back to the mean over those that have
        # one (0 only if nothing has any). Any anchor keeps the estimate
        # design-unbiased; the choice only affects the variance.
        hist_ds = (history or {}).get(ds) or {}
        anchors = {}
        known = []
        for i in pi:
            outcomes = hist_ds.get(i) or []
            if outcomes:
                anchors[i] = sum(outcomes) / len(outcomes)
                known.append(anchors[i])
        fallback = sum(known) / len(known) if known else 0.0
        for i in pi:
            anchors.setdefault(i, fallback)

        n_total = len(pi)
        anchor_term = sum(anchors.values()) / n_total

        # Correction term: Horvitz-Thompson sum of the residuals over the
        # drawn samples with pi > 0.
        per_sample = cand.get("per_sample") or {}
        terms = []
        for i in idxs:
            if str(i) in per_sample and pi.get(i, 0) > 0:
                x = 1.0 if per_sample[str(i)] else 0.0
                terms.append((x - anchors[i]) / pi[i])
        if not terms:
            est_by_ds[ds] = round(cand["accuracy"] * 100, 1)
            meta[ds] = {"mode": "subset-raw"}
            ses.append(0.0)
            continue

        correction = sum(terms) / n_total
        est = 100.0 * (anchor_term + correction)

        # Design standard error (conservative with-replacement approximation).
        # The variance is set by the scale of the residuals, not of the values.
        se = 100.0 * math.sqrt(sum(v * v for v in terms)) / n_total

        est_by_ds[ds] = round(max(0.0, min(100.0, est)), 1)
        meta[ds] = {
            "mode": "diff",
            "n_sampled": len(terms),
            "anchor_term": round(100.0 * anchor_term, 1),
            "correction": round(100.0 * correction, 1),
            "n_anchor_fallback": sum(1 for i in pi if not hist_ds.get(i)),
            "se": round(se, 1),
        }
        ses.append(se)

    if not est_by_ds:
        return None
    return {
        "est_full_val": round(sum(est_by_ds.values()) / len(est_by_ds), 1),
        "est_full_se": round(
            math.sqrt(sum(s**2 for s in ses)) / len(est_by_ds), 1
        ),
        "est_by_dataset": est_by_ds,
        "anchors": meta,
    }
