"""Phase 2 of Task-CoEvolve: Hajek estimator of the full-set score (Eq. 3).

A candidate is evaluated on a subset of samples, but the quantity of interest is
its accuracy over the whole validation split. Because the inclusion
probabilities pi of the sampling design are known, the subset score can be
reweighted back to a full-set score:

    mu_hat = sum_{i in S} x_i / pi_i  /  sum_{i in S} 1 / pi_i

Discounting each sample by 1/pi makes the estimate design-unbiased no matter how
skewed the selection is. Selection depends only on past observations and pi
comes from our own design, so the guarantee needs no model of the data.

This replaced an earlier difference-in-differences estimator that anchored on
the frontier system. That one assumed subset difficulty acts as a shock shared
by the anchor and the candidate, but variance-weighted selection concentrates
exactly the samples where the candidate succeeds and the anchor fails, which
breaks the assumption. In practice it overestimated by 10-20 points and inverted
the ranking.

Samples in a pi = 0 stratum are excluded from selection entirely and cannot be
reweighted, so they are imputed from the consensus of the history:
  - solved by every system so far (p = 1)   -> assume the candidate solves it, 1
  - solved by no system so far (p = 0)      -> 0
  - otherwise                               -> p
With the default settings pi stays positive, so this case does not normally
arise. The number of imputed samples is recorded in the returned "anchors"
metadata to keep the assumption visible.

The design standard error (a conservative with-replacement approximation) is
returned as est_full_se. At small subset sizes that error is genuinely large,
and reporting it is the point.

Enable with:
    meta_harness.py --sample-selector task-coevolve --anchored-summary hajek
"""

import math

# Tells meta_harness that this module needs the inclusion probabilities.
# If the selector does not provide inclusion_probs(), est_full is skipped.
REQUIRES_INCLUSION = True


def _hajek(xs, weights):
    """Hájek mean and its with-replacement-approximation standard error."""
    total_w = sum(weights)
    mu = sum(w * x for w, x in zip(weights, xs)) / total_w
    var = sum((w * (x - mu)) ** 2 for w, x in zip(weights, xs)) / (total_w**2)
    return mu, math.sqrt(var)


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
    """Design-based full-val estimate for one subset-evaluated candidate.

    inclusion: {dataset: {index: π}} from the selector's sampling design.
    history: {dataset: {index: [was_correct, ...]}} for the π=0 stratum.
    Returns the same shape as idea3 (est_full_val / est_by_dataset / anchors)
    plus est_full_se, or None when π is unavailable.
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
        per_sample = cand.get("per_sample") or {}
        xs, ws = [], []
        for i in idxs:
            if str(i) in per_sample and pi.get(i, 0) > 0:
                xs.append(1.0 if per_sample[str(i)] else 0.0)
                ws.append(1.0 / pi[i])
        if not xs:
            est_by_ds[ds] = round(cand["accuracy"] * 100, 1)
            meta[ds] = {"mode": "subset-raw"}  # no usable design info
            ses.append(0.0)
            continue

        mu_pos, se_pos = _hajek(xs, ws)

        # π=0 stratum: impute from historical consensus
        zero = [i for i, p in pi.items() if p <= 0]
        imputed = 0.0
        for i in zero:
            outcomes = (history or {}).get(ds, {}).get(i, [])
            if outcomes:
                imputed += sum(outcomes) / len(outcomes)
        n_pos = len(pi) - len(zero)
        n_total = len(pi)
        est = 100.0 * (n_pos * mu_pos + imputed) / n_total
        se = 100.0 * (n_pos / n_total) * se_pos

        est_by_ds[ds] = round(max(0.0, min(100.0, est)), 1)
        meta[ds] = {
            "mode": "ipw",
            "n_sampled": len(xs),
            "n_zero_imputed": len(zero),
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
