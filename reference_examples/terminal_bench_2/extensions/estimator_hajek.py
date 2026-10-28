"""Phase 2 of Task-CoEvolve: Hajek estimator of the full-set pass rate (Eq. 3).

A candidate is evaluated on a subset S, but the quantity of interest is its mean
pass rate over the whole task pool. Because the inclusion probabilities pi of
the sampling design are known, the subset score can be reweighted back to a
full-set score:

    mu_hat = sum_{t in S} x_t / pi_t  /  sum_{t in S} 1 / pi_t
    est    = [ n_pos * mu_hat + sum_{pi = 0} consensus_t ] / |all tasks|

where x_t is the candidate's pass rate on task t.

This is design-unbiased: it assumes nothing about the tasks or the harnesses,
only that pi is known. The selector supplies pi from its inclusion_probs().

Tasks in a pi = 0 stratum cannot be reweighted, so they are imputed with their
mean rate over the history and the count is recorded in the returned metadata.
The design standard error (a conservative with-replacement approximation) is
returned alongside the estimate, so the extra uncertainty of an adaptive subset
is visible rather than hidden.

Contract:
    estimate_full_rate(history, selected_tasks, per_task, inclusion)
      -> {"est_full_rate", "est_full_se", "meta"} | None
    history:   {task: [rate, ...]}  observations up to this iteration
    per_task:  {task: rate}         the candidate, measured on the selected tasks
    inclusion: {task: pi}

Enable with:
    meta_harness.py --task-selector task-coevolve --anchored-summary hajek
"""

import math

REQUIRES_INCLUSION = True


def estimate_full_rate(history, selected_tasks, per_task, inclusion):
    """Design-based full-set estimate for one subset-evaluated candidate."""
    if not inclusion or not selected_tasks or not per_task:
        return None

    xs, ws = [], []
    for t in selected_tasks:
        if t in per_task and inclusion.get(t, 0) > 0:
            xs.append(per_task[t])
            ws.append(1.0 / inclusion[t])
    if not xs:
        return None

    total_w = sum(ws)
    mu = sum(w * x for w, x in zip(ws, xs)) / total_w
    var = sum((w * (x - mu)) ** 2 for w, x in zip(ws, xs)) / (total_w**2)

    zero = [t for t, p in inclusion.items() if p <= 0]
    imputed = 0.0
    for t in zero:
        rates = history.get(t, [])
        if rates:
            imputed += sum(rates) / len(rates)
    n_total = len(inclusion)
    n_pos = n_total - len(zero)
    est = (n_pos * mu + imputed) / n_total
    se = (n_pos / n_total) * math.sqrt(var)

    return {
        "est_full_rate": round(max(0.0, min(1.0, est)), 4),
        "est_full_se": round(se, 4),
        "meta": {
            "mode": "ipw",
            "n_sampled": len(xs),
            "n_zero_imputed": len(zero),
        },
    }
