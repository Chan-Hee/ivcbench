"""Generic statistical helpers: a bootstrap CI over a supplied value vector + Benjamini-Hochberg.

The bootstrap UNIT is whatever the caller passes in. The runner (run.py) passes per-stratum macro
scores for its per-row result CIs; the final paper-level inferential CIs are computed by the bespoke
assembly scripts over the biological unit named for each task (donor, lineage, dataset, or compound),
never over the model seeds (seeds are collapsed within a biological unit before inference). See
Supplementary Note S2 for the per-task inference unit.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap a (1-alpha) CI over the supplied value vector. The caller chooses the unit of `values`
    (per-stratum macro scores in the runner; the per-task biological unit in the final paper assembly)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(v))}


def benjamini_hochberg(pvals, alpha: float = 0.05) -> dict:
    """BH FDR correction for within-cluster baseline-pair comparisons."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    crit = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= crit
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    reject = np.zeros(n, dtype=bool)
    if k > 0:
        reject[order[:k]] = True
    # adjusted p-values (step-up)
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adj_p = np.empty(n)
    adj_p[order] = np.clip(adj, 0, 1)
    return {"reject": reject, "adj_p": adj_p, "n_significant": int(reject.sum())}
