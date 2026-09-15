"""Axis 1 — Response-direction (Pearson-Δ), macro-averaged over strata.

Δ = (perturbed profile) − (control baseline). We correlate predicted Δ with observed Δ across
genes, per stratum, then macro-average. `exclude_genes` implements C3's downstream-only variant
(drop the perturbed target gene so on-target knockdown does not inflate the score).
"""

from __future__ import annotations

import numpy as np


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:  # constant vector (e.g. ctrl-pred Δ≡0) -> no direction recovered
        return 0.0
    return float(np.dot(a, b) / denom)


def pearson_delta(
    pred_cells: np.ndarray,
    test_cells: np.ndarray,
    control_mean: np.ndarray,
    test_strata: np.ndarray,
    exclude_genes: np.ndarray | None = None,
) -> dict:
    """Returns {'macro': float, 'per_stratum': {stratum: r}} — per-stratum, then macro-averaged."""
    keep = np.ones(test_cells.shape[1], dtype=bool)
    if exclude_genes is not None and len(exclude_genes):
        keep[np.asarray(exclude_genes, dtype=int)] = False

    # Accumulate the stratum mean in float64. In float32 the mean of ONE profile repeated n times
    # is not that profile: for n=1032 the drift reaches 5.4e-5, which is outside the 1e-5
    # control-equality guard and scores as a real response (+0.036 on the Chen FHIP1A stratum --
    # the ctrl-pred diagnostic, which is the control by construction, scored +0.036 there too).
    # float64 makes the repeated-constant mean exact, so a constant prediction yields Δ≡0 and
    # _pearson's zero-norm branch correctly returns 0.0.
    control_mean = np.asarray(control_mean, dtype=np.float64)
    per = {}
    for s in np.unique(test_strata):
        m = test_strata == s
        delta_obs = test_cells[m].mean(0, dtype=np.float64) - control_mean
        delta_pred = pred_cells[m].mean(0, dtype=np.float64) - control_mean
        per[str(s)] = _pearson(delta_pred[keep], delta_obs[keep])
    macro = float(np.mean(list(per.values()))) if per else float("nan")
    return {"macro": macro, "per_stratum": per}
