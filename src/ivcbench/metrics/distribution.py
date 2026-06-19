"""Axis 2 — Distributional (energy distance) in PCA-50 space. Lower is better."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def _energy(a: np.ndarray, b: np.ndarray) -> float:
    d_ab = cdist(a, b).mean()
    d_aa = cdist(a, a).mean()
    d_bb = cdist(b, b).mean()
    return float(2 * d_ab - d_aa - d_bb)


def e_distance(
    pred_cells: np.ndarray,
    test_cells: np.ndarray,
    test_strata: np.ndarray,
    n_pca: int = 50,
    fit_on: np.ndarray | None = None,
) -> dict:
    """Energy distance between predicted and observed cell clouds, per stratum then macro-avg.

    PCA is fit on `fit_on` (typically the training expression) to define a fixed PCA-50 space,
    avoiding test-set leakage into the projection.
    """
    basis = fit_on if fit_on is not None else np.vstack([pred_cells, test_cells])
    k = int(min(n_pca, basis.shape[0] - 1, basis.shape[1]))
    pca = PCA(n_components=max(2, k), random_state=0).fit(basis)
    P, T = pca.transform(pred_cells), pca.transform(test_cells)

    per = {}
    for s in np.unique(test_strata):
        m = test_strata == s
        if m.sum() < 2:
            continue
        per[str(s)] = _energy(P[m], T[m])
    macro = float(np.mean(list(per.values()))) if per else float("nan")
    return {"macro": macro, "per_stratum": per}
