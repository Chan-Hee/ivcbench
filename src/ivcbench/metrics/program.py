"""Axis 3 — Immune-program (AUCell) on cluster-specific gene sets. Higher is better.

v0 implements a rank-based AUCell (recovery-curve AUC of the gene set among each cell's top-ranked
genes). Phase-1 may swap in decoupler/pyscenic AUCell; the interface (per-cell score in [0,1]) is
identical. aucell_delta_corr correlates predicted vs observed program shift across strata.
"""
from __future__ import annotations

import numpy as np


def aucell(X: np.ndarray, gene_set_idx: np.ndarray, top_frac: float = 0.05) -> np.ndarray:
    """Per-cell AUCell-like enrichment score in [0, 1]."""
    gs = np.asarray(sorted(set(int(i) for i in gene_set_idx)), dtype=int)
    if len(gs) == 0:
        return np.zeros(X.shape[0])
    n_cells, n_genes = X.shape
    thr = max(1, int(round(top_frac * n_genes)))
    order = np.argsort(-X, axis=1)              # genes sorted desc per cell
    ranks = np.empty_like(order)
    rows = np.arange(n_cells)[:, None]
    ranks[rows, order] = np.arange(n_genes)[None, :]  # 0-based rank of each gene
    gs_ranks = ranks[:, gs]                     # (n_cells, |gs|)
    auc = np.zeros(n_cells)
    for t in range(thr):                        # recovery curve up to the top-frac threshold
        auc += (gs_ranks <= t).sum(axis=1)
    return auc / (thr * len(gs))


def aucell_delta_corr(
    pred_cells: np.ndarray,
    test_cells: np.ndarray,
    control_cells: np.ndarray,
    gene_set_idx: np.ndarray,
    test_strata: np.ndarray,
    top_frac: float = 0.05,
) -> dict:
    ctrl = float(aucell(control_cells, gene_set_idx, top_frac).mean())
    obs_d, pred_d = [], []
    for s in np.unique(test_strata):
        m = test_strata == s
        obs_d.append(aucell(test_cells[m], gene_set_idx, top_frac).mean() - ctrl)
        pred_d.append(aucell(pred_cells[m], gene_set_idx, top_frac).mean() - ctrl)
    obs_d, pred_d = np.array(obs_d), np.array(pred_d)
    if len(obs_d) < 2 or obs_d.std() < 1e-12 or pred_d.std() < 1e-12:
        return {"corr": 0.0, "obs_mean_delta": float(obs_d.mean()) if len(obs_d) else float("nan")}
    return {"corr": float(np.corrcoef(pred_d, obs_d)[0, 1]), "obs_mean_delta": float(obs_d.mean())}
