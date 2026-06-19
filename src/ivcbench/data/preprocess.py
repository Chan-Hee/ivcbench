"""Unified preprocessing — the SAME pipeline for every cluster (C1–C5).

Keeping preprocessing identical across clusters is what makes the benchmark comparable and the paper's
Methods a single statement. Every loader ends by calling `preprocess()`; nothing reimplements
normalization or HVG selection. (See CONVENTIONS.md.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .schema import CONTROL_TOKEN, CellSet


@dataclass(frozen=True)
class PreprocessConfig:
    n_hvg: int = 2000            # highly-variable genes kept (shared across clusters)
    target_sum: float = 1e4      # library-size normalization target
    log1p: bool = True
    min_genes_per_cell: int = 200
    min_cells_per_gene: int = 3
    seed: int = 0


def library_log_normalize(X: sp.csr_matrix, target_sum: float, log1p: bool) -> sp.csr_matrix:
    X = X.tocsr().astype(np.float32)
    lib = np.asarray(X.sum(1)).ravel()
    lib[lib == 0] = 1.0
    X = X.multiply(target_sum / lib[:, None]).tocsr()
    if log1p:
        X.data = np.log1p(X.data)
    return X


def select_hvg(Xlogn: sp.csr_matrix, n_hvg: int, force_idx=None) -> np.ndarray:
    """Top-variance genes, but ALWAYS retain `force_idx` (the perturbed target genes). Gene-side
    models (GEARS/scGPT/…) and the response metric require the perturbed gene to be in the panel, so
    forced genes are kept and the remaining slots filled by variance (panel size stays ~n_hvg)."""
    mean = np.asarray(Xlogn.mean(0)).ravel()
    sqmean = np.asarray(Xlogn.multiply(Xlogn).mean(0)).ravel()
    var = sqmean - mean ** 2
    k = min(n_hvg, Xlogn.shape[1])
    force = sorted(set(int(i) for i in (force_idx or [])))
    chosen = set(force)
    for i in np.argsort(-var):
        if len(chosen) >= k:
            break
        chosen.add(int(i))
    idx = np.array(sorted(chosen))
    return idx


def preprocess(counts: sp.spmatrix, var_names, obs: pd.DataFrame, *,
               side_info: dict | None = None, uns: dict | None = None,
               cfg: PreprocessConfig = PreprocessConfig()) -> CellSet:
    """Raw counts (cells x genes) + obs -> QC-filtered, log-normalized, HVG-reduced CellSet."""
    X = counts.tocsr().astype(np.float32)
    var_names = list(var_names)

    # QC: drop low-complexity cells and rarely-detected genes (applied to X, obs, var together)
    genes_per_cell = np.asarray((X > 0).sum(1)).ravel()
    cell_mask = genes_per_cell >= cfg.min_genes_per_cell
    X, obs = X[cell_mask], obs.iloc[cell_mask].reset_index(drop=True)
    cells_per_gene = np.asarray((X > 0).sum(0)).ravel()
    gene_mask = cells_per_gene >= cfg.min_cells_per_gene
    X = X[:, gene_mask]
    var_names = [g for g, keep in zip(var_names, gene_mask) if keep]

    Xn = library_log_normalize(X, cfg.target_sum, cfg.log1p)
    # force the perturbed target genes into the panel (gene-side models / response metric need them)
    pert = set(obs["perturbation"].astype(str)) - {CONTROL_TOKEN} if "perturbation" in obs else set()
    force_idx = [i for i, g in enumerate(var_names) if g in pert]
    hvg = select_hvg(Xn, cfg.n_hvg, force_idx=force_idx)
    Xd = np.asarray(Xn[:, hvg].todense(), dtype=np.float32)
    var_hvg = [var_names[i] for i in hvg]

    return CellSet(X=Xd, obs=obs.reset_index(drop=True), var_names=var_hvg,
                   side_info=side_info or {}, uns={**(uns or {}), "preprocess": cfg.__dict__})
