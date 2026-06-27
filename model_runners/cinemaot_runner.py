#!/usr/bin/env python
"""CINEMA-OT runner (NOT-DEFINED FLOOR) — executed in the `scperturbench_eval` env (pertpy.Cinemaot).

Invoked by ivcbench.baselines.heavy.CINEMAOT:  <env_python> cinemaot_runner.py <in.npz> <out.npz>

CINEMA-OT is an optimal-transport *matching / treatment-effect* method: it pairs observed treated
cells to control cells and reports the single-cell treatment effect (de.X). It is therefore
NOT-DEFINED for an *unseen* gene — there are no treated cells of a held gene to match, so it cannot
generate that gene's profile. Per the OnePager applicability matrix it is `not_defined`† for
C3_LO_gene and runs only as a FLOOR reference (excluded from headline ranking).

What we run here is the honest floor: pool ALL training perturbations as one "treated" group, run
CINEMA-OT's actual entropic-OT matching (ott-jax Sinkhorn) on training cells only, take the global
OT-matched treatment effect, and apply it to the held-context control. The prediction is
perturbation-agnostic (the same global OT shift for every held gene) — which is the whole point of
listing it as a not-defined floor: it shows where an OT method lands when the unseen-gene mechanism
it needs is absent. Leak-safe: held-gene cells never enter the OT; the control baseline is control
cells only.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # env JAX is CPU-only; keep ott-jax off the GPU


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import scanpy as sc
    from pertpy.tools import Cinemaot

    n_pca = int(os.environ.get("IVCBENCH_CINEMAOT_DIM", "20"))
    cap = int(os.environ.get("IVCBENCH_CINEMAOT_MAXCELLS", "4000"))  # per arm; OT is O(n^2) memory

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})

    rng = np.random.default_rng(0)
    ctrl_idx = np.where(is_ctrl)[0]
    treat_idx = np.where(~is_ctrl)[0]
    if len(ctrl_idx) < 5 or len(treat_idx) < 5:
        raise RuntimeError("CINEMA-OT floor: too few control/treated training cells for OT matching")
    if len(ctrl_idx) > cap:
        ctrl_idx = rng.choice(ctrl_idx, cap, replace=False)
    if len(treat_idx) > cap:
        treat_idx = rng.choice(treat_idx, cap, replace=False)

    # pooled "treated" (all train perturbations) vs control — a single global OT problem
    Xc, Xt = X[ctrl_idx], X[treat_idx]
    adata = ad.AnnData(np.vstack([Xc, Xt]).astype(np.float32))
    adata.var_names = genes
    adata.obs["perturbation"] = (["control"] * len(ctrl_idx) + ["treated"] * len(treat_idx))
    adata.obs["perturbation"] = adata.obs["perturbation"].astype("category")

    dim = int(min(n_pca, adata.n_obs - 1, adata.n_vars - 1))
    sc.pp.pca(adata, n_comps=max(2, dim))

    model = Cinemaot()
    de = model.causaleffect(adata, pert_key="perturbation", control="control",
                            use_rep="X_pca", dim=max(2, dim), thres=0.15, smoothness=1e-4,
                            eps=1e-3, solver="Sinkhorn")
    eff = de.X.toarray() if hasattr(de.X, "toarray") else np.asarray(de.X)  # (n_treated, n_genes)
    if eff.ndim != 2 or eff.shape[1] != len(genes):
        raise RuntimeError(f"CINEMA-OT floor: unexpected effect shape {eff.shape} (n_genes={len(genes)})")
    global_effect = np.nanmean(eff, axis=0).astype(np.float32)         # (n_genes,) global OT shift

    ctrl_mean = X_ctrl_inf.mean(0).astype(np.float32)
    pred_profile = (ctrl_mean + global_effect).astype(np.float32)      # perturbation-agnostic floor

    pred_perts = [g for g in test_perts]                               # same profile for every held gene
    if not pred_perts:
        raise RuntimeError("CINEMA-OT floor: no held genes to predict")
    pred_means = np.tile(pred_profile, (len(pred_perts), 1)).astype(np.float32)
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object), pred_means=pred_means)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
