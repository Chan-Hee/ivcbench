#!/usr/bin/env python
"""CPA runner (C1, applicable) — `ivc-cpa` env (cpa-tools).

Invoked by ivcbench.baselines.heavy.CPAC1:  <ivc-cpa python> cpa_c1_runner.py <in.npz> <out.npz>

C1 cytokine-response (Kang IFN-β cross-cell-type): seen cytokine, held cell type. Classic latent
δ-arithmetic in CPA's latent space — train CPA on (control, IFN-β) over training lineages, take the
global shift δ = mean(latent[stim]) − mean(latent[ctrl]), decode the held lineage's own control + δ.
Leak-safe: the held lineage's stimulated cells never train.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import cpa
    import torch

    epochs = int(os.environ.get("IVCBENCH_CPA_EPOCHS", "60"))

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if not test_perts:
        raise RuntimeError("CPA-C1: no non-control test perturbation label")
    stim_label = test_perts[0]

    cap = int(os.environ.get("IVCBENCH_CPA_MAXCELLS", "60000"))
    cond = np.where(is_ctrl, "ctrl", "stim").astype(str)
    if X.shape[0] > cap:
        rng = np.random.default_rng(0)
        per = max(2, cap // max(1, len(set(cond))))
        keep = np.sort(np.concatenate(
            [(lambda ci: ci if len(ci) <= per else rng.choice(ci, per, replace=False))(np.where(cond == c)[0])
             for c in set(cond)]))
        X, cond = X[keep], cond[keep]

    adata = ad.AnnData(X.copy()); adata.var_names = genes
    adata.obs["condition"] = cond; adata.obs["cell_type"] = "PBMC"
    cpa.CPA.setup_anndata(adata, perturbation_key="condition", control_group="ctrl",
                          is_count_data=False, categorical_covariate_keys=["cell_type"], max_comb_len=1)
    model = cpa.CPA(adata, n_latent=64, recon_loss="gauss")
    model.train(max_epochs=epochs, batch_size=256, early_stopping_patience=8,
                use_gpu=torch.cuda.is_available(), plan_kwargs={"lr": 1e-3})

    def _lat(a):
        lat = model.get_latent_representation(a)
        return np.asarray(lat["latent_after"].X if "latent_after" in lat else lat[list(lat)[-1]].X,
                          dtype=np.float32)

    z = _lat(adata)
    delta = z[cond == "stim"].mean(0) - z[cond == "ctrl"].mean(0)

    n_dec = min(512, X_ctrl_inf.shape[0])
    a_inf = ad.AnnData(X_ctrl_inf[:n_dec].copy()); a_inf.var_names = genes
    a_inf.obs["condition"] = "ctrl"; a_inf.obs["cell_type"] = "PBMC"
    # CPA's get_latent_representation needs the model's registered fields (e.g. the 'perts' obsm that
    # setup_anndata builds); a fresh inference AnnData lacks them → register it against the trained model.
    cpa.CPA.setup_anndata(a_inf, perturbation_key="condition", control_group="ctrl",
                          is_count_data=False, categorical_covariate_keys=["cell_type"], max_comb_len=1)
    z_held = _lat(a_inf)
    with torch.no_grad():
        out = model.module.generative(torch.tensor(z_held + delta[None, :], dtype=torch.float32,
                                                    device=model.device))
        px = out["px"] if isinstance(out, dict) else out
        px = px.mean if hasattr(px, "mean") and not isinstance(px, np.ndarray) else px
        profile = np.asarray(px.cpu() if hasattr(px, "cpu") else px, dtype=np.float32).mean(0)

    np.savez(out_path, pred_perts=np.array([stim_label], dtype=object),
             pred_means=profile[None, :].astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
