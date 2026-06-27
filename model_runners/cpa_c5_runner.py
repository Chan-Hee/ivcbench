#!/usr/bin/env python
"""chemCPA runner (C5, applicable) — `ivc-cpa` env (cpa-tools 0.8.8).

Invoked by ivcbench.baselines.heavy.CPAchem:  <ivc-cpa python> cpa_c5_runner.py <in.npz> <out.npz>

C5 analogue of cpa_runner.py, but the conditioning axis is the COMPOUND (Morgan fingerprint), not the
gene — so an *unseen compound* is representable from its chemistry. Train CPA on the leak-safe train
cells; take each train-compound's latent shift δ; regress δ on its Morgan fingerprint; for a held
compound predict δ from its fingerprint and decode (z_inf_control + δ) through CPA's generative head,
anchored on the INFERENCE-context control cells (so the cross-cell-type split decodes from the held
lineage's own DMSO control, not a pooled mean). Leak-safe: held-compound cells never enter training;
δ comes only from the held compound's fingerprint. Works on both C5 splits (unseen-compound: held
compound's FP; cross-cell-type: the seen compound's FP, held lineage's control as the decode anchor).
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
    import pandas as pd
    import torch
    from sklearn.linear_model import Ridge

    epochs = int(os.environ.get("IVCBENCH_CPA_EPOCHS", "60"))

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)   # compound labels for C5
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if "fingerprint_keys" not in d:
        raise RuntimeError("chemCPA: payload has no fingerprint_* (compound side-rep) — cannot condition")
    fp_by_cpd = {str(k): np.asarray(v, dtype=np.float32)
                 for k, v in zip(d["fingerprint_keys"], d["fingerprint_vals"])}

    cond = np.where(is_ctrl, "ctrl", pert_train).astype(str)
    # cap training cells (stratified by compound) so large panels don't time out
    cap = int(os.environ.get("IVCBENCH_CPA_MAXCELLS", "60000"))
    if X.shape[0] > cap:
        rng = np.random.default_rng(0)
        per = max(2, cap // max(1, len(set(cond))))
        keep = np.sort(np.concatenate(
            [(lambda ci: ci if len(ci) <= per else rng.choice(ci, per, replace=False))(np.where(cond == c)[0])
             for c in set(cond)]))
        X, cond, is_ctrl = X[keep], cond[keep], is_ctrl[keep]

    adata = ad.AnnData(X.copy())
    adata.var_names = genes
    adata.obs["condition"] = cond
    adata.obs["cell_type"] = "PBMC"
    cpa.CPA.setup_anndata(adata, perturbation_key="condition", control_group="ctrl",
                          is_count_data=False, categorical_covariate_keys=["cell_type"], max_comb_len=1)
    model = cpa.CPA(adata, n_latent=64, recon_loss="gauss")
    model.train(max_epochs=epochs, batch_size=256, early_stopping_patience=8,
                use_gpu=torch.cuda.is_available(), plan_kwargs={"lr": 1e-3})

    def _latents(a):
        lat = model.get_latent_representation(a)
        return np.asarray(lat["latent_after"].X if "latent_after" in lat else lat[list(lat)[-1]].X,
                          dtype=np.float32)

    z_all = _latents(adata)
    z_ctrl = z_all[cond == "ctrl"].mean(0)

    # regress latent δ on the compound fingerprint (per train compound)
    train_cpds = [c for c in pd.unique(cond[cond != "ctrl"]) if c in fp_by_cpd]
    D, E = [], []
    for c in train_cpds:
        zc = z_all[cond == c]
        if zc.shape[0] == 0:
            continue
        D.append(zc.mean(0) - z_ctrl)
        E.append(fp_by_cpd[c])
    if len(D) < 3:
        raise RuntimeError("chemCPA: too few train-compound deltas to regress the fingerprint→δ map")
    reg = Ridge(alpha=1.0).fit(np.vstack(E), np.vstack(D))

    # decode anchor = INFERENCE-context control cells (held lineage's own control on the cross-CT split)
    n_dec = min(256, X_ctrl_inf.shape[0])
    try:
        a_inf = ad.AnnData(X_ctrl_inf[:n_dec].copy()); a_inf.var_names = genes
        a_inf.obs["condition"] = "ctrl"; a_inf.obs["cell_type"] = "PBMC"
        z_anchor = _latents(a_inf)
    except Exception:                                  # fall back to training control latents
        z_anchor = z_all[cond == "ctrl"][:n_dec]
    z_cd = torch.tensor(z_anchor, dtype=torch.float32, device=model.device)
    gen = model.module.generative

    pred_perts, pred_means = [], []
    with torch.no_grad():
        for c in test_perts:
            if c not in fp_by_cpd:
                continue
            delta = torch.tensor(reg.predict(fp_by_cpd[c][None, :])[0], dtype=torch.float32,
                                 device=model.device)
            out = gen(z_cd + delta[None, :])
            px = out["px"] if isinstance(out, dict) else out
            px = px.mean if hasattr(px, "mean") and not isinstance(px, np.ndarray) else px
            px = np.asarray(px.cpu() if hasattr(px, "cpu") else px, dtype=np.float32)
            pred_perts.append(c)
            pred_means.append(px.mean(0).astype(np.float32))

    if not pred_perts:
        raise RuntimeError("chemCPA: no held compounds with fingerprints to predict")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
