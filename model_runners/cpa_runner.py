#!/usr/bin/env python
"""CPA runner (adapted) — executed inside the dedicated `ivc-cpa` conda env (cpa-tools 0.8.8).

Invoked by ivcbench.baselines.heavy.CPA:  <ivc-cpa python> cpa_runner.py <in.npz> <out.npz>

Same `adapted` strategy as scGen (CPA is also undefined on an unseen gene): train CPA on the leak-safe
train cells, take the per-train-gene latent shift δ in CPA's latent space, regress δ on a LEAK-SAFE
control-only PCA gene-embedding, predict δ for each held gene, and decode (z_ctrl + δ_held) through
CPA's generative head (`CPAModule.generative(z, library)` — single-latent, like scGen). Leak-safe by
construction: held-gene cells never enter training; the gene embedding uses control cells only.

Env: ivc-cpa (conda create -n ivc-cpa python=3.10; pip install cpa-tools; pip install 'pyarrow<17').
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
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    epochs = int(os.environ.get("IVCBENCH_CPA_EPOCHS", "60"))
    n_emb = int(os.environ.get("IVCBENCH_CPA_GENE_EMB", "50"))

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    gene_pos = {g: i for i, g in enumerate(genes)}

    # leak-safe gene-side embedding: control-only PCA gene-loadings
    ctrl_X = X[is_ctrl] if is_ctrl.any() else X_ctrl_inf
    k = int(min(n_emb, ctrl_X.shape[0] - 1, ctrl_X.shape[1]))
    gene_emb = PCA(n_components=max(2, k), random_state=0).fit(ctrl_X).components_.T

    cond = np.where(is_ctrl, "ctrl", pert_train).astype(str)
    # CPA's adversarial training scales with #cells × #perturbations; cap training cells
    # (stratified by condition) so large panels like Chen (301k cells × 295 perts) don't time out.
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
    adata.obs["cell_type"] = "Tcell"

    cpa.CPA.setup_anndata(adata, perturbation_key="condition", control_group="ctrl",
                          is_count_data=False, categorical_covariate_keys=["cell_type"],
                          max_comb_len=1)
    model = cpa.CPA(adata, n_latent=64, recon_loss="gauss")
    model.train(max_epochs=epochs, batch_size=256, early_stopping_patience=8,
                use_gpu=torch.cuda.is_available(), plan_kwargs={"lr": 1e-3})

    # full latent z per cell (the latent CPA.generative consumes)
    lat = model.get_latent_representation(adata)
    z_all = np.asarray(lat["latent_after"].X if "latent_after" in lat else
                       lat[list(lat)[-1]].X, dtype=np.float32)
    z_ctrl = z_all[cond == "ctrl"].mean(0)

    train_genes = [g for g in pd.unique(cond[cond != "ctrl"]) if g in gene_pos]  # from capped cond
    D, E = [], []
    for g in train_genes:
        zg = z_all[cond == g]
        if zg.shape[0] == 0:
            continue
        D.append(zg.mean(0) - z_ctrl)
        E.append(gene_emb[gene_pos[g]])
    if len(D) < 3:
        raise RuntimeError("CPA: too few train-gene deltas to regress the gene-side map")
    reg = Ridge(alpha=1.0).fit(np.vstack(E), np.vstack(D))

    # decode (z_ctrl + δ_held) via CPA's generative head, anchored on control cells
    n_dec = min(256, int((cond == "ctrl").sum()))
    z_cd = torch.tensor(z_all[cond == "ctrl"][:n_dec], dtype=torch.float32, device=model.device)
    gen = model.module.generative

    pred_perts, pred_means = [], []
    with torch.no_grad():
        for g in test_perts:
            if g not in gene_pos:
                continue
            delta = torch.tensor(reg.predict(gene_emb[gene_pos[g]][None, :])[0],
                                 dtype=torch.float32, device=model.device)
            out = gen(z_cd + delta[None, :])
            px = out["px"] if isinstance(out, dict) else out
            px = px.mean if hasattr(px, "mean") and not isinstance(px, np.ndarray) else px
            px = np.asarray(px.cpu() if hasattr(px, "cpu") else px, dtype=np.float32)
            pred_perts.append(g)
            pred_means.append(px.mean(0).astype(np.float32))

    if not pred_perts:
        raise RuntimeError("CPA: no held genes in the panel to predict")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
