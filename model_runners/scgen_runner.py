#!/usr/bin/env python
"""scGen runner (adapted) — executed inside the `scperturbench_eval` conda env (pertpy.tools.Scgen).

Invoked by ivcbench.baselines.heavy.ScGen:  <env_python> scgen_runner.py <in.npz> <out.npz>

scGen models a perturbation as a vector in a VAE latent space (predict = decode(z_ctrl + δ)). It is
undefined on an *unseen* gene (no δ). The `adapted` extension (OnePager: "adapted gene-side repr"):
learn δ in latent space for the TRAIN genes, regress δ on a LEAK-SAFE gene-side embedding (control-
only PCA gene loadings — no perturbation info), predict δ for each held gene, and decode
(δ_held + control latents) via scGen's own generative head. Leak-safe by construction: held-gene
cells never enter training, and the gene embedding uses control cells only.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import pandas as pd
    import scvi
    from pertpy.tools import Scgen
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    scvi.settings.seed = int(os.environ.get("IVCBENCH_SEED", "0"))  # training-seed plumb (multi-seed CI)
    epochs = int(os.environ.get("IVCBENCH_SCGEN_EPOCHS", "60"))
    n_emb = int(os.environ.get("IVCBENCH_SCGEN_GENE_EMB", "50"))

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    gene_pos = {g: i for i, g in enumerate(genes)}

    # ---- leak-safe gene-side embedding: PCA gene-loadings from CONTROL cells only ----
    ctrl_X = X[is_ctrl] if is_ctrl.any() else X_ctrl_inf
    k = int(min(n_emb, ctrl_X.shape[0] - 1, ctrl_X.shape[1]))
    gpca = PCA(n_components=max(2, k), random_state=0).fit(ctrl_X)
    gene_emb = gpca.components_.T                       # (n_genes, k): gene g's loading vector

    # ---- train scGen VAE on the leak-safe training cells (condition = gene / 'ctrl') ----
    cond = np.where(is_ctrl, "ctrl", pert_train).astype(str)
    adata = ad.AnnData(X.copy())
    adata.var_names = genes
    adata.obs["condition"] = cond
    adata.obs["cell_type"] = "Tcell"
    Scgen.setup_anndata(adata, batch_key="condition", labels_key="cell_type")
    model = Scgen(adata)
    # scvi's JaxSCGEN defaults to a CUDA backend; JAX here is CPU-only → force CPU (VAE is small).
    model.train(max_epochs=epochs, batch_size=128, early_stopping=True, early_stopping_patience=8,
                accelerator="cpu")

    # ---- per-train-gene latent δ ----
    z = model.get_latent_representation(adata)          # (n_cells, latent_dim)
    z_ctrl = z[cond == "ctrl"].mean(0)
    train_genes = [g for g in pd.unique(pert_train[~is_ctrl]) if g in gene_pos]
    deltas, emb = [], []
    for g in train_genes:
        zg = z[cond == g]
        if zg.shape[0] == 0:
            continue
        deltas.append(zg.mean(0) - z_ctrl)
        emb.append(gene_emb[gene_pos[g]])
    if len(deltas) < 3:
        raise RuntimeError("scGen: too few train-gene deltas to regress the gene-side map")
    D = np.vstack(deltas)                               # (n_train_genes, latent_dim)
    E = np.vstack(emb)                                  # (n_train_genes, k)

    # ---- regress latent δ on the gene embedding; predict δ for held genes ----
    reg = Ridge(alpha=1.0).fit(E, D)

    # control latents to decode from (use control cells)
    ctrl_adata = adata[cond == "ctrl"].copy()
    n_dec = min(256, ctrl_adata.n_obs)
    ctrl_adata = ctrl_adata[:n_dec].copy()
    latent_cd = model.get_latent_representation(ctrl_adata)   # (n_dec, latent_dim)
    decode = model.module.as_bound().generative

    pred_perts, pred_means = [], []
    for g in test_perts:
        if g not in gene_pos:
            continue
        delta = reg.predict(gene_emb[gene_pos[g]][None, :])[0]   # (latent_dim,)
        stim_pred = latent_cd + delta[None, :]
        px = np.asarray(decode(stim_pred)["px"])                 # (n_dec, n_genes)
        pred_perts.append(g)
        pred_means.append(px.mean(0).astype(np.float32))

    if not pred_perts:
        raise RuntimeError("scGen: no held genes in the panel to predict")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
