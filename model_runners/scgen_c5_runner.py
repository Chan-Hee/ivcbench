#!/usr/bin/env python
"""scGen runner (C5, adapted*) — `scperturbench_eval` env (pertpy.tools.Scgen).

Invoked by ivcbench.baselines.heavy.ScGenC5:  <env python> scgen_c5_runner.py <in.npz> <out.npz>

C5 analogue of scgen_runner.py: the conditioning axis is the COMPOUND Morgan fingerprint, not a gene.
Train scGen's VAE on leak-safe train cells (condition = compound / 'ctrl'); regress each train
compound's latent δ on its fingerprint; predict δ for a held compound from its fingerprint and decode
(z_inf_control + δ) anchored on the inference-context control cells. Leak-safe: held-compound cells
never train; δ is purely a function of the held compound's chemistry.
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
    from pertpy.tools import Scgen
    from sklearn.linear_model import Ridge

    epochs = int(os.environ.get("IVCBENCH_SCGEN_EPOCHS", "60"))

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)   # compound labels
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if "fingerprint_keys" not in d:
        raise RuntimeError("scGen-C5: payload has no fingerprint_* (compound side-rep)")
    fp_by_cpd = {str(k): np.asarray(v, dtype=np.float32)
                 for k, v in zip(d["fingerprint_keys"], d["fingerprint_vals"])}

    cond = np.where(is_ctrl, "ctrl", pert_train).astype(str)
    adata = ad.AnnData(X.copy()); adata.var_names = genes
    adata.obs["condition"] = cond; adata.obs["cell_type"] = "PBMC"
    Scgen.setup_anndata(adata, batch_key="condition", labels_key="cell_type")
    model = Scgen(adata)
    model.train(max_epochs=epochs, batch_size=128, early_stopping=True, early_stopping_patience=8,
                accelerator="cpu")                          # env JAX is CPU-only

    z = model.get_latent_representation(adata)
    z_ctrl = z[cond == "ctrl"].mean(0)
    train_cpds = [c for c in pd.unique(pert_train[~is_ctrl]) if c in fp_by_cpd]
    D, E = [], []
    for c in train_cpds:
        zc = z[cond == c]
        if zc.shape[0]:
            D.append(zc.mean(0) - z_ctrl); E.append(fp_by_cpd[c])
    if len(D) < 3:
        raise RuntimeError("scGen-C5: too few train-compound deltas to regress fingerprint→δ")
    reg = Ridge(alpha=1.0).fit(np.vstack(E), np.vstack(D))

    # decode anchor = inference-context control cells (held lineage's own control on cross-CT)
    n_dec = min(256, X_ctrl_inf.shape[0])
    a_inf = ad.AnnData(X_ctrl_inf[:n_dec].copy()); a_inf.var_names = genes
    a_inf.obs["condition"] = "ctrl"; a_inf.obs["cell_type"] = "PBMC"
    try:
        latent_cd = model.get_latent_representation(a_inf)
    except Exception:
        latent_cd = model.get_latent_representation(adata[cond == "ctrl"][:n_dec].copy())
    decode = model.module.as_bound().generative

    pred_perts, pred_means = [], []
    for c in test_perts:
        if c not in fp_by_cpd:
            continue
        delta = reg.predict(fp_by_cpd[c][None, :])[0]
        px = np.asarray(decode(latent_cd + delta[None, :])["px"])
        pred_perts.append(c); pred_means.append(px.mean(0).astype(np.float32))

    if not pred_perts:
        raise RuntimeError("scGen-C5: no held compounds with fingerprints to predict")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
