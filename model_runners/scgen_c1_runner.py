#!/usr/bin/env python
"""scGen runner (C1, applicable) — `scperturbench_eval` env (pertpy.tools.Scgen).

Invoked by ivcbench.baselines.heavy.ScGenC1:  <env python> scgen_c1_runner.py <in.npz> <out.npz>

C1 cytokine-response is scGen's ORIGINAL benchmark (Kang IFN-β cross-cell-type): the perturbation
(IFN-β) is SEEN, the held axis is the cell type. Classic scGen latent vector arithmetic — train the
VAE on (control, IFN-β) over the training lineages, take the global latent shift δ = mean(latent[stim])
− mean(latent[ctrl]), then for the held lineage encode ITS OWN control cells (the inference input),
add δ, and decode. Leak-safe: the held lineage's stimulated cells never enter training; the prediction
is decoded from the held lineage's control + a shift learned on other lineages.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import scvi
    from pertpy.tools import Scgen

    scvi.settings.seed = int(os.environ.get("IVCBENCH_SEED", "0"))  # training-seed plumb (multi-seed CI)
    epochs = int(os.environ.get("IVCBENCH_SCGEN_EPOCHS", "60"))

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)      # held lineage's own control cells
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if not test_perts:
        raise RuntimeError("scGen-C1: no non-control test perturbation label")
    stim_label = test_perts[0]                           # single seen cytokine, e.g. 'IFN-beta'

    cond = np.where(is_ctrl, "ctrl", "stim").astype(str)
    adata = ad.AnnData(X.copy()); adata.var_names = genes
    adata.obs["condition"] = cond; adata.obs["cell_type"] = "PBMC"
    Scgen.setup_anndata(adata, batch_key="condition", labels_key="cell_type")
    model = Scgen(adata)
    accelerator = os.environ.get("IVCBENCH_SCGEN_ACCELERATOR", "cpu")
    devices = os.environ.get("IVCBENCH_SCGEN_DEVICES")
    train_kwargs = dict(max_epochs=epochs, batch_size=128, early_stopping=True,
                        early_stopping_patience=8, accelerator=accelerator)
    if devices:
        train_kwargs["devices"] = int(devices) if devices.isdigit() else devices
    model.train(**train_kwargs)

    z = model.get_latent_representation(adata)
    delta = z[cond == "stim"].mean(0) - z[cond == "ctrl"].mean(0)   # global ctrl→stim shift

    # decode the held lineage's control + δ
    n_dec = min(512, X_ctrl_inf.shape[0])
    a_inf = ad.AnnData(X_ctrl_inf[:n_dec].copy()); a_inf.var_names = genes
    a_inf.obs["condition"] = "ctrl"; a_inf.obs["cell_type"] = "PBMC"
    z_held = model.get_latent_representation(a_inf)
    px = np.asarray(model.module.as_bound().generative(z_held + delta[None, :])["px"])
    profile = px.mean(0).astype(np.float32)

    np.savez(out_path, pred_perts=np.array([stim_label], dtype=object),
             pred_means=profile[None, :].astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
