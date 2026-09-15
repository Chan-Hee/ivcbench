#!/usr/bin/env python
"""scGen runner for the C5 CELL-CONTEXT split (OP3 held lineage) — `scperturbench_eval` env.

Invoked by ivcbench.baselines.heavy.ScGenC5LOCT:
    <env python> scgen_c5_loct_runner.py <in.npz> <out.npz>

Why this is NATIVE, not adapted
-------------------------------
The held axis here is the LINEAGE, not the compound: every scored compound is present in the
training cells of the other lineages. That is exactly the setting scGen publishes — a condition key
already present in training, and a latent shift decoded on the held group's own control cells. It is
the same operation scgen_c1_runner.py runs for Kang IFN-β, applied once per compound instead of once
for a single cytokine.

The historical scgen_c5_runner.py was written for the held-COMPOUND axis: it regresses the latent
shift on a Morgan fingerprint so it can extrapolate to a molecule never seen in training. That
regression is not part of published scGen, which is why its cell was withdrawn. On the held-lineage
split no extrapolation is needed and no such estimator is used here.

Leak-safe: the held lineage's treated cells never enter training; the prediction is decoded from the
held lineage's own control cells plus a shift learned on the other lineages.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

MIN_TRAIN_CELLS = 10  # a compound needs enough training cells for a stable latent mean


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import scvi
    from pertpy.tools import Scgen

    scvi.settings.seed = int(os.environ.get("IVCBENCH_SEED", "0"))
    epochs = int(os.environ.get("IVCBENCH_SCGEN_EPOCHS", "60"))

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)  # held lineage's own control cells
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if not test_perts:
        raise RuntimeError("scGen-C5-LOCT: no non-control test perturbation label")

    # condition key: the compound for treated training cells, 'ctrl' for controls
    cond = np.where(is_ctrl, "ctrl", pert_train).astype(str)
    n_by_cond = {c: int((cond == c).sum()) for c in set(cond)}
    scored = [
        c for c in test_perts if n_by_cond.get(c, 0) >= MIN_TRAIN_CELLS
    ]  # seen in training
    if not scored:
        raise RuntimeError(
            f"scGen-C5-LOCT: none of the {len(test_perts)} held-lineage compounds has "
            f"{MIN_TRAIN_CELLS}+ training cells in the other lineages"
        )

    adata = ad.AnnData(X.copy())
    adata.var_names = genes
    adata.obs["condition"] = cond
    adata.obs["cell_type"] = "OP3"
    Scgen.setup_anndata(adata, batch_key="condition", labels_key="cell_type")
    model = Scgen(adata)
    train_kwargs = dict(
        max_epochs=epochs,
        batch_size=128,
        early_stopping=True,
        early_stopping_patience=8,
        accelerator=os.environ.get("IVCBENCH_SCGEN_ACCELERATOR", "cpu"),
    )
    devices = os.environ.get("IVCBENCH_SCGEN_DEVICES")
    if devices:
        train_kwargs["devices"] = int(devices) if devices.isdigit() else devices
    model.train(**train_kwargs)

    z = model.get_latent_representation(adata)
    z_ctrl_mean = z[cond == "ctrl"].mean(0)

    # the held lineage's own control cells, encoded once
    n_dec = min(512, X_ctrl_inf.shape[0])
    a_inf = ad.AnnData(X_ctrl_inf[:n_dec].copy())
    a_inf.var_names = genes
    a_inf.obs["condition"] = "ctrl"
    a_inf.obs["cell_type"] = "OP3"
    z_held = model.get_latent_representation(a_inf)

    pred_perts, pred_means = [], []
    for c in scored:
        delta = z[cond == c].mean(0) - z_ctrl_mean  # published latent arithmetic
        px = np.asarray(model.module.as_bound().generative(z_held + delta[None, :])["px"])
        pred_perts.append(c)
        pred_means.append(px.mean(0).astype(np.float32))

    P = np.vstack(pred_means).astype(np.float32)
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object), pred_means=P)
    print(
        f"[scGen-C5-LOCT] wrote {len(pred_perts)}/{len(test_perts)} compound profiles "
        f"x {P.shape[1]} genes (native latent arithmetic, held lineage controls)",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
