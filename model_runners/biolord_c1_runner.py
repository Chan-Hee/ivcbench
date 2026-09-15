#!/usr/bin/env python
"""Biolord runner for the CELL-CONTEXT task (C1 Kang LOCT) — `ctx-biolord` conda env.

Invoked by ivcbench.baselines.heavy.BiolordC1:
    <env python> biolord_c1_runner.py <in.npz> <out.npz>

Biolord (Piran et al., Nat Biotechnol 2024) learns a decomposed latent space in which annotated
attributes are disentangled from the residual cell state, and generates a cell under a changed
attribute value. That is exactly the cell-context transfer setting: the perturbation attribute
(control vs stimulated) is SEEN, and the held axis is the cell type, whose identity is carried by the
held cells' own residual state. We therefore register `condition` as the categorical attribute, train
on the training fold only, and generate the stimulated counterpart of the held unit's own control
cells.

Leak-safe: the held unit's stimulated cells never enter training; only its control cells are input.
Epochs / latent size via $IVCBENCH_BIOLORD_{EPOCHS,LATENT}.
"""
from __future__ import annotations

import os
import sys

import numpy as np


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import pandas as pd
    import scanpy as sc  # noqa: F401  (biolord expects the scanpy stack present)
    from biolord import Biolord

    epochs = int(os.environ.get("IVCBENCH_BIOLORD_EPOCHS", "200"))
    n_latent = int(os.environ.get("IVCBENCH_BIOLORD_LATENT", "32"))
    max_cells = int(os.environ.get("IVCBENCH_BIOLORD_MAXCELLS", "20000"))

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if not test_perts:
        raise RuntimeError("Biolord-C1: no held perturbation label in the payload")
    stim_label = test_perts[0]
    if not is_ctrl.any() or not (~is_ctrl).any():
        raise RuntimeError(
            "Biolord-C1: training fold needs both control and stimulated cells"
        )

    rng = np.random.default_rng(0)
    rows = np.arange(X.shape[0])
    if len(rows) > max_cells:
        rows = np.sort(rng.choice(rows, max_cells, replace=False))
    Xtr, ctr = X[rows], is_ctrl[rows]

    # one AnnData holding the training fold (split="train") and the held unit's own control cells
    # (split="ood"): biolord trains on `train` only, so the held cells never enter fitting, and the
    # source rows are registered through the same data manager that the model was set up with.
    n_tr, n_hd = Xtr.shape[0], X_ctrl_inf.shape[0]
    obs = pd.DataFrame(
        {
            "condition": np.concatenate(
                [np.where(ctr, "control", stim_label), np.array(["control"] * n_hd)]
            ),
            "split": np.array(["train"] * n_tr + ["ood"] * n_hd),
        },
        index=[f"tr{i}" for i in range(n_tr)] + [f"hd{i}" for i in range(n_hd)],
    )
    adata = ad.AnnData(X=np.vstack([Xtr, X_ctrl_inf]).astype(np.float32), obs=obs)
    adata.var_names = genes
    adata.obs["condition"] = adata.obs["condition"].astype("category")
    adata.obs["split"] = adata.obs["split"].astype("category")

    print(
        f"[Biolord-C1] train cells={n_tr} (ctrl={int(ctr.sum())}) held-ctrl={n_hd} "
        f"latent={n_latent} epochs={epochs}",
        flush=True,
    )

    Biolord.setup_anndata(adata, categorical_attributes_keys=["condition"])
    # NOTE: T1/T2 are X in the final 102 ruling precisely BECAUSE this path cannot be made to
    # work without disabling a core component. The earlier unknown_attributes=False override
    # was removed: turning it off is not a fix, it runs a different method.
    # biolord's residual ("unknown attribute") latent is an nn.Embedding with ONE ROW PER CELL,
    # indexed by sample (biolord/_module.py: RegularizedEmbedding, n_input=n_samples). The held
    # unit's cells are the `ood` split and never receive a gradient, so at prediction time their
    # row is still the N(0, 1) initialisation -- for n_latent=32 a vector of norm ~5.7. The held
    # identity therefore contributes NOTHING to the prediction while that noise contributes a lot,
    # and the result is seed-dependent.
    #
    # `unknown_attributes=False` makes RegularizedEmbedding multiply the residual by 0, so the
    # prediction is a function of the registered attributes alone. That is biolord's own published
    # option (the authors use it in the genetic setting; biolord_c3_runner.py already does), and it
    # is exactly the "attribute-only path" the analysis plan means by marking biolord L rather than
    # N on the context tasks: the model does not take control expression through an encoder, so a
    # held context cannot enter it. The limitation is then honest and deterministic instead of
    # hidden behind an untrained random vector.
    model = Biolord(
        adata=adata,
        n_latent=n_latent,
        model_name="ivcbench_c1",
        split_key="split",
        train_split="train",
        valid_split="train",
        test_split="ood",
    )
    model.train(
        max_epochs=epochs,
        batch_size=256,
        early_stopping=False,
        enable_progress_bar=False,
    )

    src = adata[adata.obs["split"] == "ood"].copy()
    pred = model.compute_prediction_adata(adata, src, target_attributes=["condition"])
    P = np.asarray(pred.X, dtype=np.float32)
    lab = pred.obs["condition"].astype(str).to_numpy()
    sel = lab == stim_label
    if not sel.any():
        raise RuntimeError(
            f"Biolord-C1: prediction adata has no rows for '{stim_label}' "
            f"(labels: {sorted(set(lab))})"
        )
    profile = P[sel].mean(0).astype(np.float32)

    np.savez(
        out_path,
        pred_perts=np.array([stim_label], dtype=object),
        pred_means=profile[None, :].astype(np.float32),
    )
    print(
        f"[Biolord-C1] wrote prediction for '{stim_label}' from"
        f" {int(sel.sum())} generated cells",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
