#!/usr/bin/env python
"""scPRAM runner (Soskic CD4-activation DONOR-LODO, ②Donor×Optimal-Transport) — `ivc-scpram` env.

Invoked by scripts/scpram_soskic.py:  <ivc-scpram python> scpram_soskic_runner.py <in.npz> <out.npz>

DONOR-axis analogue of scpram_runner.py / state_soskic_runner.py / pertadapt_soskic_runner.py, for the
2nd conditioned Optimal-Transport model (scPRAM). The held UNIT is a *donor*; the perturbation
("stimulation", 0h->16h) is SEEN in every training donor. scPRAM learns the control->stim transition
across the training donors' cells (its VAE latent + OT cell-matching + per-cell attention) and predicts
the held donor's 16h profile, PER LINEAGE (CD4 Naive/Memory), from that lineage's OWN 0h cells.

To make the held donor's lineages distinct cell-types for scPRAM's per-cell-type delta loop (which
excludes the predicted type), each held-donor lineage is encoded as its own token '<lineage>__HELD' and
carries ONLY 0h (control) cells; the training donors' cells use the bare lineage token (both 0h and 16h).
scPRAM computes the ctrl->stim delta from the training-lineage cells (type != held token) and transports
the held lineage's own 0h latents. The held donor's 16h expression is NEVER read (leak-safe).

Payload keys (built by the driver, identical to pertadapt_soskic_runner.py):
  X_train, is_control_train, pert_train ('stim_<D>'/'control'), celltype_train, X_ctrl_inf, celltype_inf,
  held_label, genes.
Output: pred_perts = ['<held_label>::<lineage>', ...], pred_means = predicted 16h profile per lineage.

LEAK-SAFETY: trained on the train fold only (held donor's 16h cells absent); the held donor's 0h cells
are inference input. Refit per fold (one invocation per held donor).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

CTRL = "ctrl"
STIM = "stim"


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import torch
    from scpram import models

    seed = int(os.environ.get("IVCBENCH_SEED", "0"))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    epochs = int(os.environ.get("IVCBENCH_SCPRAM_EPOCHS", "100"))
    ratio = float(os.environ.get("IVCBENCH_SCPRAM_RATIO", "0.005"))
    pred_cap = int(os.environ.get("IVCBENCH_SCPRAM_PREDCTRL", "300"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    celltype_train = np.array([str(c) for c in d["celltype_train"]], dtype=object)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)            # held donor's OWN 0h cells
    celltype_inf = np.array([str(c) for c in d["celltype_inf"]], dtype=object)
    held_label = str(d["held_label"])
    if X.shape[0] == 0:
        raise RuntimeError("scPRAM-soskic: empty training payload")
    if is_ctrl.sum() == 0 or (~is_ctrl).sum() == 0:
        raise RuntimeError("scPRAM-soskic: need both 0h (control) and 16h (stim) training cells")
    if X_ctrl_inf.shape[0] == 0:
        raise RuntimeError("scPRAM-soskic: no held-donor 0h cells (inference input)")

    inf_lineages = sorted({str(c) for c in celltype_inf})

    # ---- build the scPRAM AnnData: training-donor cells (bare lineage token, 0h+16h) + the held
    # donor's 0h cells (each lineage as its own '<lineage>__HELD' token, control only) ------------
    held_tokens = {l: f"{l}__HELD" for l in inf_lineages}
    X_all = np.vstack([X, X_ctrl_inf]).astype(np.float32)
    cond = np.concatenate([np.where(is_ctrl, CTRL, STIM),
                           np.full(X_ctrl_inf.shape[0], CTRL)]).astype(str)
    ctype = np.concatenate([celltype_train.astype(str),
                            np.array([held_tokens[str(l)] for l in celltype_inf], dtype=object).astype(str)])

    adata = ad.AnnData(X_all.copy())
    adata.var_names = genes
    adata.obs["condition"] = cond
    adata.obs["cell_type"] = ctype

    key_dic = {
        "condition_key": "condition",
        "cell_type_key": "cell_type",
        "ctrl_key": CTRL,
        "stim_key": STIM,
        "pred_key": "predict",
    }

    # train ONLY on the training-donor cells (held tokens carry no 16h cell anywhere → leak-safe).
    train_for_fit = adata[~adata.obs["cell_type"].isin(list(held_tokens.values()))].copy()

    model = models.SCPRAM(input_dim=adata.n_vars, device=device)
    model = model.to(model.device)
    print(f"[scPRAM-soskic] {held_label}: train cells={train_for_fit.n_obs} "
          f"lineages={sorted(set(celltype_train))} held-lineages={inf_lineages} epochs={epochs}",
          flush=True)
    model.train_SCPRAM(train_for_fit, epochs=epochs)

    ctrl_full_mean = X[is_ctrl].mean(0).astype(np.float32)
    pred_perts, pred_means = [], []
    for lin in inf_lineages:
        tok = held_tokens[lin]
        n_held = int((adata.obs["cell_type"].values == tok).sum())
        if n_held == 0:
            continue
        try:
            pred = model.predict(train_adata=adata, cell_to_pred=tok, key_dic=key_dic, ratio=ratio)
            P = pred.X
            P = P.toarray() if hasattr(P, "toarray") else np.asarray(P)
            prof = P.mean(0).astype(np.float32)
            if not np.all(np.isfinite(prof)):
                raise ValueError("non-finite prediction")
        except Exception as e:                                 # degenerate lineage → control fallback
            print(f"[scPRAM-soskic] {held_label}::{lin} predict fallback ({e})", flush=True)
            prof = ctrl_full_mean.copy()
        pred_perts.append(f"{held_label}::{lin}")
        pred_means.append(prof.astype(np.float32))

    if not pred_perts:
        raise RuntimeError("scPRAM-soskic: no held-donor lineages to predict")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
