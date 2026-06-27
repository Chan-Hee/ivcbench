#!/usr/bin/env python
"""scPRAM runner (Jiang et al. 2024, Bioinformatics btae265) — `ivc-scpram` conda env.

Invoked by ivcbench.baselines.heavy.ScPRAM / scripts/scpram_kang.py:
    <ivc-scpram python> scpram_runner.py <in.npz> <out.npz>

scPRAM = VAE latent space + Optimal-Transport cell matching + per-cell attention over reference
deltas. It is the 2nd CONDITIONED Optimal-Transport-family model in the benchmark (alongside CellOT),
conditioned on (cell_type, condition). It predicts a HELD cell type's stimulated state from that cell
type's OWN control cells: the perturbation (e.g. IFN-beta) is SEEN, the held axis is the cell type
(or, on the Soskic donor task, the donor — see scpram_soskic_runner.py).

The official API (github.com/jiang-q19/scPRAM, PyPI `scpram` 0.0.3):
    from scpram import models
    m = models.SCPRAM(input_dim=n_vars, device='cuda:0')
    m.train_SCPRAM(train_adata, epochs=100)
    pred = m.predict(train_adata, cell_to_pred, key_dic, ratio=0.005)

LEAK-SAFETY (HARD RULE):
  * train_SCPRAM sees ONLY the train fold (payload X_train), which the driver has already stripped of the
    held unit's stimulated cells. The reference integration encodes this same exclusion as
    train = adata[~((cell_type == held) & (condition == stim))]; here X_train already excludes them.
  * predict needs the held unit's CONTROL cells (the inference input, legitimately available). They are
    appended to the training AnnData labeled (cell_type=<held>, condition=ctrl) — control cells only.
    scPRAM internally computes the ctrl->stim latent delta from OTHER cell types (cell_type != held) and
    transports the held unit's own control latents, so the held stimulated expression is NEVER read.
  * Invoked once per fold → the VAE + OT matching refit per fold.

Payload keys (built by the SubprocessAdapter / driver, same cross-version-safe unicode convention):
  X_train, is_control_train, pert_train, X_ctrl_inf, test_perts (labels only), genes.
Output: pred_perts = ['<stim_label>'], pred_means = predicted held-unit stimulated mean profile (1, G).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HELD = "__HELD__"          # cell-type token for the held unit (control cells only enter as this token)
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)            # held unit's OWN control cells
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if not test_perts:
        raise RuntimeError("scPRAM: no non-control test perturbation label")
    stim_label = test_perts[0]                                 # single SEEN perturbation (e.g. IFN-beta)
    if X.shape[0] == 0:
        raise RuntimeError("scPRAM: empty training payload")
    if is_ctrl.sum() == 0 or (~is_ctrl).sum() == 0:
        raise RuntimeError("scPRAM: need both control and stimulated cells in the train fold")
    if X_ctrl_inf.shape[0] == 0:
        raise RuntimeError("scPRAM: no held-unit control cells (inference input) to predict from")

    # ---- build the scPRAM AnnData: train fold + the held unit's control cells -------------------
    # Reference cell types (everything in the train fold) get a single shared token so the ctrl->stim
    # delta is learned across them; the held unit is a DISTINCT cell type carrying ONLY control cells.
    X_all = np.vstack([X, X_ctrl_inf]).astype(np.float32)
    cond = np.concatenate([np.where(is_ctrl, CTRL, STIM),
                           np.full(X_ctrl_inf.shape[0], CTRL)]).astype(str)
    # Spread the reference cells over a few pseudo cell types so scPRAM's per-cell-type delta loop has
    # >1 reference group (it iterates reference cell types != held). A single 'REF' token also works
    # (delta computed over all non-held ctrl/stim); we keep one REF token for fidelity to the paired
    # cross-cell-type setting where the held unit transfers a globally-learned shift.
    ctype = np.concatenate([np.full(X.shape[0], "REF"),
                            np.full(X_ctrl_inf.shape[0], HELD)]).astype(str)

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

    # train ONLY on the train fold (held unit's control cells excluded from training; the held unit
    # has no stim cells anywhere in the payload, so this is leak-safe by construction).
    train_for_fit = adata[adata.obs["cell_type"] == "REF"].copy()

    model = models.SCPRAM(input_dim=adata.n_vars, device=device)
    model = model.to(model.device)
    model.train_SCPRAM(train_for_fit, epochs=epochs)

    # predict the held unit's stimulated state from its OWN control cells. predict() pulls the held
    # control cells from `adata` (cell_type==HELD & condition==ctrl) and the ctrl->stim delta from the
    # reference cell types (cell_type!=HELD); the held stim expression is never available.
    pred = model.predict(train_adata=adata, cell_to_pred=HELD, key_dic=key_dic, ratio=ratio)
    P = pred.X
    P = P.toarray() if hasattr(P, "toarray") else np.asarray(P)
    profile = P.mean(0).astype(np.float32)

    if not np.all(np.isfinite(profile)):
        raise RuntimeError("scPRAM: predicted profile contains non-finite values")

    np.savez(out_path, pred_perts=np.array([stim_label], dtype=object),
             pred_means=profile[None, :].astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
