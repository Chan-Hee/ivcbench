#!/usr/bin/env python
"""CellOT runner for held-GROUP splits (cell-context / donor) — `cellot` conda env.

Invoked by ivcbench.baselines.heavy.CellOTC1:  <cellot python> cellot_c1_runner.py <in.npz> <out.npz>

Same model and the same helper functions as every other CellOT row (scripts/cellot_runner.py): an
scgen autoencoder is trained on the leak-safe train cells, CellOT f/g ICNN potentials are trained in
that latent space with source = train controls and target = train treated cells, and the held group's
OWN control cells are encoded, transported and decoded. The held group's treated cells never enter
training, so the leak boundary holds on the model side.

It exists because scripts/cellot_runner.py is a standalone driver, not an adapter runner, and
cellot_frangieh_runner.py returns per-cell profiles under a different key. This one speaks the
adapter protocol and returns ONE predicted profile for the held group, keyed by its label, which is
what pred_key_is_group consumes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cellot_runner as R  # noqa: E402  the SAME helpers the deposited CellOT rows use


def main(in_path, out_path):
    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    held = str(d["held_lineage"]) if "held_lineage" in d.files else "held"

    ae_iters = int(os.environ.get("IVCBENCH_CELLOT_AE_ITERS", "12000"))
    ot_iters = int(os.environ.get("IVCBENCH_CELLOT_ITERS", "8000"))
    cap = int(os.environ.get("IVCBENCH_CELLOT_MAXCELLS", "20000"))
    R.set_seed(0)
    rng = np.random.default_rng(0)

    Xtr_ctrl, Xtr_treat = X[is_ctrl], X[~is_ctrl]
    if Xtr_ctrl.shape[0] == 0 or Xtr_treat.shape[0] == 0:
        raise RuntimeError(
            "CellOT-group: the training fold has no control or no treated cells"
        )
    for name, arr in (("all", X), ("ctrl", Xtr_ctrl), ("treat", Xtr_treat)):
        if arr.shape[0] > cap:
            idx = rng.choice(arr.shape[0], cap, replace=False)
            if name == "all":
                X = X[idx]
            elif name == "ctrl":
                Xtr_ctrl = Xtr_ctrl[idx]
            else:
                Xtr_treat = Xtr_treat[idx]

    print(
        f"[CellOT-group] held={held} train={X.shape[0]} ctrl={Xtr_ctrl.shape[0]} "
        f"treat={Xtr_treat.shape[0]} held-ctrl={X_ctrl_inf.shape[0]}",
        flush=True,
    )

    ae = R.train_ae(R.build_ae(X.shape[1]), X, n_iters=ae_iters)
    Zsrc, Ztgt = R.ae_encode(ae, Xtr_ctrl), R.ae_encode(ae, Xtr_treat)
    f, g = R.build_fg(latent_dim=Zsrc.shape[1])
    bs = int(min(64, Zsrc.shape[0], Ztgt.shape[0]))
    f, g, best_mmd = R.train_cellot_latent(
        f, g, Zsrc, Ztgt, n_iters=ot_iters, batch_size=bs
    )
    pred = R.ae_decode(ae, R.transport_latent(g, R.ae_encode(ae, X_ctrl_inf))).astype(
        np.float32
    )
    if not np.all(np.isfinite(pred)):
        raise RuntimeError("CellOT-group: predicted profile contains non-finite values")

    prof = pred.mean(0).astype(np.float32)
    np.savez(
        out_path,
        pred_perts=np.asarray([f"treated::{held}"]),
        pred_means=prof[None, :],
        best_mmd=np.float32(best_mmd),
    )
    print(f"[CellOT-group] wrote 1 profile (mmd={best_mmd:.4f})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
