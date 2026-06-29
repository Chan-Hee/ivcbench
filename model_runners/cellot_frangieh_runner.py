#!/usr/bin/env python
"""CellOT-on-Frangieh model runner — executed inside the `cellot` conda env (cellot repo + torch).

Invoked by scripts/cellot_frangieh.py:  <cellot python> cellot_frangieh_runner.py <in.npz> <out.npz>

The Frangieh scPerturb h5ad needs a NEW anndata (numpy 2.x), but the cellot package lives in the old
`cellot` env (numpy 1.19). So the driver loads + scores in the GPU-free `.venv` and shells the MODEL step
here. This runner does NOT reinvent the OT — it imports the SAME helper functions the in-process
CellOT-Kang / CellOT-Soskic rows use (scripts/cellot_runner.py: build_ae/train_ae/ae_encode/
build_fg/train_cellot_latent/transport_latent/ae_decode). It receives ONLY the leak-safe payload (train
expression + control mask + the held-KO group's control cells = the shared non-targeting control) and
returns the per-held-control-cell CellOT-predicted treated profile. It never sees held-out test
expression — the leak boundary holds on the model side too.

Payload (in.npz, cross-version-safe: unicode str + float32, no object pickle):
  X_train (n_train, G) f32, is_control_train (n_train,) bool, X_ctrl_inf (n_ctrl, G) f32,
  ae_iters int, cellot_iters int, seed int, cap int.
Output (out.npz): pred_genes (n_ctrl, G) f32 — CellOT-predicted treated profile per held control cell;
  best_mmd float (the latent OT model-selection MMD).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("IVCBENCH_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))


def main(in_path: str, out_path: str) -> None:
    import cellot_runner as R  # the SAME runner the in-process CellOT rows use

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    ae_iters = int(d["ae_iters"]); cellot_iters = int(d["cellot_iters"])
    seed = int(d["seed"]); cap = int(d["cap"]) if "cap" in d else 4000

    if is_ctrl.sum() < 5 or (~is_ctrl).sum() < 5:
        raise RuntimeError("CellOT-Frangieh: need >=5 control and >=5 treated (pooled-KO) train cells")
    if X_ctrl_inf.shape[0] == 0:
        raise RuntimeError("CellOT-Frangieh: no held-group control cells (inference input) to push")

    R.set_seed(seed)
    Xtr_ctrl = X[is_ctrl]
    Xtr_treat = X[~is_ctrl]                       # pooled training KO cells = the OT "treated" target
    Xtr_all = X
    # honor the per-arm OT cell cap (OT is O(n^2) in the MMD model-selection; the sampler batches, but
    # cap the encoded source/target clouds to keep latent OT tractable on big panels).
    rng = np.random.default_rng(seed)
    if Xtr_ctrl.shape[0] > cap:
        Xtr_ctrl = Xtr_ctrl[rng.choice(Xtr_ctrl.shape[0], cap, replace=False)]
    if Xtr_treat.shape[0] > cap:
        Xtr_treat = Xtr_treat[rng.choice(Xtr_treat.shape[0], cap, replace=False)]

    ae = R.train_ae(R.build_ae(X.shape[1]), Xtr_all, n_iters=ae_iters)
    Zsrc, Ztgt = R.ae_encode(ae, Xtr_ctrl), R.ae_encode(ae, Xtr_treat)
    f, g = R.build_fg(latent_dim=Zsrc.shape[1])
    # Frangieh has very few non-targeting CONTROL cells (control is one perturbation label, capped by the
    # loader's subsample_per_group). CellOT's compute_loss_f needs source/target batches of EQUAL size,
    # so cap the OT batch size at the smaller arm (the control source) — otherwise the source sampler
    # yields min(64, n_ctrl) and the target yields 64, which mismatches. Default 64 when both arms are big.
    bs = int(min(64, Zsrc.shape[0], Ztgt.shape[0]))
    f, g, best_mmd = R.train_cellot_latent(f, g, Zsrc, Ztgt, n_iters=cellot_iters, batch_size=bs)
    Zpush = R.transport_latent(g, R.ae_encode(ae, X_ctrl_inf))
    pred_genes = R.ae_decode(ae, Zpush).astype(np.float32)

    if not np.all(np.isfinite(pred_genes)):
        raise RuntimeError("CellOT-Frangieh: predicted profile contains non-finite values")
    np.savez(out_path, pred_genes=pred_genes, best_mmd=np.float32(best_mmd))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
