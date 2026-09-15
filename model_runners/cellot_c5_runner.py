#!/usr/bin/env python
"""CellOT on the held-lineage COMPOUND split (T5c) — one transport per compound.

Invoked by ivcbench.baselines.heavy.CellOTC5:  <cellot python> cellot_c5_runner.py <in.npz> <out.npz>

WHY A SEPARATE RUNNER. cellot_c1_runner.py pools every treated cell into one target cloud and learns
ONE transport map, which is right where the treatment is a single seen stimulus (T1, T2). On T5c the
treatment is one of ~141 named compounds, and the pooled map was being broadcast to all of them:
every compound received the same profile, so the cell carried no compound-specific information at
all. That is not CellOT's published operation. CellOT learns a map between a control cloud and ONE
treated cloud (Bunne et al.), so the native use here is one map per compound.

WHAT IS SHARED AND WHAT IS NOT. The scgen autoencoder is trained once on the leak-safe train fold and
reused: it is unconditional, the analysis plan allows a shared backbone, and retraining it per
compound would change nothing but the clock. Only the f/g ICNN potentials -- the part that actually
encodes the control -> treated map -- are fitted per compound.

TRAINING BUDGET. The deposited pooled run used 8,000 OT iterations against ALL treated cells. A
per-compound target cloud is ~141x smaller, so the same count is not automatically the right one in
either direction. $IVCBENCH_CELLOT_ITERS_PER_CPD sets it; the per-compound MMD trace is printed at
every eval so the choice can be made from the data rather than assumed, and the value used is
recorded in the output for provenance.

COVERAGE. A compound whose training cells are too few to fit a transport is DECLINED by name and
left out of pred_perts -- never filled with the control mean, and never silently skipped.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cellot_runner as R  # noqa: E402  the SAME helpers the deposited CellOT rows use

MIN_TARGET_CELLS = 10  # below this a control -> treated map is not identifiable


def main(in_path: str, out_path: str) -> None:
    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    is_ctrl = d["is_control_train"].astype(bool)
    pert_train = np.asarray([str(p) for p in d["pert_train"]])
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    held = str(d["held_lineage"]) if "held_lineage" in d.files else "held"
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})

    ae_iters = int(os.environ.get("IVCBENCH_CELLOT_AE_ITERS", "12000"))
    ot_iters = int(os.environ.get("IVCBENCH_CELLOT_ITERS_PER_CPD", "8000"))
    cap = int(os.environ.get("IVCBENCH_CELLOT_MAXCELLS", "20000"))
    progress_path = os.environ.get("IVCBENCH_CELLOT_PROGRESS", "")
    R.set_seed(0)
    rng = np.random.default_rng(0)

    Xtr_ctrl = X[is_ctrl]
    if Xtr_ctrl.shape[0] == 0:
        raise RuntimeError("CellOT-C5: the training fold has no control cells")
    Xae = X if X.shape[0] <= cap else X[rng.choice(X.shape[0], cap, replace=False)]
    if Xtr_ctrl.shape[0] > cap:
        Xtr_ctrl = Xtr_ctrl[rng.choice(Xtr_ctrl.shape[0], cap, replace=False)]

    print(
        f"[CellOT-C5] held={held} train={X.shape[0]} ctrl={Xtr_ctrl.shape[0]} "
        f"held-ctrl={X_ctrl_inf.shape[0]} compounds={len(test_perts)} "
        f"ae_iters={ae_iters} ot_iters_per_compound={ot_iters}",
        flush=True,
    )
    if progress_path:
        with open(progress_path, "a") as fh:
            fh.write(f"{held}\tSTART compounds={len(test_perts)} ae_iters={ae_iters} "
                     f"ot_iters_per_compound={ot_iters}\n")

    # one autoencoder for the whole cell -- unconditional, so nothing compound-specific is shared
    ae = R.train_ae(R.build_ae(X.shape[1]), Xae, n_iters=ae_iters)
    Zsrc = R.ae_encode(ae, Xtr_ctrl)
    Zheld = R.ae_encode(ae, X_ctrl_inf)

    labels, profiles, declined = [], [], []
    for cpd in test_perts:
        m = (~is_ctrl) & (pert_train == cpd)
        n = int(m.sum())
        if n < MIN_TARGET_CELLS:
            declined.append(f"{cpd}(n={n})")
            continue
        Xt = X[m]
        if Xt.shape[0] > cap:
            Xt = Xt[rng.choice(Xt.shape[0], cap, replace=False)]
        Ztgt = R.ae_encode(ae, Xt)
        f, g = R.build_fg(latent_dim=Zsrc.shape[1])
        bs = int(min(64, Zsrc.shape[0], Ztgt.shape[0]))
        f, g, best_mmd = R.train_cellot_latent(
            f, g, Zsrc, Ztgt, n_iters=ot_iters, batch_size=bs
        )
        pred = R.ae_decode(ae, R.transport_latent(g, Zheld)).astype(np.float32)
        if not np.all(np.isfinite(pred)):
            declined.append(f"{cpd}(non-finite transport)")
            continue
        labels.append(cpd)
        profiles.append(pred.mean(0).astype(np.float32))
        line = f"  [{len(labels)}/{len(test_perts)}] {cpd} n={n} mmd={best_mmd:.5f}"
        print(line, flush=True)
        # SubprocessAdapter captures this runner's stdout and only surfaces it on failure, so a
        # multi-hour per-compound run is invisible while it is healthy. Mirror progress to a file
        # so it can be watched without changing the adapter contract.
        if progress_path:
            with open(progress_path, "a") as fh:
                fh.write(f"{held}\t{line.strip()}\n")

    if declined:
        print(
            f"[decline] CellOT-C5 {held}: {len(declined)} compound(s) not fitted: "
            + ", ".join(declined[:12])
            + (" ..." if len(declined) > 12 else ""),
            file=sys.stderr,
            flush=True,
        )
    if not labels:
        raise RuntimeError(
            f"CellOT-C5: no compound had at least {MIN_TARGET_CELLS} training cells in {held}"
        )
    np.savez(
        out_path,
        pred_perts=np.array(labels, dtype=object),
        pred_means=np.vstack(profiles).astype(np.float32),
        ot_iters_per_compound=np.int64(ot_iters),
    )
    print(f"[CellOT-C5] wrote {len(labels)} compound profiles for '{held}'", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
