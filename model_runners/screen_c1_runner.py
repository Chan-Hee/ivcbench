#!/usr/bin/env python
"""SCREEN (Wang et al. 2024) runner for CELL-CONTEXT / held-group splits — `cellot` conda env.

Invoked by ivcbench.baselines.heavy.ScreenC1:
    <env python> screen_c1_runner.py <in.npz> <out.npz>

SCREEN predicts the perturbed counterpart of a held-out group from that group's own control cells
via optimal transport in a VAE latent space, which is exactly the interface this task needs, so it
runs NATIVE: the published script is called unmodified with its published defaults (latent_dim 100,
batch 64, 40 epochs, leaky_relu, Adam), and only the group label is supplied.

Added at the reviewers' request for a more comprehensive recent-method panel (Reviewer 1, comment 3)
and because it tests directly whether the donor-axis optimal-transport result is specific to CellOT
or general to the OT family.

Source: https://github.com/Califorya/SCREEN ($IVCBENCH_SCREEN_DIR).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HELD = "held"
TRAIN = "train"


def _screen_dir() -> Path:
    d = os.environ.get("IVCBENCH_SCREEN_DIR", "/data1/home/chlee/software/SCREEN")
    p = Path(d) / "screen" / "screen.py"
    if not p.exists():
        raise FileNotFoundError(f"SCREEN not found at {p} (set $IVCBENCH_SCREEN_DIR)")
    return p


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import pandas as pd
    import scanpy as sc  # noqa: F401  (SCREEN imports scanpy; fail early if missing)

    z = np.load(in_path, allow_pickle=True)
    X_train = np.asarray(z["X_train"], dtype=np.float32)
    is_ctrl = np.asarray(z["is_control_train"], dtype=bool)
    X_held_ctrl = np.asarray(z["X_ctrl_inf"], dtype=np.float32)
    genes = [str(g) for g in np.asarray(z["genes"])]
    test_perts = np.asarray(z["test_perts"]).astype(str)
    stim_label = next(
        (p for p in test_perts if p.lower() not in ("control", "ctrl")), "stimulated"
    )

    # One AnnData: the training cells keep their own group labels and the held unit's control cells
    # form the group to predict. SCREEN is built for a multi-group panel and balances by group, so
    # collapsing every training cell into a single group would handicap it; when the driver supplies
    # `group_train` we pass the real structure through. SCREEN itself drops
    # (group == label & condition == stim) from training, so nothing leaks.
    grp = (
        z["group_train"].astype(str)
        if "group_train" in z.files
        else np.full(len(X_train), TRAIN)
    )
    grp = np.asarray([f"g_{g}" for g in grp])  # never collides with the held label

    # Donor-scale training folds (T2 trains on 105 donors) exceed what the published defaults
    # converge on: the VAE diverged to NaN on some folds and others ran past two hours. Cap the
    # training cells -- the same disclosed practice already used for scGPT (8,000 cells) and
    # scFoundation (4,000); the cap is reported in Supplementary Table S15. Group structure is
    # preserved by sampling within the cap, so SCREEN still sees a multi-group panel.
    _cap = int(os.environ.get("IVCBENCH_SCREEN_MAXCELLS", "0"))
    if _cap and X_train.shape[0] > _cap:
        _rng = np.random.default_rng(0)
        _sel = np.sort(_rng.choice(X_train.shape[0], _cap, replace=False))
        X_train, is_ctrl, grp = X_train[_sel], is_ctrl[_sel], grp[_sel]
        print(
            f"[SCREEN] training cells capped {_cap} (of {len(_sel)} sampled)",
            flush=True,
        )
    X = np.vstack([X_train, X_held_ctrl])
    obs = pd.DataFrame(
        {
            "condition": np.concatenate(
                [
                    np.where(is_ctrl, "control", "stimulated"),
                    np.full(len(X_held_ctrl), "control"),
                ]
            ),
            "cell_type": np.concatenate([grp, np.full(len(X_held_ctrl), HELD)]),
        }
    )
    obs.index = [f"c{i}" for i in range(len(obs))]
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = genes

    with tempfile.TemporaryDirectory() as td:
        inp = str(Path(td) / "in.h5ad")
        adata.write_h5ad(inp)
        cmd = [
            sys.executable,
            str(_screen_dir()),
            "-in",
            inp,
            "-ou",
            td,
            "--label",
            HELD,
            "--condition_key",
            "condition",
            "--cell_type_key",
            "cell_type",
            "--ctrl_key",
            "control",
            "--stim_key",
            "stimulated",
            "--latent_dim",
            os.environ.get("IVCBENCH_SCREEN_LATENT", "100"),
            "--batch_size",
            os.environ.get("IVCBENCH_SCREEN_BATCH", "64"),
            "--epochs",
            os.environ.get("IVCBENCH_SCREEN_EPOCHS", "40"),
            "--full_quadratic",
            "False",
            "--activation",
            "leaky_relu",
            "--optimizer",
            "Adam",
        ]
        print("[SCREEN] " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=td, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-4000:])
            print(r.stderr[-4000:])
            raise SystemExit("SCREEN failed")
        got = Path(td) / f"SCREEN_{HELD}.h5ad"
        if not got.exists():
            raise SystemExit(f"SCREEN produced no output at {got}")
        out = ad.read_h5ad(got)
        key = f"{HELD}_SCREEN"
        sel = np.asarray(out.obs["condition"]).astype(str) == key
        if not sel.any():
            raise SystemExit(f"no '{key}' rows in SCREEN output")
        P = out[sel].X
        P = np.asarray(P.todense() if hasattr(P, "todense") else P, dtype=np.float32)

    profile = P.mean(0)
    np.savez(
        out_path,
        pred_perts=np.array([stim_label], dtype=object),
        pred_means=profile[None, :].astype(np.float32),
    )
    print(
        f"[SCREEN] wrote prediction for '{stim_label}' ({P.shape[0]} predicted cells)",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
