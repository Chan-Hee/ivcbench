#!/usr/bin/env python
"""STATE runner (Soskic CD4-activation DONOR-LODO, ②Donor×Hybrid, EMPTY-STRONG) — `ivc-state` env.

Invoked by scripts/state_soskic.py:  <ivc-state python> state_soskic_runner.py <in.npz> <out.npz>

DONOR-axis analogue of state_runner.py / state_c5_runner.py. Here the held UNIT is a *donor*, not a
perturbation: the perturbation ("stimulation", 0h→16h) is SEEN in every training donor. We map the
problem onto cell_load's fewshot machinery by tagging the stimulated cells with a per-donor label
`stim_<donor>`: training donors' `stim_<D>` labels go to the fewshot `train` arm, the held donor's
`stim_<Dheld>` goes to the `test` arm. The held donor's OWN 0h control cells are re-tagged
`stim_<Dheld>` (test query) — its real 16h cells are NEVER a training target → leak-safe by
construction.

EMPTY-STRONG: every stim label shares the SAME constant perturbation-feature vector (a 1-vector). STATE
therefore gets NO per-donor distinguishing side rep — it must transfer the shared control→stim
transition through the lineage (cell_type_key) + batch (gem_group) context. This is the honest
hybrid lower-bound (from-scratch ST, embed_key=null, no SE-600M download).

Context preserved end-to-end: cell_type_key = CD4 lineage (Naive/Memory, the eval stratum), gem_group =
donor. The predicted anndata keeps `gene` + `cell_type`, so we aggregate per (held label × lineage) and
emit one predicted profile per lineage stratum, keyed `stim_<Dheld>::<lineage>`.

Payload (built by the driver in the benchmark .venv):
  X_train, is_control_train, pert_train (donor-stim labels / 'control'), celltype_train, gem_train,
  X_ctrl_inf, celltype_inf, gem_inf, held_label, genes.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

if os.environ.get("IVCBENCH_STATE_FORCE_CPU") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import torch
    from scipy import sparse

    epochs_steps = int(os.environ.get("IVCBENCH_STATE_STEPS", "400"))
    max_cells = int(os.environ.get("IVCBENCH_STATE_MAXCELLS", "50000"))
    state_py = sys.executable

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array(
        [str(p) for p in d["pert_train"]], dtype=object
    )  # 'stim_<D>' / 'control'
    is_ctrl = d["is_control_train"].astype(bool)
    celltype_train = np.array([str(c) for c in d["celltype_train"]], dtype=object)
    gem_train = np.array([str(g) for g in d["gem_train"]], dtype=object)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)  # held donor's OWN 0h cells
    celltype_inf = np.array([str(c) for c in d["celltype_inf"]], dtype=object)
    gem_inf = np.array([str(g) for g in d["gem_inf"]], dtype=object)
    held_label = str(d["held_label"])  # e.g. 'stim_D348'

    # stratified cap on training cells (keep each label×lineage representative)
    if X.shape[0] > max_cells:
        rng0 = np.random.default_rng(0)
        labels = np.char.add(
            np.where(is_ctrl, "control", pert_train).astype(str),
            np.char.add("|", celltype_train.astype(str)),
        )
        keep = []
        for lab, cnt in zip(*np.unique(labels, return_counts=True)):
            idx = np.where(labels == lab)[0]
            take = max(1, int(round(max_cells * cnt / X.shape[0])))
            keep.append(idx if take >= cnt else rng0.choice(idx, take, replace=False))
        sel = np.sort(np.concatenate(keep))
        X, pert_train, is_ctrl = X[sel], pert_train[sel], is_ctrl[sel]
        celltype_train, gem_train = celltype_train[sel], gem_train[sel]

    work = Path(tempfile.mkdtemp(prefix="state_soskic_"))
    train_labels = sorted({p for p in pert_train[~is_ctrl] if p != "control"})
    if not train_labels:
        raise RuntimeError("STATE-soskic: no training stim labels")

    # AnnData blocks: (1) all training cells (control + train-donor stim), (2) held donor's OWN 0h cells
    # re-tagged with held_label per lineage — the leak-safe test query. Real 16h is never a target.
    blk_X = [X]
    blk_pert = [np.where(is_ctrl, "control", pert_train).astype(object)]
    blk_ct = [celltype_train.astype(object)]
    blk_gem = [gem_train.astype(object)]

    rng = np.random.default_rng(0)
    lineages_inf = sorted(set(celltype_inf))
    per_test_cap = 200
    held_lineages = []
    for lin in lineages_inf:
        idx = np.where(celltype_inf == lin)[0]
        if len(idx) == 0:
            continue
        if len(idx) > per_test_cap:
            idx = rng.choice(idx, per_test_cap, replace=False)
        # (a) the held donor's own 0h cells re-tagged with the held stim label = the leak-safe TEST
        #     query (its real 16h response is never seen).
        blk_X.append(X_ctrl_inf[idx])
        blk_pert.append(np.array([held_label] * len(idx), dtype=object))
        blk_ct.append(np.array([lin] * len(idx), dtype=object))
        blk_gem.append(gem_inf[idx].astype(object))
        # (b) the SAME cells also kept as `control`. Without this they are invisible to the basal
        #     mapper: state's random strategy draws the basal cell from the CONTROL-labelled pool of
        #     the same cell type (cell_load/mapping_strategies/random.py), and what the network
        #     actually reads is ctrl_cell_emb (state/tx/models/state_transition.py). Stim-labelled
        #     only, the held donor's own basal state never reached the forward pass at all, and the
        #     basal cell came from other donors -- so the held donor contributed nothing.
        #     This mirrors state_c1_runner.py, which already does it. Controls are not the held
        #     perturbation, so re-using them as observational data is leak-safe.
        blk_X.append(X_ctrl_inf[idx])
        blk_pert.append(np.array(["control"] * len(idx), dtype=object))
        blk_ct.append(np.array([lin] * len(idx), dtype=object))
        blk_gem.append(gem_inf[idx].astype(object))
        held_lineages.append(lin)
    if not held_lineages:
        raise RuntimeError("STATE-soskic: held donor has no inference (0h) cells")
    # basal_mapping_strategy=batch keys the control pool on (donor, lineage) and falls back to the
    # lineage-wide pool only when that pool is empty. Check it is never empty for a stratum we
    # score, so the fallback -- which would hand a held-donor query a training donor's control --
    # cannot fire unnoticed.
    _held_donor_ids = set(np.asarray(gem_inf, dtype=object).tolist())
    for _lin in held_lineages:
        _n = int(((celltype_inf == _lin)).sum())
        if _n == 0:
            raise RuntimeError(
                f"STATE-soskic: lineage {_lin!r} is scored but the held donor has no control "
                "cells for it; (donor, lineage) basal mapping would fall back across donors"
            )
    print(
        f"[STATE-soskic] basal pool per (donor, lineage): held donor(s) {sorted(_held_donor_ids)} "
        + ", ".join(f"{l}={int((celltype_inf == l).sum())}" for l in held_lineages),
        flush=True,
    )

    Xall = np.vstack(blk_X).astype(np.float32)
    # Map the space AFFINELY into state's output box before handing it over, and undo it on the
    # way out. The Soskic T2 coordinate is covariate-regressed/scaled -- measured here at 81.9%
    # negative -- while state's prediction path ends in a ReLU and a [0, 14] clip
    # (state/tx/models/state_transition.py; _cli/_tx/_predict.py:462 clips both pred and real).
    #
    # Translation ALONE is not enough, and shipped as a fix it made the cell worse. On D348 the
    # shift is +7.29 and the post-shift maximum is 31.8, so the top of the distribution lands past
    # the clip and is truncated: pearson-delta 0.1186 without the shift, 0.0616 with it, and the
    # energy distance moves the same way (6.57 -> 10.37). Scaling into the box as well removes the
    # truncation, and both metrics survive it: pearson-delta is invariant to a global affine map
    # (the shift cancels in delta = pred - control, the scale cancels in the correlation), and the
    # profiles written below are mapped back, so the energy distance is computed in the harness's
    # own coordinate.
    #
    # MEASURED, and the answer is to leave the coordinate alone. One donor (D348), everything else
    # held fixed:
    #     no map                     pearson-delta 0.1793   energy distance 6.60
    #     translate (+7.29)                        0.0638                  10.60
    #     translate and scale to box               0.0252                  10.50
    # The translation's own diagnostic says why: post-shift maximum 31.8 against state's [0, 14]
    # clip, so it trades an unreachable NEGATIVE tail for a truncated POSITIVE one, and the
    # positive tail is where the signal is. Scaling into the box removes the truncation and loses
    # more, because it compresses the dynamic range. Pearson-delta is a correlation of deltas, so
    # the ReLU floor costs less than either remedy.
    # IVCBENCH_STATE_BOXMAP=shift or =box restores the two variants for re-measurement.
    _mode = os.environ.get("IVCBENCH_STATE_BOXMAP", "none")
    _lo, _hi0 = float(Xall.min()), float(Xall.max())
    STATE_SHIFT, STATE_SCALE = 0.0, 1.0
    if _mode != "none" and _lo < 0.0:
        STATE_SHIFT = -_lo
        span = (_hi0 - _lo)
        if _mode == "box" and span > 0:
            STATE_SCALE = min(1.0, 13.5 / span)      # 13.5 leaves margin under the 14.0 clip
        Xall = (Xall + STATE_SHIFT) * STATE_SCALE
        print(
            f"[STATE-soskic] box map: shift +{STATE_SHIFT:.4f}, scale x{STATE_SCALE:.4f} "
            f"(space was {100 * (np.vstack(blk_X) < 0).mean():.1f}% negative, range "
            f"[{_lo:.2f}, {_hi0:.2f}]); post-map max={float(Xall.max()):.4f} against state's "
            f"[0, 14] clip. Undone before the profiles are returned.",
            flush=True,
        )
    adata = ad.AnnData(sparse.csr_matrix(Xall))
    adata.var_names = genes
    adata.obs["gene"] = np.concatenate(blk_pert)
    adata.obs["cell_type"] = np.concatenate(blk_ct)
    adata.obs["gem_group"] = np.concatenate(blk_gem)
    h5 = work / "soskic.h5ad"
    adata.write_h5ad(h5)

    # EMPTY-STRONG side rep: one shared constant feature for every stim label (no per-donor signal)
    fdim = 8
    one = torch.ones(fdim, dtype=torch.float32)
    feats = {lab: one.clone() for lab in (train_labels + [held_label])}
    feats["control"] = torch.zeros(fdim, dtype=torch.float32)
    fpath = work / "pert_features.pt"
    torch.save(feats, fpath)

    # fewshot TOML — one [fewshot."ds.<lineage>"] per lineage present in the held arm; train labels in
    # `train`, held label in `test`. Lineage = cell_type_key (the eval stratum).
    lines = ["[datasets]", f'ds = "{h5}"', "", "[fewshot]"]
    for lin in held_lineages:
        lines += [
            f'[fewshot."ds.{lin}"]',
            "train = [" + ", ".join(f'"{l}"' for l in train_labels) + "]",
            "test = [" + f'"{held_label}"' + "]",
        ]
    (work / "data.toml").write_text("\n".join(lines))

    out_dir = work / "run"
    common = [
        f"data.kwargs.toml_config_path={work / 'data.toml'}",
        "data.kwargs.pert_col=gene",
        "data.kwargs.control_pert=control",
        "data.kwargs.cell_type_key=cell_type",
        "data.kwargs.batch_col=gem_group",
        # Basal mapping by BATCH, not the installed default "random".
        #
        # gem_group is the donor here, so "batch" keys the control pool on (donor x lineage)
        # (cell_load/mapping_strategies/batch.py:59-77, :129-167) -- which is exactly this task's
        # contract: predict a held donor's stimulated state from THAT donor's own controls. Under
        # the default "random" the pool is keyed on lineage alone
        # (mapping_strategies/random.py:103-115), so two things went wrong at once: a held-donor
        # query cell could be paired with a training donor's control, and during fitting the
        # fewshot route hands EVERY control of the lineage to the train subset
        # (perturbation_dataloader.py:789-812 shuffles all ctrl_indices into all three splits),
        # including the held donor's -- so the fit was not train_idx.
        #
        # batch.py:154-160 falls back to the lineage-wide pool only when the (batch, cell_type)
        # pool is EMPTY; the assertion above keeps that branch unreachable for every evaluated
        # stratum.
        # env-overridable so the two strategies can be compared on the same donor; "batch" is the
        # one this task's contract calls for and the default.
        f"data.kwargs.basal_mapping_strategy={os.environ.get('IVCBENCH_STATE_BASAL', 'batch')}",
        f"data.kwargs.perturbation_features_file={fpath}",
        "data.kwargs.embed_key=null",
        "data.kwargs.output_space=all",
    ]
    train_cmd = [
        state_py,
        "-m",
        "state",
        "tx",
        "train",
        "data=perturbation",
        "model=state_sm",
        f"training.max_steps={epochs_steps}",
        "training.val_freq=100000",
        "training.ckpt_every_n_steps=100000",
        f"output_dir={out_dir}",
        "name=ivc",
        *common,
    ]
    r = subprocess.run(
        train_cmd, capture_output=True, text=True, timeout=5400, cwd=work
    )
    if r.returncode != 0:
        raise RuntimeError("state tx train failed:\n" + r.stderr[-3500:])

    pred_cmd = [
        state_py,
        "-m",
        "state",
        "tx",
        "predict",
        "--output-dir",
        str(out_dir / "ivc"),
        "--profile",
        "anndata",
        "--predict-only",
    ]
    r = subprocess.run(pred_cmd, capture_output=True, text=True, timeout=3600, cwd=work)
    if r.returncode != 0:
        raise RuntimeError("state tx predict failed:\n" + r.stderr[-3500:])

    from state_output import prediction_path

    pa = ad.read_h5ad(str(prediction_path(out_dir)))
    pX = pa.X.toarray() if sparse.issparse(pa.X) else np.asarray(pa.X)
    pg = (
        pa.obs["gene"].astype(str).to_numpy()
        if "gene" in pa.obs
        else pa.obs.iloc[:, 0].astype(str).to_numpy()
    )
    pct = (
        pa.obs["cell_type"].astype(str).to_numpy()
        if "cell_type" in pa.obs
        else np.array(["?"] * len(pg))
    )

    # one predicted profile per (held label × lineage); key = '<held_label>::<lineage>'
    pred_perts, pred_means = [], []
    for lin in held_lineages:
        m = (pg == held_label) & (pct == lin)
        if m.sum():
            pred_perts.append(f"{held_label}::{lin}")
            # undo the box map in float64 before casting back, so the round trip adds no more
            # float32 rounding than the single forward cast the model's own pipeline already does.
            # The inverse must divide by the scale BEFORE subtracting the shift -- the forward map
            # is (x + shift) * scale.
            pred_means.append(
                (
                    pX[m].mean(0, dtype=np.float64)[: len(genes)] / STATE_SCALE - STATE_SHIFT
                ).astype(np.float32)
            )
    shutil.rmtree(work, ignore_errors=True)
    if not pred_perts:
        raise RuntimeError("STATE-soskic: no held-donor predictions recovered")
    np.savez(
        out_path,
        pred_perts=np.array(pred_perts, dtype=object),
        pred_means=np.vstack(pred_means).astype(np.float32),
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
