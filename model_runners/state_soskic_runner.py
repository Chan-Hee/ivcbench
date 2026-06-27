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
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)   # 'stim_<D>' / 'control'
    is_ctrl = d["is_control_train"].astype(bool)
    celltype_train = np.array([str(c) for c in d["celltype_train"]], dtype=object)
    gem_train = np.array([str(g) for g in d["gem_train"]], dtype=object)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)                          # held donor's OWN 0h cells
    celltype_inf = np.array([str(c) for c in d["celltype_inf"]], dtype=object)
    gem_inf = np.array([str(g) for g in d["gem_inf"]], dtype=object)
    held_label = str(d["held_label"])                                         # e.g. 'stim_D348'

    # stratified cap on training cells (keep each label×lineage representative)
    if X.shape[0] > max_cells:
        rng0 = np.random.default_rng(0)
        labels = np.char.add(np.where(is_ctrl, "control", pert_train).astype(str),
                             np.char.add("|", celltype_train.astype(str)))
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
        blk_X.append(X_ctrl_inf[idx])
        blk_pert.append(np.array([held_label] * len(idx), dtype=object))
        blk_ct.append(np.array([lin] * len(idx), dtype=object))
        blk_gem.append(gem_inf[idx].astype(object))
        held_lineages.append(lin)
    if not held_lineages:
        raise RuntimeError("STATE-soskic: held donor has no inference (0h) cells")

    Xall = np.vstack(blk_X).astype(np.float32)
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
    lines = ['[datasets]', f'ds = "{h5}"', '', '[fewshot]']
    for lin in held_lineages:
        lines += [f'[fewshot."ds.{lin}"]',
                  'train = [' + ", ".join(f'"{l}"' for l in train_labels) + ']',
                  'test = [' + f'"{held_label}"' + ']']
    (work / "data.toml").write_text("\n".join(lines))

    out_dir = work / "run"
    common = [f"data.kwargs.toml_config_path={work / 'data.toml'}", "data.kwargs.pert_col=gene",
              "data.kwargs.control_pert=control", "data.kwargs.cell_type_key=cell_type",
              "data.kwargs.batch_col=gem_group", f"data.kwargs.perturbation_features_file={fpath}",
              "data.kwargs.embed_key=null", "data.kwargs.output_space=all"]
    train_cmd = [state_py, "-m", "state", "tx", "train", "data=perturbation", "model=state_sm",
                 f"training.max_steps={epochs_steps}", "training.val_freq=100000",
                 "training.ckpt_every_n_steps=100000", f"output_dir={out_dir}", "name=ivc", *common]
    r = subprocess.run(train_cmd, capture_output=True, text=True, timeout=5400, cwd=work)
    if r.returncode != 0:
        raise RuntimeError("state tx train failed:\n" + r.stderr[-3500:])

    pred_cmd = [state_py, "-m", "state", "tx", "predict", "--output-dir", str(out_dir / "ivc"),
                "--profile", "anndata", "--predict-only"]
    r = subprocess.run(pred_cmd, capture_output=True, text=True, timeout=3600, cwd=work)
    if r.returncode != 0:
        raise RuntimeError("state tx predict failed:\n" + r.stderr[-3500:])

    preds = sorted(Path(out_dir).rglob("adata_pred.h5ad")) or sorted(Path(out_dir).rglob("*.h5ad"))
    if not preds:
        raise RuntimeError("STATE-soskic: no prediction h5ad produced")
    pa = ad.read_h5ad(str(preds[-1]))
    pX = pa.X.toarray() if sparse.issparse(pa.X) else np.asarray(pa.X)
    pg = pa.obs["gene"].astype(str).to_numpy() if "gene" in pa.obs else pa.obs.iloc[:, 0].astype(str).to_numpy()
    pct = pa.obs["cell_type"].astype(str).to_numpy() if "cell_type" in pa.obs else np.array(["?"] * len(pg))

    # one predicted profile per (held label × lineage); key = '<held_label>::<lineage>'
    pred_perts, pred_means = [], []
    for lin in held_lineages:
        m = (pg == held_label) & (pct == lin)
        if m.sum():
            pred_perts.append(f"{held_label}::{lin}")
            pred_means.append(pX[m].mean(0)[:len(genes)].astype(np.float32))
    shutil.rmtree(work, ignore_errors=True)
    if not pred_perts:
        raise RuntimeError("STATE-soskic: no held-donor predictions recovered")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
