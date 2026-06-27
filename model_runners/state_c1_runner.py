#!/usr/bin/env python
"""STATE runner (Kang C1 cytokine LOCT, ①Cytokine×Hybrid, EMPTY-STRONG) — `ivc-state` env.

Invoked by scripts/state_kang.py:  <ivc-state python> state_c1_runner.py <in.npz> <out.npz>

CELL-axis analogue of state_soskic_runner.py. Here the held UNIT is a *cell-type lineage*, not a donor
or a gene: the perturbation (IFN-β stimulation) is SEEN in every training lineage. We map the problem
onto cell_load's fewshot machinery with a single shared stim label `stim`: training lineages' stim cells
go to the fewshot `train` arm, the held lineage's `stim` query cells go to the `test` arm. The held
lineage's OWN control cells are re-tagged `stim` (the test query) — its real IFN-β cells are NEVER a
training target → leak-safe by construction.

EMPTY-STRONG: the single stim label shares one constant perturbation-feature vector (a 1-vector). STATE
gets NO held-lineage distinguishing side rep — it must transfer the shared control→stim (IFN-β)
transition through the lineage (cell_type_key) + donor (gem_group / batch) context. This is the honest
hybrid lower-bound (from-scratch ST, embed_key=null, no SE-600M download), directly comparable to the
scGen / CPA / CellOT / scPRAM Kang LOCT rows.

Context preserved end-to-end: cell_type_key = the held vs training lineages (the LOCT axis); gem_group =
donor (the eval stratum on C1 LOCT). The predicted anndata keeps `gene` + `cell_type` + `gem_group`, so
we aggregate per (stim × donor) and emit one predicted profile per donor stratum, keyed `stim::<donor>`.

Payload (built by the driver in the benchmark .venv):
  X_train, is_control_train, pert_train ('stim'/'control'), celltype_train (lineage), gem_train (donor),
  X_ctrl_inf (held lineage's OWN control cells), celltype_inf (held lineage), gem_inf (held donors),
  held_lineage, genes.
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
    is_ctrl = d["is_control_train"].astype(bool)
    celltype_train = np.array([str(c) for c in d["celltype_train"]], dtype=object)   # lineage
    gem_train = np.array([str(g) for g in d["gem_train"]], dtype=object)             # donor
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)                                  # held lineage's controls
    celltype_inf = np.array([str(c) for c in d["celltype_inf"]], dtype=object)
    gem_inf = np.array([str(g) for g in d["gem_inf"]], dtype=object)
    held_lineage = str(d["held_lineage"])
    STIM = "stim"                                          # single shared (seen) cytokine label

    # stratified cap on training cells (keep each (stim/ctrl × lineage) representative)
    if X.shape[0] > max_cells:
        rng0 = np.random.default_rng(0)
        labels = np.char.add(np.where(is_ctrl, "control", STIM).astype(str),
                             np.char.add("|", celltype_train.astype(str)))
        keep = []
        for lab, cnt in zip(*np.unique(labels, return_counts=True)):
            idx = np.where(labels == lab)[0]
            take = max(1, int(round(max_cells * cnt / X.shape[0])))
            keep.append(idx if take >= cnt else rng0.choice(idx, take, replace=False))
        sel = np.sort(np.concatenate(keep))
        X, is_ctrl = X[sel], is_ctrl[sel]
        celltype_train, gem_train = celltype_train[sel], gem_train[sel]

    work = Path(tempfile.mkdtemp(prefix="state_c1_"))

    # AnnData blocks: (1) all training cells (control + train-lineage stim), (2) the held lineage's OWN
    # control cells re-tagged `stim` per donor — the leak-safe test query. Real IFN-β is never a target.
    blk_X = [X]
    blk_pert = [np.where(is_ctrl, "control", STIM).astype(object)]
    blk_ct = [celltype_train.astype(object)]
    blk_gem = [gem_train.astype(object)]

    rng = np.random.default_rng(0)
    donors_inf = sorted(set(gem_inf))
    per_test_cap = 200
    held_donors = []
    for don in donors_inf:
        idx = np.where(gem_inf == don)[0]
        if len(idx) == 0:
            continue
        if len(idx) > per_test_cap:
            idx = rng.choice(idx, per_test_cap, replace=False)
        # (a) held lineage's controls re-tagged `stim` per donor = the leak-safe TEST query (its real
        #     IFN-β is never seen). (b) the SAME controls also kept as `control` under the held lineage,
        #     so the zeroshot held cell type has a non-empty train-observational control subset (the
        #     batch sampler needs every train subset non-empty). Controls are NOT the held perturbation,
        #     so re-using them as observational train data is leak-safe.
        blk_X.append(X_ctrl_inf[idx])
        blk_pert.append(np.array([STIM] * len(idx), dtype=object))
        blk_ct.append(np.array([held_lineage] * len(idx), dtype=object))
        blk_gem.append(np.array([don] * len(idx), dtype=object))
        blk_X.append(X_ctrl_inf[idx])
        blk_pert.append(np.array(["control"] * len(idx), dtype=object))
        blk_ct.append(np.array([held_lineage] * len(idx), dtype=object))
        blk_gem.append(np.array([don] * len(idx), dtype=object))
        held_donors.append(don)
    if not held_donors:
        raise RuntimeError("STATE-C1: held lineage has no inference (control) cells")

    Xall = np.vstack(blk_X).astype(np.float32)
    adata = ad.AnnData(sparse.csr_matrix(Xall))
    adata.var_names = genes
    adata.obs["gene"] = np.concatenate(blk_pert)
    adata.obs["cell_type"] = np.concatenate(blk_ct)
    adata.obs["gem_group"] = np.concatenate(blk_gem)
    h5 = work / "kang.h5ad"
    adata.write_h5ad(h5)

    # EMPTY-STRONG side rep: one shared constant feature for the single stim label (no per-lineage signal)
    fdim = 8
    feats = {STIM: torch.ones(fdim, dtype=torch.float32),
             "control": torch.zeros(fdim, dtype=torch.float32)}
    fpath = work / "pert_features.pt"
    torch.save(feats, fpath)

    # cell_load TOML — LOCT = ZEROSHOT held cell type. [training] marks ds a training dataset, so every
    # UNLISTED (training) lineage becomes a regular training cell type (its control + `stim` cells train).
    # [zeroshot] routes the HELD lineage's `stim` cells (= its re-tagged controls, the test query) to
    # `test`; the held lineage's controls go to a train-observational subset. Real IFN-β never trains.
    lines = ['[datasets]', f'ds = "{h5}"', '', '[training]', 'ds = "train"', '',
             '[zeroshot]', f'"ds.{held_lineage}" = "test"']
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
        raise RuntimeError("STATE-C1: no prediction h5ad produced")
    pa = ad.read_h5ad(str(preds[-1]))
    pX = pa.X.toarray() if sparse.issparse(pa.X) else np.asarray(pa.X)
    pg = pa.obs["gene"].astype(str).to_numpy() if "gene" in pa.obs else pa.obs.iloc[:, 0].astype(str).to_numpy()
    pgem = pa.obs["gem_group"].astype(str).to_numpy() if "gem_group" in pa.obs else np.array(["?"] * len(pg))

    # one predicted profile per (stim × donor); key = 'stim::<donor>' (donor = the C1-LOCT eval stratum)
    pred_perts, pred_means = [], []
    for don in held_donors:
        m = (pg == STIM) & (pgem == don)
        if m.sum():
            pred_perts.append(f"{STIM}::{don}")
            pred_means.append(pX[m].mean(0)[:len(genes)].astype(np.float32))
    shutil.rmtree(work, ignore_errors=True)
    if not pred_perts:
        raise RuntimeError("STATE-C1: no held-lineage predictions recovered")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
