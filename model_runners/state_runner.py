#!/usr/bin/env python
"""STATE runner — executed in the `ivc-state` conda env (arc-state; State Transition / `state tx`).

Invoked by ivcbench.baselines.heavy.STATE:  <ivc-state python> state_runner.py <in.npz> <out.npz>

STATE's ST model is designed for unseen-perturbation prediction (Virtual Cell Challenge). We keep
ivcbench's leak-safe-by-construction guarantee via cell_load's `fewshot` split: held genes go to the
`test` arm, train genes to `train`; the model trains only on train perturbations. Unseen genes are
representable because we supply a `perturbation_features_file` (a leak-safe control-only PCA gene
embedding per gene), so ST predicts a held gene from its features without ever seeing its cells.

Pipeline: build a cell_load h5ad + features.pt + TOML → `state tx train` (state_sm, raw expression,
embed_key=null → no SE-600M download) → `state tx predict` → read the per-perturbation predicted
profiles. CLI/Hydra-driven; bleeding-edge stack (torch 2.12).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# arc-state now runs torch 2.10.0+cu128 in this env (driver-compatible with the host CUDA-12.2);
# the GPU (L40) IS usable. We DON'T clear CUDA_VISIBLE_DEVICES here — the parallel dispatcher pins it
# (heavy.py SubprocessAdapter.cuda_device) so each STATE job lands on its assigned GPU; Lightning
# auto-detects it. Falls back to CPU only if no GPU is visible. (Set IVCBENCH_STATE_FORCE_CPU=1 to
# force CPU, e.g. for a driver regression.)
if os.environ.get("IVCBENCH_STATE_FORCE_CPU") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import pandas as pd
    import torch
    from scipy import sparse
    from sklearn.decomposition import PCA

    epochs_steps = int(os.environ.get("IVCBENCH_STATE_STEPS", "400"))
    n_emb = int(os.environ.get("IVCBENCH_STATE_GENE_EMB", "64"))
    state_py = sys.executable

    max_cells = int(os.environ.get("IVCBENCH_STATE_MAXCELLS", "50000"))

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    gpos = {g: i for i, g in enumerate(genes)}

    # cap training cells (Chen ≈300k OOMs/times-out the from-scratch ST model) — stratified by
    # perturbation so every train gene + the control pool keep representative counts.
    if X.shape[0] > max_cells:
        rng0 = np.random.default_rng(0)
        labels = np.where(is_ctrl, "control", pert_train).astype(str)
        keep = []
        uniq, counts = np.unique(labels, return_counts=True)
        for lab, cnt in zip(uniq, counts):
            idx = np.where(labels == lab)[0]
            take = max(1, int(round(max_cells * cnt / X.shape[0])))
            keep.append(idx if take >= cnt else rng0.choice(idx, take, replace=False))
        sel = np.sort(np.concatenate(keep))
        X, pert_train, is_ctrl = X[sel], pert_train[sel], is_ctrl[sel]

    # leak-safe gene embedding (control-only PCA gene-loadings) — also covers held genes
    ctrl_X = X[is_ctrl] if is_ctrl.any() else X_ctrl_inf
    k = int(min(n_emb, ctrl_X.shape[0] - 1, ctrl_X.shape[1]))
    gene_emb = PCA(n_components=max(2, k), random_state=0).fit(ctrl_X).components_.T  # (n_genes, k)

    work = Path(tempfile.mkdtemp(prefix="state_"))
    # build held-gene "test" cells from controls tagged with the held gene (STATE predicts the test
    # arm; held cells' real expression is NOT used as a training target — leak-safe via fewshot split).
    train_genes = sorted({p for p in pert_train[~is_ctrl] if p in gpos})
    pred_genes = [g for g in test_perts if g in gpos]

    # AnnData: train cells (control + train-gene perts) + control cells re-tagged as held genes (test arm)
    blocks_X, blocks_pert = [X], [np.where(is_ctrl, "control", pert_train).astype(object)]
    ctrl_pool = X[is_ctrl] if is_ctrl.any() else X_ctrl_inf
    rng = np.random.default_rng(0)
    per_test = min(200, ctrl_pool.shape[0])
    for g in pred_genes:                          # held-gene query cells = controls tagged g (test arm)
        idx = rng.choice(ctrl_pool.shape[0], per_test, replace=False)
        blocks_X.append(ctrl_pool[idx]); blocks_pert.append(np.array([g] * per_test, dtype=object))
    Xall = np.vstack(blocks_X).astype(np.float32)
    pert_all = np.concatenate(blocks_pert)
    adata = ad.AnnData(sparse.csr_matrix(Xall))
    adata.var_names = genes
    adata.obs["gene"] = pert_all
    adata.obs["cell_type"] = "Tcell"
    adata.obs["gem_group"] = "b0"
    h5 = work / "schmidt.h5ad"; adata.write_h5ad(h5)

    feats = {g: torch.tensor(gene_emb[gpos[g]], dtype=torch.float32) for g in (train_genes + pred_genes)}
    feats["control"] = torch.zeros(gene_emb.shape[1], dtype=torch.float32)
    fpath = work / "pert_features.pt"; torch.save(feats, fpath)

    toml = work / "data.toml"
    tl = ['[datasets]', f'ds = "{h5}"', '', '[fewshot]', '[fewshot."ds.Tcell"]',
          'train = [' + ", ".join(f'"{g}"' for g in train_genes) + ']',
          'test = [' + ", ".join(f'"{g}"' for g in pred_genes) + ']']
    toml.write_text("\n".join(tl))
    out_dir = work / "run"

    common = [f"data.kwargs.toml_config_path={toml}", "data.kwargs.pert_col=gene",
              "data.kwargs.control_pert=control", "data.kwargs.cell_type_key=cell_type",
              "data.kwargs.batch_col=gem_group", f"data.kwargs.perturbation_features_file={fpath}",
              "data.kwargs.embed_key=null", "data.kwargs.output_space=all"]
    train_cmd = [state_py, "-m", "state", "tx", "train", "data=perturbation", "model=state_sm",
                 f"training.max_steps={epochs_steps}", "training.val_freq=100000", "training.ckpt_every_n_steps=100000",
                 f"output_dir={out_dir}", "name=ivc", *common]
    r = subprocess.run(train_cmd, capture_output=True, text=True, timeout=5400, cwd=work)
    if r.returncode != 0:
        raise RuntimeError("state tx train failed:\n" + r.stderr[-3500:])

    # --predict-only: emit the prediction anndata and SKIP arc-state's internal MetricsEvaluator
    # (its cell-eval API mismatch crashes the eval step, which we don't need — we score ourselves).
    pred_cmd = [state_py, "-m", "state", "tx", "predict", "--output-dir", str(out_dir / "ivc"),
                "--profile", "anndata", "--predict-only"]
    r = subprocess.run(pred_cmd, capture_output=True, text=True, timeout=3600, cwd=work)
    if r.returncode != 0:
        raise RuntimeError("state tx predict failed:\n" + r.stderr[-3500:])

    # locate the predicted anndata and aggregate per held gene
    preds = sorted(Path(out_dir).rglob("*.h5ad"))
    if not preds:
        raise RuntimeError("STATE: no prediction h5ad produced")
    pa = ad.read_h5ad(str(preds[-1]))
    pX = pa.X.toarray() if sparse.issparse(pa.X) else np.asarray(pa.X)
    pg = pa.obs["gene"].astype(str).to_numpy() if "gene" in pa.obs else pa.obs.iloc[:, 0].astype(str).to_numpy()
    pred_perts, pred_means = [], []
    for g in pred_genes:
        m = pg == g
        if m.sum():
            pred_perts.append(g); pred_means.append(pX[m].mean(0)[:len(genes)].astype(np.float32))
    shutil.rmtree(work, ignore_errors=True)
    if not pred_perts:
        raise RuntimeError("STATE: no held-gene predictions recovered")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
