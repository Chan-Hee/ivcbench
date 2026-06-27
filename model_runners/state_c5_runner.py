#!/usr/bin/env python
"""STATE runner (C5, adapted*) — `ivc-state` env (arc-state; `state tx`).

Invoked by ivcbench.baselines.heavy.STATEc5:  <ivc-state python> state_c5_runner.py <in.npz> <out.npz>

C5 analogue of state_runner.py. STATE predicts an unseen perturbation from its `perturbation_features`
vector; here that vector is the compound's Morgan FINGERPRINT (instead of C3's control-PCA gene
embedding). cell_load's fewshot split puts train compounds in `train`, held compounds in `test`; the
held compound's query cells are controls re-tagged with it (its real treated cells are NEVER a training
target → leak-safe). embed_key=null (from-scratch ST, no SE-600M) — a conservative lower bound.
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
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)   # compound labels
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if "fingerprint_keys" not in d:
        raise RuntimeError("STATE-C5: payload has no fingerprint_* (compound side-rep)")
    fp_by_cpd = {str(k): np.asarray(v, dtype=np.float32)
                 for k, v in zip(d["fingerprint_keys"], d["fingerprint_vals"])}
    fp_dim = len(next(iter(fp_by_cpd.values())))

    if X.shape[0] > max_cells:                       # stratified cap (large control pools)
        rng0 = np.random.default_rng(0)
        labels = np.where(is_ctrl, "control", pert_train).astype(str)
        keep = []
        for lab, cnt in zip(*np.unique(labels, return_counts=True)):
            idx = np.where(labels == lab)[0]
            take = max(1, int(round(max_cells * cnt / X.shape[0])))
            keep.append(idx if take >= cnt else rng0.choice(idx, take, replace=False))
        sel = np.sort(np.concatenate(keep))
        X, pert_train, is_ctrl = X[sel], pert_train[sel], is_ctrl[sel]

    work = Path(tempfile.mkdtemp(prefix="state_c5_"))
    train_cpds = sorted({c for c in pert_train[~is_ctrl] if c in fp_by_cpd})
    pred_cpds = [c for c in test_perts if c in fp_by_cpd]

    # AnnData: train cells + control cells re-tagged as held compounds (the leak-safe test arm)
    blocks_X, blocks_p = [X], [np.where(is_ctrl, "control", pert_train).astype(object)]
    ctrl_pool = X[is_ctrl] if is_ctrl.any() else X_ctrl_inf
    rng = np.random.default_rng(0)
    per_test = min(200, ctrl_pool.shape[0])
    for c in pred_cpds:
        idx = rng.choice(ctrl_pool.shape[0], per_test, replace=False)
        blocks_X.append(ctrl_pool[idx]); blocks_p.append(np.array([c] * per_test, dtype=object))
    adata = ad.AnnData(sparse.csr_matrix(np.vstack(blocks_X).astype(np.float32)))
    adata.var_names = genes
    adata.obs["gene"] = np.concatenate(blocks_p)          # 'gene' column holds the compound label
    adata.obs["cell_type"] = "PBMC"; adata.obs["gem_group"] = "b0"
    h5 = work / "op3.h5ad"; adata.write_h5ad(h5)

    feats = {c: torch.tensor(fp_by_cpd[c], dtype=torch.float32) for c in (train_cpds + pred_cpds)}
    feats["control"] = torch.zeros(fp_dim, dtype=torch.float32)
    fpath = work / "pert_features.pt"; torch.save(feats, fpath)

    toml = work / "data.toml"
    toml.write_text("\n".join([
        '[datasets]', f'ds = "{h5}"', '', '[fewshot]', '[fewshot."ds.PBMC"]',
        'train = [' + ", ".join(f'"{c}"' for c in train_cpds) + ']',
        'test = [' + ", ".join(f'"{c}"' for c in pred_cpds) + ']']))
    out_dir = work / "run"
    common = [f"data.kwargs.toml_config_path={toml}", "data.kwargs.pert_col=gene",
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

    preds = sorted(Path(out_dir).rglob("*.h5ad"))
    if not preds:
        raise RuntimeError("STATE-C5: no prediction h5ad produced")
    pa = ad.read_h5ad(str(preds[-1]))
    pX = pa.X.toarray() if sparse.issparse(pa.X) else np.asarray(pa.X)
    pg = pa.obs["gene"].astype(str).to_numpy() if "gene" in pa.obs else pa.obs.iloc[:, 0].astype(str).to_numpy()
    pred_perts, pred_means = [], []
    for c in pred_cpds:
        m = pg == c
        if m.sum():
            pred_perts.append(c); pred_means.append(pX[m].mean(0)[:len(genes)].astype(np.float32))
    shutil.rmtree(work, ignore_errors=True)
    if not pred_perts:
        raise RuntimeError("STATE-C5: no held-compound predictions recovered")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
