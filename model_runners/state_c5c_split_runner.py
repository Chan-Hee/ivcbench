#!/usr/bin/env python
"""STATE on the held-LINEAGE compound split (T5c), train and inference kept apart — `ivc-state`.

Invoked by ivcbench.baselines.heavy.StateC5cSplit.

WHY A SEPARATE RUNNER. state_c5_runner.py serves the unseen-COMPOUND split (T5u), where holding the
test compounds out of training is the point. On T5c the held axis is the LINEAGE and every compound
is SEEN, but that runner is handed the same payload shape and puts the test compounds -- here the
seen ones -- into the fewshot `test` list. cell_load then excludes those labels from training
(cell_load/data_modules/perturbation_dataloader.py:768-787), so the run silently becomes a
compound-label holdout: the model is asked about compounds it was never trained on, on a split whose
whole claim is that they were. Two further mismatches follow from the same file: obs['cell_type'] is
set to the constant "PBMC", so the held lineage is not a cell type the model can be asked about at
all, and the query cells are drawn from the TRAINING control pool rather than the held lineage's own
controls, which is the basal state the task specifies.

WHAT THIS DOES INSTEAD. The same two-file shape as state_c1_split_runner.py:
  train.h5ad / train.toml   training lineages only, every compound present as a training label
  infer.h5ad / infer.toml   the held lineage's OWN control cells, re-tagged with each compound
The released CLI supports it directly: `state tx predict --toml` swaps the inference config over the
saved training config (state/_cli/_tx/_predict.py:15-19, :145-150) and reloads the checkpoint
unchanged (:234). No model code is modified -- only which cells each phase may see.

Leak-safety is unchanged: the held lineage's TREATED cells never appear in either file. Its control
cells are the model's input, which is what the split defines as available.
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
    per_test = int(os.environ.get("IVCBENCH_STATE_C5C_QUERY", "200"))
    state_py = sys.executable

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    held_lineage = str(d["held_lineage"])
    celltype_train = np.array([str(c) for c in d["celltype_train"]], dtype=object)

    if "fingerprint_keys" not in d:
        raise RuntimeError("STATE-C5c: payload has no fingerprint_* (compound side-rep)")
    fp_by_cpd = {
        str(k): np.asarray(v, dtype=np.float32)
        for k, v in zip(d["fingerprint_keys"], d["fingerprint_vals"])
    }
    fp_dim = len(next(iter(fp_by_cpd.values())))

    # The held lineage must not be a training cell type -- that is the split.
    keep = celltype_train != held_lineage
    if not keep.all():
        X, pert_train, is_ctrl = X[keep], pert_train[keep], is_ctrl[keep]
        celltype_train = celltype_train[keep]

    if X.shape[0] > max_cells:  # stratified cap over (label x lineage)
        rng0 = np.random.default_rng(0)
        labels = np.char.add(
            np.where(is_ctrl, "control", pert_train).astype(str),
            np.char.add("|", celltype_train.astype(str)),
        )
        sel_parts = []
        for lab, cnt in zip(*np.unique(labels, return_counts=True)):
            idx = np.where(labels == lab)[0]
            take = max(1, int(round(max_cells * cnt / X.shape[0])))
            sel_parts.append(idx if take >= cnt else rng0.choice(idx, take, replace=False))
        sel = np.sort(np.concatenate(sel_parts))
        X, pert_train, is_ctrl = X[sel], pert_train[sel], is_ctrl[sel]
        celltype_train = celltype_train[sel]

    train_cpds = sorted({c for c in pert_train[~is_ctrl] if c in fp_by_cpd})
    # On this split the evaluated compounds are the SEEN ones: predict only what training carries.
    pred_cpds = [c for c in test_perts if c in fp_by_cpd and c in set(train_cpds)]
    dropped = [c for c in test_perts if c not in set(pred_cpds)]
    if not pred_cpds:
        raise RuntimeError("STATE-C5c: no evaluated compound is present in training")

    work = Path(tempfile.mkdtemp(prefix="state_c5c_"))

    # ---- dataset 1: TRAINING ONLY. The held lineage appears nowhere. ------------------------
    tr = ad.AnnData(sparse.csr_matrix(X.astype(np.float32)))
    tr.var_names = genes
    tr.obs["gene"] = np.where(is_ctrl, "control", pert_train).astype(object)
    tr.obs["cell_type"] = celltype_train.astype(object)
    tr.obs["gem_group"] = "b0"
    h5_train = work / "train.h5ad"
    tr.write_h5ad(h5_train)
    assert held_lineage not in set(tr.obs["cell_type"]), "held lineage leaked into training"

    # ---- dataset 2: INFERENCE ONLY. The held lineage's own controls. ------------------------
    # One basal sample, reused for every compound, so differences between predicted profiles come
    # from the compound and not from which control cells each one happened to draw.
    rng = np.random.default_rng(0)
    n_ctrl = X_ctrl_inf.shape[0]
    if n_ctrl == 0:
        raise RuntimeError("STATE-C5c: held lineage has no control cells to predict from")
    take = min(per_test, n_ctrl)
    basal = X_ctrl_inf[rng.choice(n_ctrl, take, replace=False)] if take < n_ctrl else X_ctrl_inf

    inf_X = [basal]                                   # the basal pool, labelled control
    inf_p = [np.array(["control"] * len(basal), dtype=object)]
    for c in pred_cpds:                               # one query block per evaluated compound
        inf_X.append(basal)
        inf_p.append(np.array([c] * len(basal), dtype=object))
    inf = ad.AnnData(sparse.csr_matrix(np.vstack(inf_X).astype(np.float32)))
    inf.var_names = genes
    inf.obs["gene"] = np.concatenate(inf_p)
    inf.obs["cell_type"] = held_lineage
    inf.obs["gem_group"] = "b0"
    h5_infer = work / "infer.h5ad"
    inf.write_h5ad(h5_infer)
    print(
        f"[STATE-C5c-split] held={held_lineage} | train cells={tr.n_obs} over "
        f"{len(set(celltype_train))} lineage(s), {len(train_cpds)} compound(s) | "
        f"inference cells={inf.n_obs} ({len(basal)} own controls x {len(pred_cpds)} compound(s))"
        + (f" | {len(dropped)} evaluated compound(s) absent from training: {dropped[:5]}" if dropped else ""),
        flush=True,
    )

    feats = {c: torch.tensor(fp_by_cpd[c], dtype=torch.float32) for c in train_cpds}
    feats["control"] = torch.zeros(fp_dim, dtype=torch.float32)
    fpath = work / "pert_features.pt"
    torch.save(feats, fpath)

    # TRAINING toml: one dataset, all of it training. No [fewshot] -- every compound is seen, which
    # is the split's premise; no [zeroshot] -- the held lineage is not in this file at all.
    (work / "train.toml").write_text("\n".join([
        "[datasets]", f'ds = "{h5_train}"', "", "[training]", 'ds = "train"',
    ]))
    # INFERENCE toml: the held lineage routed to `test`, swapped in at predict time.
    (work / "infer.toml").write_text("\n".join([
        "[datasets]", f'ds = "{h5_infer}"', "", "[training]", 'ds = "train"', "",
        "[zeroshot]", f'"ds.{held_lineage}" = "test"',
    ]))

    out_dir = work / "run"
    common = [
        f"data.kwargs.toml_config_path={work / 'train.toml'}",
        "data.kwargs.pert_col=gene",
        "data.kwargs.control_pert=control",
        "data.kwargs.cell_type_key=cell_type",
        "data.kwargs.batch_col=gem_group",
        f"data.kwargs.perturbation_features_file={fpath}",
        "data.kwargs.embed_key=null",
        "data.kwargs.output_space=all",
    ]
    train_cmd = [
        state_py, "-m", "state", "tx", "train",
        "data=perturbation", "model=state_sm",
        f"training.max_steps={epochs_steps}",
        "training.val_freq=100000",
        "training.ckpt_every_n_steps=100000",
        f"output_dir={out_dir}", "name=ivc", *common,
    ]
    r = subprocess.run(train_cmd, capture_output=True, text=True, timeout=5400, cwd=work)
    if r.returncode != 0:
        raise RuntimeError("state tx train failed:\n" + r.stderr[-3500:])

    pred_cmd = [
        state_py, "-m", "state", "tx", "predict",
        "--output-dir", str(out_dir / "ivc"),
        "--profile", "anndata", "--predict-only",
        "--toml", str(work / "infer.toml"),
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
    pred_perts, pred_means = [], []
    for c in pred_cpds:
        m = pg == c
        if m.sum():
            pred_perts.append(c)
            pred_means.append(pX[m].mean(0)[: len(genes)].astype(np.float32))
    shutil.rmtree(work, ignore_errors=True)
    if not pred_perts:
        raise RuntimeError("STATE-C5c: no held-lineage predictions recovered")
    np.savez(
        out_path,
        pred_perts=np.array(pred_perts, dtype=object),
        pred_means=np.vstack(pred_means).astype(np.float32),
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
