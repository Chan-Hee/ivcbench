#!/usr/bin/env python
"""STATE on the held-DONOR split (T2), training and inference kept apart — `ivc-state` env.

Invoked by scripts/state_soskic.py (set IVCBENCH_STATE_SOSKIC_RUNNER to override).

WHY A SECOND RUNNER. state_soskic_runner.py puts the held donor's own 0 h cells into the SAME
AnnData as the training cells -- once tagged with the held stim label (the query) and once tagged
`control`, so state's basal mapper can see them. The installed loader then routes that file through
[fewshot], and _split_fewshot_celltype hands EVERY control cell of the lineage to the train subset:

    cell_load/data_modules/perturbation_dataloader.py:789-812
        ctrl_indices_shuffled = rng.permutation(ctrl_indices)
        subset = ds.to_subset_dataset("train", train_pert_indices, ctrl_indices_shuffled)
        self.train_datasets.append(subset)

and state/configs/data/perturbation.yaml sets should_yield_control_cells: true, so those control
cells are themselves fitted on. The held donor's controls therefore enter the fit. No held RESPONSE
leaks -- its 16 h cells are never present -- but the benchmark's contract is that the fit is
train_idx and nothing else, which T1 and T5c now honour and T2 did not.

basal_mapping_strategy=batch does NOT fix this. It changes which control is DRAWN as the basal
partner; it does not change which cells are in the train subset. That distinction was got wrong
once already, in the comment this runner replaces.

WHAT THIS DOES INSTEAD, the same two-file shape as state_c1_split_runner.py:
  train.h5ad / train.toml   training donors only -- the held donor appears nowhere
  infer.h5ad / infer.toml   the held donor's OWN 0 h cells, as the query and as the basal pool
`state tx predict --toml` swaps the inference config over the saved training config
(state/_cli/_tx/_predict.py:15-19, :145-150) and reloads the checkpoint unchanged (:234).

The held axis here is the DONOR while cell_type is the LINEAGE, so the inference TOML routes each
lineage present to `test`; the file holds only the held donor, so that is exactly its cells.

basal_mapping_strategy=batch is kept, and now means what it says in both phases: batch_col is the
donor, so a training cell draws its basal from its OWN donor and lineage, and at predict time the
held donor's query draws from the held donor's own controls -- the only donor in that file.

COORDINATE. Left alone by default. A translation into state's ReLU/[0,14] output box was tried and
measured worse on this cell (see IVCBENCH_STATE_BOXMAP below); the measurement is reported, the
mechanism is not asserted.
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

STIM = "stim"          # one shared label: the stimulation is SEEN in every donor


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import torch
    from scipy import sparse

    steps = int(os.environ.get("IVCBENCH_STATE_STEPS", "400"))
    max_cells = int(os.environ.get("IVCBENCH_STATE_MAXCELLS", "50000"))
    per_lineage_cap = int(os.environ.get("IVCBENCH_STATE_T2_QUERY", "300"))
    state_py = sys.executable

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    celltype_train = np.array([str(c) for c in d["celltype_train"]], dtype=object)
    gem_train = np.array([str(g) for g in d["gem_train"]], dtype=object)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    celltype_inf = np.array([str(c) for c in d["celltype_inf"]], dtype=object)
    gem_inf = np.array([str(g) for g in d["gem_inf"]], dtype=object)
    held_label = str(d["held_label"])              # e.g. 'D348' or 'stim_D348'
    held_donors = sorted(set(gem_inf.tolist()))

    # The held donor must not be a training donor -- that is the split.
    keep = ~np.isin(gem_train, held_donors)
    if not keep.all():
        X, is_ctrl = X[keep], is_ctrl[keep]
        celltype_train, gem_train = celltype_train[keep], gem_train[keep]

    if X.shape[0] > max_cells:      # stratified cap over (label x lineage x donor)
        rng0 = np.random.default_rng(0)
        labels = np.char.add(
            np.where(is_ctrl, "control", STIM).astype(str),
            np.char.add("|", np.char.add(celltype_train.astype(str),
                                         np.char.add("|", gem_train.astype(str)))),
        )
        parts = []
        for lab, cnt in zip(*np.unique(labels, return_counts=True)):
            idx = np.where(labels == lab)[0]
            take = max(1, int(round(max_cells * cnt / X.shape[0])))
            parts.append(idx if take >= cnt else rng0.choice(idx, take, replace=False))
        sel = np.sort(np.concatenate(parts))
        X, is_ctrl = X[sel], is_ctrl[sel]
        celltype_train, gem_train = celltype_train[sel], gem_train[sel]

    work = Path(tempfile.mkdtemp(prefix="state_t2_"))

    # ---- dataset 1: TRAINING ONLY. The held donor appears nowhere. --------------------------
    tr = ad.AnnData(sparse.csr_matrix(X.astype(np.float32)))
    tr.var_names = genes
    tr.obs["gene"] = np.where(is_ctrl, "control", STIM).astype(object)
    tr.obs["cell_type"] = celltype_train.astype(object)
    tr.obs["gem_group"] = gem_train.astype(object)
    h5_train = work / "train.h5ad"
    tr.write_h5ad(h5_train)
    assert not set(tr.obs["gem_group"]) & set(held_donors), "held donor leaked into training"

    # ---- dataset 2: INFERENCE ONLY. The held donor's own 0 h cells. --------------------------
    rng = np.random.default_rng(0)
    inf_X, inf_p, inf_ct, inf_gem = [], [], [], []
    scored_lineages, empty_lineages = [], []
    for lin in sorted(set(celltype_inf.tolist())):
        idx = np.where(celltype_inf == lin)[0]
        if len(idx) == 0:
            empty_lineages.append(lin)
            continue
        if len(idx) > per_lineage_cap:
            idx = rng.choice(idx, per_lineage_cap, replace=False)
        for tag in (STIM, "control"):      # the query, and the basal pool it must draw from
            inf_X.append(X_ctrl_inf[idx])
            inf_p.append(np.array([tag] * len(idx), dtype=object))
            inf_ct.append(np.array([lin] * len(idx), dtype=object))
            inf_gem.append(gem_inf[idx].astype(object))
        scored_lineages.append(lin)
    if not scored_lineages:
        raise RuntimeError("STATE-soskic-split: held donor has no 0 h cells to predict from")

    inf = ad.AnnData(sparse.csr_matrix(np.vstack(inf_X).astype(np.float32)))
    inf.var_names = genes
    inf.obs["gene"] = np.concatenate(inf_p)
    inf.obs["cell_type"] = np.concatenate(inf_ct)
    inf.obs["gem_group"] = np.concatenate(inf_gem)
    h5_infer = work / "infer.h5ad"
    inf.write_h5ad(h5_infer)

    # The (donor, lineage) basal pool must be non-empty for every scored lineage, or batch.py
    # :154-160 falls back to the lineage-wide pool -- which in this file is the same donor, but
    # say so rather than assume it.
    pools = {lin: int(((inf.obs["cell_type"] == lin) & (inf.obs["gene"] == "control")).sum())
             for lin in scored_lineages}
    assert all(v > 0 for v in pools.values()), f"empty basal pool: {pools}"
    print(
        f"[STATE-soskic] SPLIT: train {tr.n_obs} cells / {len(set(gem_train))} donors "
        f"(held {held_donors} absent) | infer {inf.n_obs} cells, held donor only; "
        f"basal pool per (donor, lineage) {pools}"
        + (f" | lineages with no 0 h cells, DECLINED: {empty_lineages}" if empty_lineages else ""),
        flush=True,
    )

    # EMPTY-STRONG side rep: one constant feature for the single seen stim label.
    fdim = 8
    torch.save({STIM: torch.ones(fdim, dtype=torch.float32),
                "control": torch.zeros(fdim, dtype=torch.float32)}, work / "pert_features.pt")

    (work / "train.toml").write_text("\n".join(
        ["[datasets]", f'ds = "{h5_train}"', "", "[training]", 'ds = "train"']))
    (work / "infer.toml").write_text("\n".join(
        ["[datasets]", f'ds = "{h5_infer}"', "", "[training]", 'ds = "train"', "", "[zeroshot]"]
        + [f'"ds.{lin}" = "test"' for lin in scored_lineages]))

    out_dir = work / "run"
    common = [
        f"data.kwargs.toml_config_path={work / 'train.toml'}",
        "data.kwargs.pert_col=gene",
        "data.kwargs.control_pert=control",
        "data.kwargs.cell_type_key=cell_type",
        "data.kwargs.batch_col=gem_group",
        # batch_col is the donor, so this keys the basal pool on (donor, lineage) in BOTH phases.
        "data.kwargs.basal_mapping_strategy=batch",
        f"data.kwargs.perturbation_features_file={work / 'pert_features.pt'}",
        "data.kwargs.embed_key=null",
        "data.kwargs.output_space=all",
    ]
    r = subprocess.run(
        [state_py, "-m", "state", "tx", "train", "data=perturbation", "model=state_sm",
         f"training.max_steps={steps}", "training.val_freq=100000",
         "training.ckpt_every_n_steps=100000", f"output_dir={out_dir}", "name=ivc", *common],
        capture_output=True, text=True, timeout=7200, cwd=work)
    if r.returncode != 0:
        raise RuntimeError("state tx train failed:\n" + r.stderr[-3500:])

    r = subprocess.run(
        [state_py, "-m", "state", "tx", "predict", "--output-dir", str(out_dir / "ivc"),
         "--profile", "anndata", "--predict-only", "--toml", str(work / "infer.toml")],
        capture_output=True, text=True, timeout=3600, cwd=work)
    if r.returncode != 0:
        raise RuntimeError("state tx predict failed:\n" + r.stderr[-3500:])

    from state_output import prediction_path

    pa = ad.read_h5ad(str(prediction_path(out_dir)))
    pX = pa.X.toarray() if sparse.issparse(pa.X) else np.asarray(pa.X)
    pg = (pa.obs["gene"].astype(str).to_numpy() if "gene" in pa.obs
          else pa.obs.iloc[:, 0].astype(str).to_numpy())
    pct = (pa.obs["cell_type"].astype(str).to_numpy() if "cell_type" in pa.obs
           else np.array(["?"] * len(pg)))

    # One profile per lineage. A lineage with no rows back is DECLINED by name -- never filled
    # with the control mean, which downstream is indistinguishable from a real "no response".
    pred_perts, pred_means, declined = [], [], []
    for lin in scored_lineages + empty_lineages:
        m = (pg == STIM) & (pct == lin)
        pred_perts.append(f"{held_label}::{lin}")
        if lin in scored_lineages and m.sum():
            pred_means.append(pX[m].mean(0, dtype=np.float64)[: len(genes)].astype(np.float32))
            declined.append(False)
        else:
            print(f"[decline] STATE-soskic {held_label}::{lin}: no predicted rows returned",
                  flush=True)
            pred_means.append(np.zeros(len(genes), dtype=np.float32))
            declined.append(True)
    shutil.rmtree(work, ignore_errors=True)
    if all(declined):
        raise RuntimeError("STATE-soskic-split: no lineage produced a prediction")
    np.savez(
        out_path,
        pred_perts=np.array(pred_perts, dtype=object),
        pred_means=np.vstack(pred_means).astype(np.float32),
        pred_declined=np.asarray(declined, dtype=bool),
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
