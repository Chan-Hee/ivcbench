#!/usr/bin/env python
"""GEARS runner — executed inside the `scgpt` conda env (cell-gears + torch-geometric).

Invoked by ivcbench.baselines.heavy.GEARS:
    <scgpt_python> gears_runner.py <in.npz> <out.npz>

Leak-safety: the payload's X_train / pert_train already have the held-out target genes removed
(ivcbench builds the split before serialising). GEARS therefore never sees a held-gene cell; it
trains on the train perturbations and *predicts* each held gene from the GO graph (gene2go) — its
native unseen-perturbation capability. We assign GEARS a held gene only if it is in its perturbable
gene universe (gene2go ∩ HVG); otherwise the adapter falls back to the control mean for that gene.

gene2go: looked up from $IVCBENCH_GENE2GO or benchmark/data/_assets/gears/gene2go_all.pkl, copied
into the per-run PertData dir (cell-gears would otherwise try to download it).
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def _gene2go_path() -> str:
    p = os.environ.get("IVCBENCH_GENE2GO")
    if p and Path(p).exists():
        return p
    here = Path(__file__).resolve().parents[1]  # benchmark/
    cand = here / "data" / "_assets" / "gears" / "gene2go_all.pkl"
    if cand.exists():
        return str(cand)
    raise FileNotFoundError("gene2go_all.pkl not found; set $IVCBENCH_GENE2GO "
                            "(see model_runners/README.md).")


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import scanpy as sc  # noqa: F401
    import torch
    from scipy import sparse
    from gears import GEARS, PertData

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    is_ctrl = d["is_control_train"].astype(bool)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})

    # GEARS requires every training perturbation's gene to be IN the expression matrix
    # (get_pert_idx looks up its index). Drop train cells whose perturbed gene is unmeasured
    # (e.g. CRISPRa targets filtered out of the panel) — a leak-safe subset of train. Held genes
    # not in the panel are simply not predicted (adapter falls back to control).
    gene_set = set(genes)
    ok = is_ctrl | np.array([p in gene_set for p in pert_train])
    X, pert_train, is_ctrl = X[ok], pert_train[ok], is_ctrl[ok]

    # ---- build the GEARS-format AnnData from the leak-safe training cells ----
    cond = np.where(is_ctrl, "ctrl", np.array([f"{p}+ctrl" for p in pert_train], dtype=object))
    # GEARS builds a per-cell graph → cap training cells (stratified by condition) for tractability
    cap = int(os.environ.get("IVCBENCH_GEARS_MAX_CELLS", "40000"))
    if X.shape[0] > cap:
        rng = np.random.default_rng(0)
        per = max(1, cap // max(1, len(set(cond))))
        keep = []
        for c in set(cond):
            ci = np.where(cond == c)[0]
            keep.append(ci if len(ci) <= per else rng.choice(ci, per, replace=False))
        keep = np.sort(np.concatenate(keep))
        X, cond = X[keep], cond[keep]
    adata = ad.AnnData(sparse.csr_matrix(X))
    adata.var_names = genes
    adata.var["gene_name"] = genes
    adata.obs["condition"] = cond
    adata.obs["cell_type"] = "Tcell"
    adata.uns["log1p"] = {"base": None}

    work = Path(tempfile.mkdtemp(prefix="gears_"))
    data_dir = work / "data"
    data_dir.mkdir(parents=True)
    shutil.copy(_gene2go_path(), data_dir / "gene2go_all.pkl")
    seed = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cwd = os.getcwd()
    os.chdir(work)
    try:
        pert_data = PertData(str(data_dir))
        pert_data.new_data_process(dataset_name="train", adata=adata)
        pert_data.load(data_path=str(data_dir / "train"))
        # split over the TRAIN genes only (held genes are absent) → leak-safe internal val/test
        pert_data.prepare_split(split="simulation", seed=seed, train_gene_set_size=0.75)
        pert_data.get_dataloader(batch_size=64, test_batch_size=128)

        model = GEARS(pert_data, device=device)
        model.model_initialize(hidden_size=64)
        model.train(epochs=int(os.environ.get("IVCBENCH_GEARS_EPOCHS", "15")))

        # GEARS can only perturb genes in its gene2go ∩ HVG universe
        perturbable = set(getattr(model, "pert_list", []) or getattr(pert_data, "pert_names", []))
        gene_pos = {g: i for i, g in enumerate(genes)}
        pred_perts, pred_means = [], []
        for g in test_perts:
            if perturbable and g not in perturbable:
                continue
            try:
                out = model.predict([[g]])
            except Exception:
                continue
            vec = np.asarray(list(out.values())[0], dtype=np.float32).ravel()
            if vec.shape[0] != len(genes):
                continue
            pred_perts.append(g)
            pred_means.append(vec)
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)

    if not pred_perts:
        raise RuntimeError("GEARS predicted no held genes (none in gene2go ∩ HVG universe).")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
