#!/usr/bin/env python
"""AttentionPert runner — executed in the `scgpt` conda env (torch-geometric; attnpert via sys.path).

Invoked by ivcbench.baselines.heavy.AttentionPert:  <scgpt python> attentionpert_runner.py <in> <out>

AttentionPert (a GEARS-style graph model with gene2vec attention) has NO native unseen-gene predict;
its `evaluate` only scores perturbations already in its PertData. To keep ivcbench's leak-safe-by-
construction guarantee (held-gene cells NEVER enter training — same as GEARS/scGPT/scGen/CPA), we:
  1. train AttentionPert on the leak-safe train cells only (held genes removed upstream);
  2. predict each held gene ON THE FLY by building control-cell graphs tagged with that gene's
     perturbation one-hot (`create_cell_graph`, x=[ctrl_expr, pert_onehot]) and running model.forward.
The held gene must be in the gene panel (force-included) so it has a gene2vec/graph node; its cells
are never seen in training. This mirrors GEARS' `predict([[gene]])` for attnpert's model.

gene2vec: a per-panel (n_genes, 200) embedding generated from gene2vec_dim_200_iter_9_w2v.txt
(Gaussian-sampled for genes missing from the w2v), as in AttentionPert's gene2vec_example.py.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ATTN_SRC = os.environ.get("IVCBENCH_ATTNPERT_SRC")
if not ATTN_SRC:
    raise FileNotFoundError("set $IVCBENCH_ATTNPERT_SRC to the AttentionPert source directory")
sys.path.insert(0, ATTN_SRC)


def _gene2vec(genes, work):
    """(n_genes, 200) embedding aligned to `genes`; w2v lookup + Gaussian fallback for missing genes."""
    w2v = Path(ATTN_SRC) / "gene2vec_dim_200_iter_9_w2v.txt"
    d, vecs = {}, []
    with open(w2v) as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\n").split(" ")
            v = np.array([float(x) for x in p[1:-1]] if p[-1] == "" else [float(x) for x in p[1:]])
            d[p[0]] = v
            vecs.append(v)
    vecs = np.array(vecs)
    mean, cov = vecs.mean(0), np.cov(vecs.T)
    rng = np.random.default_rng(0)
    out = np.array([d[g] if g in d else rng.multivariate_normal(mean, cov) for g in genes], dtype=np.float32)
    np.save(work / "data" / "train" / "gene2vec.npy", out)
    return out


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import numpy as np
    import scanpy as sc
    import torch
    from scipy import sparse
    from torch_geometric.data import Batch
    from attnpert import PertData
    from attnpert.attnpert import ATTNPERT_RECORD_TRAIN
    from attnpert.model import PL_PW_non_add_Model

    epochs = int(os.environ.get("IVCBENCH_ATTNPERT_EPOCHS", "20"))
    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    is_ctrl = d["is_control_train"].astype(bool)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    gene_set = set(genes)

    # drop train cells whose perturbed gene is not in the panel (attnpert needs a gene-graph node)
    ok = is_ctrl | np.array([p in gene_set for p in pert_train])
    X, pert_train, is_ctrl = X[ok], pert_train[ok], is_ctrl[ok]

    # cap training cells (Chen ≈300k cells times out the 3600s subprocess budget) — stratified by
    # perturbation so every train gene + the control pool keep representative counts.
    max_cells = int(os.environ.get("IVCBENCH_ATTNPERT_MAXCELLS", "40000"))
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

    cond = np.where(is_ctrl, "ctrl", np.array([f"{p}+ctrl" for p in pert_train], dtype=object)).astype(str)

    adata = ad.AnnData(sparse.csr_matrix(X))
    adata.var_names = genes
    adata.var["gene_name"] = genes
    adata.obs["condition"] = cond
    adata.obs["cell_type"] = "Tcell"
    adata.obs["condition_name"] = adata.obs["condition"]
    adata.uns["log1p"] = {"base": None}

    work = Path(tempfile.mkdtemp(prefix="attnpert_"))
    (work / "data" / "train").mkdir(parents=True)
    # attnpert's get_go_auto/get_similarity_network read GO support files from ./data (cwd-relative)
    for fn in ("gene2go_all.pkl", "go.csv", "essential_all_data_pert_genes.pkl"):
        src = Path(ATTN_SRC) / fn
        if src.exists():
            shutil.copy(src, work / "data" / fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cwd = os.getcwd()
    os.chdir(work)
    try:
        pert_data = PertData(str(work / "data"))
        pert_data.new_data_process(dataset_name="train", adata=adata)
        pert_data.load(data_path=str(work / "data" / "train"))
        pert_data.prepare_split(split="simulation", seed=1, train_gene_set_size=0.75)
        pert_data.get_dataloader(batch_size=128, test_batch_size=128)
        _gene2vec(list(pert_data.gene_names), work)

        model = ATTNPERT_RECORD_TRAIN(pert_data, device=device, weight_bias_track=False,
                                      proj_name="attnpert", exp_name="ivc")
        model.model_initialize(
            hidden_size=64, model_class=PL_PW_non_add_Model,
            gene2vec_args={"gene2vec_file": str(work / "data" / "train" / "gene2vec.npy")},
            pert_local_min_weight=0.75, pert_local_conv_K=1, pert_weight_heads=2,
            pert_weight_head_dim=64, pert_weight_act="softmax", non_add_beta=5e-2,
            record_pred=False, exp_name="ivc")
        model.train(epochs=epochs, valid_every=max(1, epochs))

        net = model.best_model if getattr(model, "best_model", None) is not None else model.model
        net.eval().to(device)
        gene_names = list(pert_data.gene_names)
        gpos = {g: i for i, g in enumerate(gene_names)}
        n_full = len(gene_names)
        ctrl_X = np.asarray(pert_data.ctrl_adata.X.todense() if sparse.issparse(pert_data.ctrl_adata.X)
                            else pert_data.ctrl_adata.X, dtype=np.float32)
        n_ctrl = min(128, ctrl_X.shape[0])
        base = ctrl_X[np.random.default_rng(0).choice(ctrl_X.shape[0], n_ctrl, replace=False)]

        def cell_graph(expr_row, pidx):
            pf = np.zeros(n_full, np.float32); pf[pidx] = 1
            fm = torch.Tensor(np.concatenate([expr_row[None, :], pf[None, :]])).T  # (n_genes, 2)
            from torch_geometric.data import Data
            return Data(x=fm, pert=f"g+ctrl", de_idx=[-1] * 20)

        pred_perts, pred_means = [], []
        with torch.no_grad():
            for g in test_perts:
                if g not in gpos:
                    continue
                graphs = [cell_graph(base[i], gpos[g]) for i in range(n_ctrl)]
                batch = Batch.from_data_list(graphs).to(device)
                p = net(batch)
                p = p[:, :n_full] if p.dim() == 2 else p
                pred_perts.append(g)
                pred_means.append(np.asarray(p.cpu(), np.float32).mean(0)[:len(genes)])
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)

    if not pred_perts:
        raise RuntimeError("AttentionPert: no held genes in the gene panel to predict")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
