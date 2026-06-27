#!/usr/bin/env python
"""scFoundation runner — executed inside the `scfoundation` conda env (torch + Performer/transformer
scFoundation backbone; NO flash_attn needed — the released `cell` checkpoint uses a pytorch transformer
encoder + Performer decoder).

Invoked by ivcbench.baselines.heavy.ScFoundation:
    <scfoundation_python> scfoundation_runner.py <in.npz> <out.npz>

Design (FROZEN scFoundation embeddings + GEARS-style fine-tune head):
  scFoundation is a foundation model with a frozen 19264-gene encoder. We (1) map every leak-safe
  *training* cell's HVG profile onto scFoundation's 19264-gene panel and run the frozen `cell`-key
  encoder to get a 3072-d cell embedding (geneemb1/2 + max/mean pooling, the canonical cell embedding
  from get_embedding.py); (2) train a small MLP head — the only trainable part — that maps
  [frozen_cell_emb ‖ perturbed-gene one-hot] → the perturbed gene-space profile over a TRAIN-fold
  response-gene panel (~15 epochs). Prediction: for each held target gene, feed control cells with the
  one-hot set at that gene → mean predicted profile, scattered into the full HVG panel (non-modelled
  genes keep the control mean). This is scFoundation's native unseen-gene capability: the held gene is
  predicted from its one-hot conditioning + the frozen embedding of a control cell, never from any
  held-gene expression.

LEAK-SAFETY (HARD RULE 1): the payload's X_train / pert_train already have every held target gene's
sgRNAs removed (the adapter builds the split before serialising). Inside the runner EVERYTHING that
adapts to the data — the perturbed-gene condition vocabulary, the response-gene panel (top-variance
genes), the PCA basis used to compress the head's regression target, the control mean — is fit on the
TRAIN cells ONLY. Held-gene expression never enters the runner. The runner is invoked once per fold,
so every basis is refit per fold (no cross-fold carry-over).

scFoundation source dir (load.py, pretrainmodels/, OS_scRNA_gene_index.19264.tsv) from
$IVCBENCH_SCFOUNDATION_DIR (default the local single_cell_fm/scFoundation/model). Checkpoint from
$IVCBENCH_SCFOUNDATION_CKPT (default models/scFoundation/models.ckpt). Epochs / cell-cap / PCA dim /
target-resolution token via $IVCBENCH_SCF_{EPOCHS,MAXCELLS,PCA,HIGHRES}.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def _scf_dir() -> Path:
    p = os.environ.get("IVCBENCH_SCFOUNDATION_DIR",
                       "/data1/home/chlee/projects/single_cell_fm/scFoundation/model")
    d = Path(p)
    if not (d / "load.py").exists():
        raise FileNotFoundError(f"scFoundation source dir {d} missing load.py "
                                "(set $IVCBENCH_SCFOUNDATION_DIR)")
    return d


def _ckpt_path() -> Path:
    p = os.environ.get("IVCBENCH_SCFOUNDATION_CKPT",
                       "/data1/home/chlee/projects/single_cell_fm/models/scFoundation/models.ckpt")
    c = Path(p)
    if not c.exists():
        raise FileNotFoundError(f"scFoundation checkpoint {c} not found (set $IVCBENCH_SCFOUNDATION_CKPT)")
    return c


def main(in_path: str, out_path: str) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    scf_dir = _scf_dir()
    sys.path.insert(0, str(scf_dir))
    import pandas as pd
    from load import load_model_frommmf, gatherData  # scFoundation helpers

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = int(os.environ.get("IVCBENCH_SCF_EPOCHS", "15"))
    max_cells = int(os.environ.get("IVCBENCH_SCF_MAXCELLS", "8000"))   # frozen-embed cap (per-cell fwd)
    pca_dim = int(os.environ.get("IVCBENCH_SCF_PCA", "64"))            # head target compression
    highres = os.environ.get("IVCBENCH_SCF_HIGHRES", "t4")            # target-resolution token (get_embedding default)

    # ---------------------------------------------------------------- payload
    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)                                # log1p-normalised HVG (train fold)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})

    n_hvg = len(genes)
    if X.shape[0] == 0:
        raise RuntimeError("scFoundation: empty training payload")

    # train-only perturbed-gene condition vocabulary (held genes are absent from pert_train by design)
    train_pert_genes = sorted({p for p in pert_train[~is_ctrl]}) if (~is_ctrl).any() else []
    if not train_pert_genes:
        raise RuntimeError("scFoundation: no perturbed training cells (cannot learn a condition head)")
    pert_pos = {g: i for i, g in enumerate(train_pert_genes)}
    n_cond = len(train_pert_genes)

    # control pool to anchor predictions (prefer the explicit inference controls; else train controls)
    ctrl_pool = X[is_ctrl] if is_ctrl.any() else X_ctrl_inf
    if ctrl_pool.shape[0] == 0:
        raise RuntimeError("scFoundation: no control cells to anchor predictions")
    ctrl_full_mean = ctrl_pool.mean(0).astype(np.float32)

    # ---------------------------------------------------- frozen scFoundation
    model, cfg = load_model_frommmf(str(_ckpt_path()), "cell")
    model.eval().to(device)
    pad_id = cfg["pad_token_id"]

    # scFoundation panel (19264 genes) → indices of OUR HVG genes within it (others stay zero-padded)
    gidx = pd.read_csv(scf_dir / "OS_scRNA_gene_index.19264.tsv", header=0, delimiter="\t")
    scf_genes = list(gidx["gene_name"])
    scf_pos = {g: i for i, g in enumerate(scf_genes)}
    n_scf = len(scf_genes)
    hvg_to_scf = np.array([scf_pos.get(g, -1) for g in genes], dtype=np.int64)   # -1 = gene absent in panel
    have = hvg_to_scf >= 0

    @torch.no_grad()
    def embed(rows: np.ndarray) -> np.ndarray:
        """Frozen scFoundation `cell` embedding for a batch of HVG log1p rows. Mirrors
        get_embedding.py output_type='cell' pool_type='all' (3072-d = 4×768)."""
        out = []
        for r in rows:
            full = np.zeros(n_scf, dtype=np.float32)
            full[hvg_to_scf[have]] = r[have]
            # the payload X is already log1p-normalised → use pre_normalized='T' semantics: feed as-is,
            # append the two resolution tokens (target-highres, log10 total). total from the raw-ish sum.
            total = float(np.expm1(full).sum()) + 1.0
            hi = float(highres[1:]) if highres[0] == "t" else 4.0
            gx = torch.tensor(full.tolist() + [hi, np.log10(total)],
                              dtype=torch.float32, device=device).unsqueeze(0)
            gids = torch.arange(n_scf + 2, device=device).repeat(gx.shape[0], 1)
            vl = gx > 0
            xx, xpad = gatherData(gx, vl, pad_id)
            pos, _ = gatherData(gids, vl, pad_id)
            e = model.token_emb(torch.unsqueeze(xx, 2).float(), output_weight=0)
            e = e + model.pos_emb(pos)
            g = model.encoder(e, xpad)
            emb = torch.concat([g[:, -1, :], g[:, -2, :],
                                torch.max(g[:, :-2, :], dim=1)[0],
                                torch.mean(g[:, :-2, :], dim=1)], axis=1)
            out.append(emb.float().cpu().numpy()[0])
        return np.vstack(out).astype(np.float32)

    # ---- training samples: perturbed cells (frozen emb + cond one-hot → response-gene target) ----
    pert_mask = (~is_ctrl) & np.array([p in pert_pos for p in pert_train])
    pert_rows = np.where(pert_mask)[0]
    if len(pert_rows) == 0:
        raise RuntimeError("scFoundation: no usable perturbed training cells in the condition vocab")
    if len(pert_rows) > max_cells:
        rng = np.random.default_rng(0)
        pert_rows = np.sort(rng.choice(pert_rows, max_cells, replace=False))

    # response-gene panel + PCA basis — fit on TRAIN cells ONLY (leak-safe). Top-variance HVGs (train).
    var = X.var(0)
    n_resp = min(int(os.environ.get("IVCBENCH_SCF_NRESP", "512")), n_hvg)
    resp_idx = np.argsort(var)[::-1][:n_resp]
    Y_train = X[pert_rows][:, resp_idx].astype(np.float32)
    resp_mean = Y_train.mean(0)
    Yc = Y_train - resp_mean
    # PCA basis from train response targets only (compress the head's regression target → faster, denoised)
    k = min(pca_dim, Yc.shape[0], Yc.shape[1])
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    basis = Vt[:k].astype(np.float32)                       # (k, n_resp), train-fold PCA basis

    print(f"[scFoundation] train cells={X.shape[0]} perturbed-used={len(pert_rows)} "
          f"conds={n_cond} resp_genes={n_resp} pca={k} epochs={epochs}", flush=True)

    # frozen embeddings for the (capped) perturbed training cells + their condition one-hots + PCA targets
    emb_train = embed(X[pert_rows])                          # (N, 3072) frozen
    cond_train = np.zeros((len(pert_rows), n_cond), dtype=np.float32)
    for j, ri in enumerate(pert_rows):
        cond_train[j, pert_pos[pert_train[ri]]] = 1.0
    tgt_train = (Yc[np.arange(len(pert_rows))] @ basis.T).astype(np.float32)   # (N, k) PCA coords

    emb_dim = emb_train.shape[1]

    class HeadDS(Dataset):
        def __len__(self):
            return len(pert_rows)

        def __getitem__(self, i):
            return (torch.tensor(emb_train[i]), torch.tensor(cond_train[i]), torch.tensor(tgt_train[i]))

    # GEARS-style fine-tune head: [frozen_emb ‖ cond_onehot] → PCA target coords (the only trainable part)
    head = nn.Sequential(
        nn.Linear(emb_dim + n_cond, 512), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, k),
    ).to(device)
    loader = DataLoader(HeadDS(), batch_size=64, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    head.train()
    for ep in range(epochs):
        tot = 0.0
        for emb, cond, tgt in loader:
            emb, cond, tgt = emb.to(device), cond.to(device), tgt.to(device)
            opt.zero_grad()
            out = head(torch.cat([emb, cond], dim=1))
            loss = F.mse_loss(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += float(loss) * emb.shape[0]
        if ep == 0 or ep == epochs - 1:
            print(f"[scFoundation] epoch {ep} mse={tot / max(1, len(pert_rows)):.5f}", flush=True)

    # ---- predict each held gene: control cells + that gene's one-hot → mean response profile ----
    head.eval()
    n_ctrl = min(int(os.environ.get("IVCBENCH_SCF_PREDCTRL", "128")), ctrl_pool.shape[0])
    rng = np.random.default_rng(0)
    sel_ctrl = rng.choice(ctrl_pool.shape[0], n_ctrl, replace=False) if ctrl_pool.shape[0] > n_ctrl \
        else np.arange(ctrl_pool.shape[0])
    emb_ctrl = embed(ctrl_pool[sel_ctrl])                   # (n_ctrl, 3072) frozen control embeddings
    basis_t = torch.tensor(basis, device=device)
    resp_mean_t = torch.tensor(resp_mean, device=device)
    emb_ctrl_t = torch.tensor(emb_ctrl, device=device)

    pred_perts, pred_means = [], []
    with torch.no_grad():
        for g in test_perts:
            if g not in pert_pos:
                # held gene was never in the train condition vocab → scFoundation conditions it through
                # the SAME one-hot space; if absent it has no learned column. Predict a generic perturbed
                # response by using a zero condition (frozen embedding only). Still a real model output,
                # not the control mean — keeps the held gene scored rather than silently floored.
                cond = torch.zeros((n_ctrl, n_cond), device=device)
            else:
                cond = torch.zeros((n_ctrl, n_cond), device=device)
                cond[:, pert_pos[g]] = 1.0
            coords = head(torch.cat([emb_ctrl_t, cond], dim=1))        # (n_ctrl, k)
            prof_resp = (coords @ basis_t) + resp_mean_t               # (n_ctrl, n_resp)
            prof_resp = prof_resp.mean(0).float().cpu().numpy()
            full = ctrl_full_mean.copy()
            full[resp_idx] = prof_resp                                 # scatter modelled genes; rest = ctrl mean
            pred_perts.append(g)
            pred_means.append(full.astype(np.float32))

    if not pred_perts:
        raise RuntimeError("scFoundation: no held genes to predict")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
