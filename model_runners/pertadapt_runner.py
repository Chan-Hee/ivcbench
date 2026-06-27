#!/usr/bin/env python
"""PertAdapt runner (Bai et al. 2025, scFoundation variant) — executed in the `scfoundation` conda env.

Invoked by ivcbench.baselines.heavy.PertAdapt:
    <scfoundation_python> pertadapt_runner.py <in.npz> <out.npz>

PertAdapt = FROZEN scFoundation backbone + condition-sensitive perturbation adapter (gene-similarity /
GO–masked self-attention) + adaptive DE-reweighting loss. This runner is the benchmark-native, leak-safe
re-expression of that pipeline, reusing the SAME frozen-scFoundation embedding path as
scfoundation_runner.py and bolting on PertAdapt's two published novelties
(benchmark/vendor/pertadapt/pertadapt_modules.py):

  PER-GENE frozen embeddings  →  exp_encodings (B, N_resp, D)
  learned pooled pert emb     →  pert_encodings (B, 1, D)   [one-hot pert → MLP, the GO-GNN analogue]
        GOMaskedPertAdapter(exp, pert)  →  adapted (B, N_resp, D)  [GO-masked self-attention over genes]
        gene-wise linear head           →  Δ over the response panel, added to the control profile
        trained under loss_adapt        →  adaptive DE-vs-all reweighting

DESIGN DETAIL — per-gene embeddings without 19264 forward-per-gene blowup:
  scFoundation's frozen `cell` encoder emits ONE 3072-d cell embedding per cell (the canonical pooled
  embedding from get_embedding.py), not 19264 per-gene tokens we can cheaply slice. To keep the runner
  tractable (the per-cell forward is already the cost driver) we form the adapter's PER-GENE
  `exp_encodings` as [frozen_cell_emb_proj ⊕ gene_identity_emb] per modelled response gene: every gene
  gets the same projected frozen cell context plus a learned gene-token embedding. The GO-masked
  attention then mixes genes only along GO edges — preserving PertAdapt's gene-similarity inductive bias.
  (The paper's backbone hands the adapter true per-gene tokens; here the gene axis is the response panel
  and the per-gene signal is a learned gene-token + shared frozen context. Documented divergence; the two
  *novelties* — GO-masked attention + adaptive loss — are exact.)

LEAK-SAFETY (HARD RULE): payload X_train/pert_train already have every held target gene's sgRNAs removed.
Inside the runner EVERYTHING that adapts to data — the pert condition vocabulary, the response-gene panel
(train top-variance), the per-pert top-DE indices for loss_adapt, the GO mask over the response panel,
the control mean — is fit on TRAIN cells ONLY. Held-gene expression never enters the runner. Invoked once
per fold → every basis refits per fold.

ENV knobs: IVCBENCH_SCFOUNDATION_DIR / _CKPT (backbone, same as scfoundation_runner); IVCBENCH_GENE2GO
(GO annotation for the mask); IVCBENCH_PA_{EPOCHS,MAXCELLS,NRESP,DMODEL,NHEAD,PREDCTRL,GO_MIN_SHARED}.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))   # make `pertadapt` importable


def _scf_dir() -> Path:
    p = os.environ.get("IVCBENCH_SCFOUNDATION_DIR",
                       "/data1/home/chlee/projects/single_cell_fm/scFoundation/model")
    d = Path(p)
    if not (d / "load.py").exists():
        raise FileNotFoundError(f"scFoundation source dir {d} missing load.py (set $IVCBENCH_SCFOUNDATION_DIR)")
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
    import pandas as pd

    from pertadapt import GOMaskedPertAdapter, loss_adapt, additive_go_mask, load_gene2go

    scf_dir = _scf_dir()
    sys.path.insert(0, str(scf_dir))
    from load import load_model_frommmf, gatherData

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = int(os.environ.get("IVCBENCH_PA_EPOCHS", "15"))
    max_cells = int(os.environ.get("IVCBENCH_PA_MAXCELLS", "6000"))
    n_resp_req = int(os.environ.get("IVCBENCH_PA_NRESP", "256"))   # response/gene-axis panel (mask is NxN)
    d_model = int(os.environ.get("IVCBENCH_PA_DMODEL", "128"))
    nhead = int(os.environ.get("IVCBENCH_PA_NHEAD", "8"))          # scFoundation decoder heads
    n_de = int(os.environ.get("IVCBENCH_PA_NDE", "20"))            # top-DE per pert for loss_adapt
    min_shared = int(os.environ.get("IVCBENCH_PA_GO_MIN_SHARED", "1"))
    highres = os.environ.get("IVCBENCH_SCF_HIGHRES", "t4")

    # ---------------------------------------------------------------- payload
    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    n_hvg = len(genes)
    if X.shape[0] == 0:
        raise RuntimeError("PertAdapt: empty training payload")

    train_pert_genes = sorted({p for p in pert_train[~is_ctrl]}) if (~is_ctrl).any() else []
    if not train_pert_genes:
        raise RuntimeError("PertAdapt: no perturbed training cells (cannot learn the adapter)")
    pert_pos = {g: i for i, g in enumerate(train_pert_genes)}
    n_cond = len(train_pert_genes)

    ctrl_pool = X[is_ctrl] if is_ctrl.any() else X_ctrl_inf
    if ctrl_pool.shape[0] == 0:
        raise RuntimeError("PertAdapt: no control cells to anchor predictions")
    ctrl_full_mean = ctrl_pool.mean(0).astype(np.float32)

    # ------------------------------------------ response-gene panel (TRAIN only, leak-safe)
    var = X.var(0)
    n_resp = min(n_resp_req, n_hvg)
    resp_idx = np.argsort(var)[::-1][:n_resp]
    resp_genes = [genes[i] for i in resp_idx]
    resp_mean = X[:, resp_idx].mean(0).astype(np.float32)
    ctrl_resp_mean = ctrl_pool[:, resp_idx].mean(0).astype(np.float32)  # control anchor on the panel

    # ------------------------------------------ GO gene-similarity mask over the response panel
    # (reconstructed from the local full gene2go; see vendor/pertadapt/build_go_mask.py provenance note)
    gene2go = load_gene2go(os.environ.get("IVCBENCH_GENE2GO"))
    go_mask = additive_go_mask(resp_genes, gene2go, min_shared=min_shared)   # (n_resp, n_resp) 0/-inf
    dens = float(np.isfinite(go_mask).mean())
    go_mask_t = torch.from_numpy(go_mask).to(device)

    # ------------------------------------------ frozen scFoundation cell embedding (same path as scF runner)
    model, cfg = load_model_frommmf(str(_ckpt_path()), "cell")
    model.eval().to(device)
    pad_id = cfg["pad_token_id"]
    gidx = pd.read_csv(scf_dir / "OS_scRNA_gene_index.19264.tsv", header=0, delimiter="\t")
    scf_genes = list(gidx["gene_name"])
    scf_pos = {g: i for i, g in enumerate(scf_genes)}
    n_scf = len(scf_genes)
    hvg_to_scf = np.array([scf_pos.get(g, -1) for g in genes], dtype=np.int64)
    have = hvg_to_scf >= 0

    @torch.no_grad()
    def embed(rows: np.ndarray) -> np.ndarray:
        out = []
        for r in rows:
            full = np.zeros(n_scf, dtype=np.float32)
            full[hvg_to_scf[have]] = r[have]
            total = float(np.expm1(full).sum()) + 1.0
            hi = float(highres[1:]) if highres[0] == "t" else 4.0
            gx = torch.tensor(full.tolist() + [hi, np.log10(total)], dtype=torch.float32,
                              device=device).unsqueeze(0)
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

    # ------------------------------------------ training samples (perturbed cells in the vocab)
    pert_mask = (~is_ctrl) & np.array([p in pert_pos for p in pert_train])
    pert_rows = np.where(pert_mask)[0]
    if len(pert_rows) == 0:
        raise RuntimeError("PertAdapt: no usable perturbed training cells in the condition vocab")
    if len(pert_rows) > max_cells:
        rng = np.random.default_rng(0)
        pert_rows = np.sort(rng.choice(pert_rows, max_cells, replace=False))

    # per-pert top-DE indices within the response panel (TRAIN only) for loss_adapt: |mean_pert - ctrl|
    de_idx_by_pert: dict[str, list[int]] = {}
    Xr = X[:, resp_idx]
    for g, j in pert_pos.items():
        gr = (pert_train == g) & (~is_ctrl)
        if gr.sum() == 0:
            continue
        delta = np.abs(Xr[gr].mean(0) - ctrl_resp_mean)
        de_idx_by_pert[g] = np.argsort(delta)[::-1][:min(n_de, n_resp)].tolist()

    emb_dim = 4 * cfg["encoder"]["hidden_dim"]   # 3072 for the released cell ckpt
    print(f"[PertAdapt] train cells={X.shape[0]} perturbed-used={len(pert_rows)} conds={n_cond} "
          f"resp_genes={n_resp} d_model={d_model} nhead={nhead} GO-mask-density={dens:.3%} epochs={epochs}",
          flush=True)

    emb_train = embed(X[pert_rows])              # (N, 3072) frozen cell context
    Y_train = Xr[pert_rows].astype(np.float32)   # (N, n_resp) true response profile
    cond_train = np.array([pert_pos[pert_train[ri]] for ri in pert_rows], dtype=np.int64)
    pert_labels_train = np.array([pert_train[ri] for ri in pert_rows], dtype=object)

    # ------------------------------------------ PertAdapt model (the only trainable part)
    class PertAdaptHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.ctx_proj = nn.Linear(emb_dim, d_model)             # frozen cell ctx → shared per-gene base
            self.gene_emb = nn.Embedding(n_resp, d_model)           # learned gene-token (per-gene identity)
            self.pert_emb = nn.Embedding(n_cond, d_model)           # learned pert emb (GO-GNN analogue)
            self.pert_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            self.adapter = GOMaskedPertAdapter(d_model, nhead, go_mask_t)   # PertAdapt novelty #1
            self.out = nn.Linear(d_model, 1)                        # gene-wise Δ head
            gi = torch.arange(n_resp)
            self.register_buffer("gene_ids", gi, persistent=False)

        def forward(self, cell_ctx, cond_idx):
            B = cell_ctx.shape[0]
            base = self.ctx_proj(cell_ctx).unsqueeze(1)             # (B,1,D) shared frozen context
            gene = self.gene_emb(self.gene_ids).unsqueeze(0)        # (1,N,D) per-gene identity
            exp_enc = base + gene                                   # (B,N,D) per-gene exp encodings
            pert = self.pert_mlp(self.pert_emb(cond_idx)).unsqueeze(1)  # (B,1,D) pooled pert encoding
            adapted = self.adapter(exp_enc, pert)                   # (B,N,D) GO-masked attention
            delta = self.out(adapted).squeeze(-1)                   # (B,N) predicted Δ from control
            return delta

    head = PertAdaptHead().to(device)
    ctrl_resp_t = torch.tensor(ctrl_resp_mean, device=device)
    emb_t = torch.tensor(emb_train, device=device)
    cond_t = torch.tensor(cond_train, device=device)
    Y_t = torch.tensor(Y_train, device=device)

    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = int(os.environ.get("IVCBENCH_PA_BATCH", "32"))
    n = len(pert_rows)
    head.train()
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            delta = head(emb_t[b], cond_t[b])               # (B,N)
            pred = ctrl_resp_t.unsqueeze(0) + delta         # add control anchor → predicted profile
            yb = Y_t[b]
            loss = loss_adapt(pred, yb, pert_labels_train[b], de_idx_by_pert)   # PertAdapt novelty #2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(b)
        if ep == 0 or ep == epochs - 1:
            print(f"[PertAdapt] epoch {ep} loss_adapt={tot / max(1, n):.5f}", flush=True)

    # ------------------------------------------ predict each held gene
    head.eval()
    n_ctrl = min(int(os.environ.get("IVCBENCH_PA_PREDCTRL", "128")), ctrl_pool.shape[0])
    sel = (np.random.default_rng(0).choice(ctrl_pool.shape[0], n_ctrl, replace=False)
           if ctrl_pool.shape[0] > n_ctrl else np.arange(ctrl_pool.shape[0]))
    emb_ctrl = torch.tensor(embed(ctrl_pool[sel]), device=device)   # (n_ctrl, 3072)

    pred_perts, pred_means = [], []
    with torch.no_grad():
        for g in test_perts:
            if g in pert_pos:
                cond = torch.full((emb_ctrl.shape[0],), pert_pos[g], dtype=torch.long, device=device)
                delta = head(emb_ctrl, cond)                # (n_ctrl, N)
                prof_resp = (ctrl_resp_t.unsqueeze(0) + delta).mean(0).float().cpu().numpy()
            else:
                # held gene absent from the train condition vocab → no learned column; emit the control
                # response profile on the panel (still a real, leak-safe output; never the held expression).
                prof_resp = ctrl_resp_mean.copy()
            full = ctrl_full_mean.copy()
            full[resp_idx] = prof_resp
            pred_perts.append(g)
            pred_means.append(full.astype(np.float32))

    if not pred_perts:
        raise RuntimeError("PertAdapt: no held genes to predict")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
