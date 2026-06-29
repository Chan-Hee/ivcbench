#!/usr/bin/env python
"""PertAdapt runner (Soskic CD4-activation DONOR-LODO, ②Donor×Hybrid) — `scfoundation` conda env.

Invoked by scripts/pertadapt_soskic.py:  <scfoundation python> pertadapt_soskic_runner.py <in.npz> <out.npz>

DONOR-axis analogue of state_soskic_runner.py, but for the 2nd Hybrid model (PertAdapt). Here the held
UNIT is a *donor*; the perturbation ("stimulation", 0h→16h) is SEEN in every training donor. PertAdapt
learns the control→stim transition conditioned on the lineage (cell_type: CD4 Naive/Memory) using its
GO-masked perturbation adapter + adaptive loss, then predicts the held donor's 16h profile per lineage
from that donor's OWN 0h cells (inference input) — its real 16h cells NEVER enter training (leak-safe).

EMPTY-STRONG parity with the STATE donor row: the perturbation side carries no per-donor info; PertAdapt
transfers the SHARED stim transition through lineage context. The adapter's "perturbation embedding" is a
single learned stim token (the seen perturbation); the lineage is the context that routes the transition.

Payload (built by the driver, identical keys to state_soskic_runner.py):
  X_train, is_control_train, pert_train ('stim_<D>'/'control'), celltype_train, gem_train,
  X_ctrl_inf, celltype_inf, gem_inf, held_label, genes.
Output: pred_perts = ['<held_label>::<lineage>', ...], pred_means = predicted 16h profile per lineage.

LEAK-SAFETY: response panel + GO mask + per-lineage DE indices are fit on TRAINING cells only; the held
donor's 16h expression is never read. Refit per fold (one invocation per held donor).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))


def _scf_dir() -> Path:
    p = os.environ.get("IVCBENCH_SCFOUNDATION_DIR")
    if not p:
        raise FileNotFoundError("set $IVCBENCH_SCFOUNDATION_DIR to the scFoundation source directory")
    if not (Path(p) / "load.py").exists():
        raise FileNotFoundError(f"scFoundation source dir {p} missing load.py")
    return Path(p)


def _ckpt_path() -> Path:
    p = os.environ.get("IVCBENCH_SCFOUNDATION_CKPT")
    if not p:
        raise FileNotFoundError("set $IVCBENCH_SCFOUNDATION_CKPT to the scFoundation checkpoint")
    if not Path(p).exists():
        raise FileNotFoundError(f"scFoundation checkpoint {p} not found")
    return Path(p)


def main(in_path: str, out_path: str) -> None:
    import torch
    import torch.nn as nn
    import pandas as pd

    from pertadapt import GOMaskedPertAdapter, loss_adapt, additive_go_mask, load_gene2go

    scf_dir = _scf_dir()
    sys.path.insert(0, str(scf_dir))
    from load import load_model_frommmf, gatherData

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = int(os.environ.get("IVCBENCH_PA_EPOCHS", "20"))
    max_cells = int(os.environ.get("IVCBENCH_PA_MAXCELLS", "6000"))
    n_resp_req = int(os.environ.get("IVCBENCH_PA_NRESP", "256"))
    d_model = int(os.environ.get("IVCBENCH_PA_DMODEL", "128"))
    nhead = int(os.environ.get("IVCBENCH_PA_NHEAD", "8"))
    n_de = int(os.environ.get("IVCBENCH_PA_NDE", "20"))
    min_shared = int(os.environ.get("IVCBENCH_PA_GO_MIN_SHARED", "1"))
    highres = os.environ.get("IVCBENCH_SCF_HIGHRES", "t4")

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)   # 'stim_<D>' / 'control'
    is_ctrl = d["is_control_train"].astype(bool)
    celltype_train = np.array([str(c) for c in d["celltype_train"]], dtype=object)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)                          # held donor's OWN 0h cells
    celltype_inf = np.array([str(c) for c in d["celltype_inf"]], dtype=object)
    held_label = str(d["held_label"])
    n_hvg = len(genes)
    if X.shape[0] == 0:
        raise RuntimeError("PertAdapt-soskic: empty training payload")

    stim = (~is_ctrl)
    if stim.sum() == 0 or is_ctrl.sum() == 0:
        raise RuntimeError("PertAdapt-soskic: need both 0h (control) and 16h (stim) training cells")

    # response panel + control anchor on the panel — TRAIN only
    var = X.var(0)
    n_resp = min(n_resp_req, n_hvg)
    resp_idx = np.argsort(var)[::-1][:n_resp]
    resp_genes = [genes[i] for i in resp_idx]
    Xr = X[:, resp_idx]
    ctrl_resp_mean = Xr[is_ctrl].mean(0).astype(np.float32)
    ctrl_full_mean = X[is_ctrl].mean(0).astype(np.float32)

    # GO mask over the response panel (reconstructed; see vendor/pertadapt provenance)
    gene2go = load_gene2go(os.environ.get("IVCBENCH_GENE2GO"))
    go_mask_t = torch.from_numpy(additive_go_mask(resp_genes, gene2go, min_shared=min_shared)).to(device)

    # lineage vocabulary (the context that routes the transition) — TRAIN lineages
    lineages = sorted({str(c) for c in celltype_train})
    lin_pos = {l: i for i, l in enumerate(lineages)}
    n_lin = len(lineages)

    # per-lineage top-DE indices for loss_adapt: |mean_stim - mean_ctrl| within each lineage (TRAIN only)
    de_idx_by_lin: dict[str, list[int]] = {}
    for l in lineages:
        sm = stim & (celltype_train == l)
        cm = is_ctrl & (celltype_train == l)
        if sm.sum() == 0 or cm.sum() == 0:
            continue
        delta = np.abs(Xr[sm].mean(0) - Xr[cm].mean(0))
        de_idx_by_lin[l] = np.argsort(delta)[::-1][:min(n_de, n_resp)].tolist()

    # frozen scFoundation cell embedding (same path as the C3 runner)
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

    # training samples: STIM (16h) cells — predict their profile from their frozen ctx + lineage routing
    stim_rows = np.where(stim)[0]
    if len(stim_rows) > max_cells:
        rng = np.random.default_rng(0)
        stim_rows = np.sort(rng.choice(stim_rows, max_cells, replace=False))
    emb_train = embed(X[stim_rows])                       # (N, 3072) frozen ctx (16h cells)
    Y_train = Xr[stim_rows].astype(np.float32)            # (N, n_resp) true 16h profile
    lin_train = np.array([lin_pos.get(str(celltype_train[r]), 0) for r in stim_rows], dtype=np.int64)
    lin_labels_train = np.array([str(celltype_train[r]) for r in stim_rows], dtype=object)

    emb_dim = 4 * cfg["encoder"]["hidden_dim"]
    print(f"[PertAdapt-soskic] {held_label}: train cells={X.shape[0]} stim-used={len(stim_rows)} "
          f"lineages={n_lin} resp={n_resp} d_model={d_model} epochs={epochs}", flush=True)

    class PertAdaptHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.ctx_proj = nn.Linear(emb_dim, d_model)
            self.gene_emb = nn.Embedding(n_resp, d_model)
            self.stim_emb = nn.Parameter(torch.zeros(d_model))      # single SEEN stim token
            self.lin_emb = nn.Embedding(max(1, n_lin), d_model)     # lineage context
            self.fuse = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
            self.adapter = GOMaskedPertAdapter(d_model, nhead, go_mask_t)
            self.out = nn.Linear(d_model, 1)
            self.register_buffer("gene_ids", torch.arange(n_resp), persistent=False)

        def forward(self, cell_ctx, lin_idx):
            B = cell_ctx.shape[0]
            base = self.ctx_proj(cell_ctx).unsqueeze(1)             # (B,1,D)
            gene = self.gene_emb(self.gene_ids).unsqueeze(0)        # (1,N,D)
            exp_enc = base + gene
            pert = self.fuse(self.stim_emb.unsqueeze(0) + self.lin_emb(lin_idx)).unsqueeze(1)  # (B,1,D)
            adapted = self.adapter(exp_enc, pert)                   # GO-masked attention
            return self.out(adapted).squeeze(-1)                    # (B,N) Δ from control

    head = PertAdaptHead().to(device)
    ctrl_resp_t = torch.tensor(ctrl_resp_mean, device=device)
    emb_t = torch.tensor(emb_train, device=device)
    lin_t = torch.tensor(lin_train, device=device)
    Y_t = torch.tensor(Y_train, device=device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = int(os.environ.get("IVCBENCH_PA_BATCH", "32"))
    n = len(stim_rows)
    head.train()
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            pred = ctrl_resp_t.unsqueeze(0) + head(emb_t[b], lin_t[b])
            loss = loss_adapt(pred, Y_t[b], lin_labels_train[b], de_idx_by_lin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(b)
        if ep == 0 or ep == epochs - 1:
            print(f"[PertAdapt-soskic] epoch {ep} loss_adapt={tot / max(1, n):.5f}", flush=True)

    # predict held donor's 16h profile per lineage from its OWN 0h cells
    head.eval()
    pred_perts, pred_means = [], []
    inf_lineages = sorted({str(c) for c in celltype_inf})
    with torch.no_grad():
        for lin in inf_lineages:
            sel = np.where(celltype_inf == lin)[0]
            if len(sel) == 0:
                continue
            cap = min(int(os.environ.get("IVCBENCH_PA_PREDCTRL", "128")), len(sel))
            if len(sel) > cap:
                sel = np.random.default_rng(0).choice(sel, cap, replace=False)
            emb_c = torch.tensor(embed(X_ctrl_inf[sel]), device=device)
            li = lin_pos.get(lin, 0)
            lin_idx = torch.full((emb_c.shape[0],), li, dtype=torch.long, device=device)
            prof_resp = (ctrl_resp_t.unsqueeze(0) + head(emb_c, lin_idx)).mean(0).float().cpu().numpy()
            full = ctrl_full_mean.copy()
            full[resp_idx] = prof_resp
            pred_perts.append(f"{held_label}::{lin}")
            pred_means.append(full.astype(np.float32))

    if not pred_perts:
        raise RuntimeError("PertAdapt-soskic: no held-donor lineages to predict")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
