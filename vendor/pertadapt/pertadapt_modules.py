"""PertAdapt novelty modules — vendored & adapted from Bai et al. 2025 (github.com/BaiDing1234/PertAdapt,
upstream commit in ../UPSTREAM_COMMIT.txt). These are the TWO published contributions, lifted out of the
heavy DDP/GEARS training stack so they can run inside the benchmark's leak-safe npz runner pattern:

  1) GOMaskedPertAdapter  — the *condition-sensitive perturbation adapter* with a gene-similarity–masked
     multi-head self-attention (upstream `gears/model_new.py::PertAdapterNew`). It takes per-gene
     embeddings (exp_encodings: B,N,D) + a pooled perturbation embedding (pert_encodings: B,1,D),
     adds+norms them, runs a self-attention whose attention mask is the GO gene-similarity adjacency
     (-inf where genes are NOT GO-connected → attention restricted to GO neighbours), and a residual FFN.

  2) loss_adapt  — the *adaptive loss* (upstream `gears/utils.py::loss_adapt`) that balances the DE-gene
     MSE and the all-gene MSE by their *relative magnitudes* each step:
         w_de  = (mse_de + mse_all) / mse_de ;  w_mse = 2  → loss = w_de*mse_de + w_mse*mse_all
     This up-weights the (small, hard) set of perturbation-sensitive DE genes against the (large, easy)
     insensitive background — the imbalance the paper targets.

NOT vendored (intentionally): the full scFoundation MAEAutobin encoder wrapper, the GEARS GO-GNN, the
DDP/wandb train loop. The runner reuses the already-working frozen scFoundation backbone (the same
`load.py` path scfoundation_runner.py uses) for `exp_encodings`, so we don't re-vendor the fragile
backbone stack. The math of the two contributions is preserved verbatim.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, MultiheadAttention


class GOMaskedPertAdapter(nn.Module):
    """Gene-similarity–masked perturbation adapter (upstream PertAdapterNew, verbatim math).

    Args:
        d_model: per-gene embedding dim (scFoundation hidden = 768 for the released `cell` ckpt; or a
                 projected dim — pass whatever the runner uses for exp/pert encodings).
        nhead:   attention heads (paper uses the scFoundation decoder head count = 8).
        go_mask: (N, N) float32 ADDITIVE attention mask over the N modelled genes: 0.0 where genes are
                 GO-connected (attention allowed), -inf elsewhere. Diagonal is forced to 0 (self always
                 allowed). This is the gene-similarity mask; in the paper it is `go_mask_19264.npz`
                 restricted to the modelled gene panel.
    """

    def __init__(self, d_model: int, nhead: int, go_mask: torch.Tensor):
        super().__init__()
        self.mha = MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = Linear(d_model, d_model // 2)
        self.linear2 = Linear(d_model // 2, d_model)
        self.activation = F.relu
        # additive mask buffer (moves with .to(device); never trained) — upstream forces diagonal allow
        m = go_mask.clone().float()
        for i in range(m.shape[0]):
            if not torch.isfinite(m[i, i]) or m[i, i] != 0:
                m[i, i] = 0.0
        self.register_buffer("attn_mask", m, persistent=False)

    def forward(self, exp_encodings: torch.Tensor, pert_encodings: torch.Tensor) -> torch.Tensor:
        # B,N,D  +  B,1,D  (broadcast pooled pert emb onto every gene) — upstream PertAdapterNew.forward
        x_in = exp_encodings + pert_encodings
        x_in = self.norm0(x_in)
        x = self.mha(query=x_in, key=x_in, value=x_in,
                     attn_mask=self.attn_mask, need_weights=False)[0]
        x = self.norm1(x + x_in)
        x = self.norm2(x + self.linear2(self.activation(self.linear1(x))))
        return x


def build_additive_go_mask(go_adj: np.ndarray) -> torch.Tensor:
    """Convert a (N,N) GO gene-similarity adjacency (nonzero = connected) into the ADDITIVE attention
    mask the adapter consumes (0.0 where connected, -inf elsewhere), exactly as upstream PertAdapterNew
    does on `go_mask_19264.npz`:  diag→1, then where(>0, 0.0, -inf)."""
    a = np.asarray(go_adj, dtype=np.float32).copy()
    n = a.shape[0]
    for i in range(n):
        if a[i, i] < 1:
            a[i, i] = 1.0
    m = np.where(a > 0, 0.0, -np.inf).astype(np.float32)
    return torch.from_numpy(m)


def loss_adapt(pred: torch.Tensor, y: torch.Tensor, perts, de_idx_by_pert: dict,
               retain_idx_by_pert: dict | None = None) -> torch.Tensor:
    """Adaptive DE-reweighted loss — upstream gears/utils.py::loss_adapt, re-expressed for the runner's
    (B, N) tensors with explicit per-perturbation DE-gene index lists (no GEARS adata dependency).

    pred, y : (B, N) predicted / true post-perturbation profiles over the N modelled genes.
    perts   : length-B array of perturbation labels per row ('ctrl' for controls).
    de_idx_by_pert    : {pert_label: list[int]} top-DE gene indices (≤20) within the N panel, TRAIN-fold.
    retain_idx_by_pert: {pert_label: list[int]} non-all-zero gene indices to score the all-gene MSE on
                        (upstream `dict_filter`); defaults to all genes when not supplied.

    Reweighting (verbatim): after computing mean DE-MSE and all-gene-MSE across perturbations,
        w_de = (mse_de + mse_all)/mse_de  (if mse_de>0 else 0);  w_mse = 2 (if mse_all>0 else 0)
        loss = w_de*mse_de + w_mse*mse_all.
    """
    mse = nn.MSELoss()
    perts = np.asarray(perts)
    dev = pred.device
    losses_de = torch.zeros((), device=dev, requires_grad=True)
    losses_mse = torch.zeros((), device=dev, requires_grad=True)
    uniq = list(dict.fromkeys(perts.tolist()))
    for pert in uniq:
        rows = np.where(perts == pert)[0]
        if pert != "ctrl" and pert in de_idx_by_pert and len(de_idx_by_pert[pert]) > 0:
            de_idx = de_idx_by_pert[pert]
            retain = (retain_idx_by_pert or {}).get(pert)
            pred_de, y_de = pred[rows][:, de_idx], y[rows][:, de_idx]
            if retain is not None and len(retain) > 0:
                pred_all, y_all = pred[rows][:, retain], y[rows][:, retain]
            else:
                pred_all, y_all = pred[rows], y[rows]
            losses_de = losses_de + mse(pred_de, y_de)
            losses_mse = losses_mse + mse(pred_all, y_all)
        else:
            losses_mse = losses_mse + mse(pred[rows], y[rows])
    n = max(1, len(uniq))
    losses_de = losses_de / n
    losses_mse = losses_mse / n
    de_v, mse_v = losses_de.detach(), losses_mse.detach()
    de_w = ((de_v + mse_v) / de_v) if de_v > 0 else torch.zeros((), device=dev)
    mse_w = (2.0 * torch.ones((), device=dev)) if mse_v > 0 else torch.zeros((), device=dev)
    return de_w * losses_de + mse_w * losses_mse
