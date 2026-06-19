#!/usr/bin/env python
"""Run the REAL CellOT (Bunne et al. 2023, bunnech/cellot @ ff28778) on the benchmark's
Kang IFN-beta LOLO and Soskic CD4-activation LODO splits.

Faithful to the official scPerturBench invocation (official_snapshot/manuscript1/ood/myCellot.py):
  * train an scgen autoencoder (cellot.models.ae.AutoEncoder) on all NON-held cells -> 50-dim latent
  * train CellOT f/g ICNN potentials (cellot.networks.icnns.ICNN) IN that latent space; source=control,
    target=treated, both restricted to NON-held cells (held group never trained on)
  * predict: encode held group's OWN control cells -> push through g.transport -> decode to gene space
    (exactly the official `ae` embedding / `data_space` prediction path in cellot.utils.evaluate)

The SPLIT and DATA come from the repo (build_split + audit_split), so this is the identical leak-safe
split the manuscript uses; metrics are the repo implementations and same orientation.
"""
from __future__ import annotations
import os, sys, time, json, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split
from ivcbench.metrics.response import pearson_delta
from ivcbench.metrics.distribution import e_distance
from ivcbench.metrics.program import aucell

from cellot.networks.icnns import ICNN
from cellot.models.cellot import compute_loss_f, compute_loss_g
from cellot.models.ae import AutoEncoder
from cellot.losses.mmd import compute_scalar_mmd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IFN_GENES = ["ISG15", "IFI6", "MX1", "MX2", "OAS1", "OAS2", "IFIT1", "IFIT3", "ISG20",
             "STAT1", "IRF7", "IFI44", "IFI44L", "RSAD2", "USP18"]
SOSKIC_PROGRAMS = {
    "T_cell_activation": ["CD69", "IL2RA", "CD40LG", "TNFRSF9", "NR4A1", "NR4A2", "NR4A3",
                          "EGR1", "EGR2", "IRF4", "REL", "NFKBIA", "CD28", "TNFRSF4"],
    "IL2_STAT5": ["IL2RA", "IL2RB", "IL2RG", "STAT5A", "STAT5B", "CISH", "SOCS1", "SOCS3",
                  "BCL2", "MYC", "IL2"],
    "type_I_IFN": ["ISG15", "IFI6", "MX1", "MX2", "OAS1", "OAS2", "OAS3", "OASL", "IFIT1",
                   "IFIT2", "IFIT3", "IFITM1", "IFITM3", "ISG20", "IRF7", "STAT1", "STAT2",
                   "RSAD2", "USP18", "IFI44", "IFI44L", "BST2", "XAF1", "HERC5", "LY6E"],
    "type_II_IFN": ["IFNG", "STAT1", "IRF1", "CXCL9", "CXCL10", "CXCL11", "GBP1", "GBP2",
                    "GBP5", "TAP1", "PSMB8", "PSMB9", "HLA-DRA", "HLA-DRB1", "SOCS1"],
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---- scgen AutoEncoder (configs/models/scgen.yaml: hidden [512,512], lat 50, beta 0, lr 1e-3) -----
def build_ae(input_dim):
    return AutoEncoder(input_dim=input_dim, latent_dim=50, hidden_units=[512, 512],
                       beta=0.0, dropout=0.0).to(DEVICE)


def train_ae(ae, X_train, n_iters, batch_size=256, lr=1e-3, weight_decay=1e-5):
    opt = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100000, gamma=0.5)
    Xt = torch.tensor(np.asarray(X_train, np.float32), device=DEVICE)
    n = Xt.shape[0]
    ae.train()
    for _ in range(n_iters):
        idx = torch.randint(0, n, (min(batch_size, n),), device=DEVICE)
        opt.zero_grad()
        loss, _, _ = ae(Xt[idx])
        loss = loss.mean()
        if torch.isnan(loss):
            raise ValueError("AE loss NaN")
        loss.backward(); opt.step(); sched.step()
    return ae


@torch.no_grad()
def ae_encode(ae, X):
    ae.eval()
    return ae.encode(torch.tensor(np.asarray(X, np.float32), device=DEVICE)).cpu().numpy()


@torch.no_grad()
def ae_decode(ae, Z):
    ae.eval()
    return ae.decode(torch.tensor(np.asarray(Z, np.float32), device=DEVICE)).cpu().numpy()


# ---- CellOT f/g ICNN (configs/models/cellot.yaml: hidden [64]*4, lr 1e-4 b(.5,.9), g.fnorm 1) -----
def build_fg(latent_dim):
    common = dict(input_dim=latent_dim, hidden_units=[64, 64, 64, 64], activation="LeakyReLU",
                  softplus_W_kernels=False,
                  kernel_init_fxn=lambda w: torch.nn.init.uniform_(w, b=0.1))
    f = ICNN(fnorm_penalty=0, **common).to(DEVICE)
    g = ICNN(fnorm_penalty=1, **common).to(DEVICE)
    return f, g


def _sampler(Z, batch_size, seed):
    rng = np.random.default_rng(seed)
    Zt = torch.tensor(np.asarray(Z, np.float32), device=DEVICE)
    n = Zt.shape[0]; bs = min(batch_size, n)
    while True:
        idx = torch.as_tensor(rng.choice(n, bs, replace=(n < bs)), device=DEVICE)
        yield Zt[idx]


def train_cellot_latent(f, g, Zsrc, Ztgt, n_iters, n_inner=10, batch_size=64,
                        lr=1e-4, betas=(0.5, 0.9), eval_every=250):
    opt_f = torch.optim.Adam(f.parameters(), lr=lr, betas=betas)
    opt_g = torch.optim.Adam(g.parameters(), lr=lr, betas=betas)
    src_it, tgt_it = _sampler(Zsrc, batch_size, 1), _sampler(Ztgt, batch_size, 2)
    src_ev, tgt_ev = _sampler(Zsrc, batch_size, 11), _sampler(Ztgt, batch_size, 12)
    best_mmd, best_state = np.inf, None
    for step in range(n_iters):
        target = next(tgt_it)
        for _ in range(n_inner):
            source = next(src_it).requires_grad_(True)
            opt_g.zero_grad()
            gl = compute_loss_g(f, g, source).mean()
            if (not g.softplus_W_kernels) and g.fnorm_penalty > 0:
                gl = gl + g.penalize_w()
            gl.backward(); opt_g.step()
        source = next(src_it).requires_grad_(True)
        opt_f.zero_grad()
        fl = compute_loss_f(f, g, source, target).mean()
        fl.backward(); opt_f.step()
        if torch.isnan(gl) or torch.isnan(fl):
            raise ValueError("CellOT loss NaN")
        f.clamp_w()
        if step % eval_every == 0:
            s = next(src_ev).requires_grad_(True); t = next(tgt_ev)
            transport = g.transport(s).detach()
            mmd = compute_scalar_mmd(t.detach().cpu().numpy(), transport.cpu().numpy())
            if mmd < best_mmd:
                best_mmd = mmd
                best_state = ({k: v.detach().cpu().clone() for k, v in f.state_dict().items()},
                              {k: v.detach().cpu().clone() for k, v in g.state_dict().items()})
    if best_state is not None:
        f.load_state_dict(best_state[0]); g.load_state_dict(best_state[1])
    return f, g, float(best_mmd)


def transport_latent(g, Zsrc):
    g.eval()
    Zt = torch.tensor(np.asarray(Zsrc, np.float32), device=DEVICE).requires_grad_(True)
    return g.transport(Zt).detach().cpu().numpy()


def run_cellot_on_split(cs, sp, seed, ae_iters, cellot_iters):
    """Returns (pred_genes for held control cells, control strata, best latent MMD).
    Faithful ae-embedding / data_space prediction. pred_genes[i] is the CellOT-predicted treated
    profile of the i-th held control cell (same donor/lineage stratum)."""
    set_seed(seed)
    X = cs.X
    tr = sp.train_idx
    is_ctrl_tr = cs.obs.iloc[tr]["is_control"].to_numpy().astype(bool)
    Xtr_ctrl, Xtr_treat, Xtr_all = X[tr[is_ctrl_tr]], X[tr[~is_ctrl_tr]], X[tr]
    Xheld_ctrl = X[sp.inference_input_idx]

    ae = train_ae(build_ae(X.shape[1]), Xtr_all, n_iters=ae_iters)
    Zsrc, Ztgt = ae_encode(ae, Xtr_ctrl), ae_encode(ae, Xtr_treat)
    f, g = build_fg(latent_dim=Zsrc.shape[1])
    f, g, best_mmd = train_cellot_latent(f, g, Zsrc, Ztgt, n_iters=cellot_iters)
    Zpush = transport_latent(g, ae_encode(ae, Xheld_ctrl))
    pred_genes = ae_decode(ae, Zpush)
    return pred_genes, best_mmd


def stratum_align(pred_cells, pred_strata, test_strata):
    """Tile per-stratum predicted cloud to match the test-row count per stratum (for pearson_delta,
    whose per-stratum mean is invariant to tiling). For E-distance we pass the raw clouds instead."""
    pred_strata, test_strata = np.asarray(pred_strata), np.asarray(test_strata)
    aligned = np.zeros((len(test_strata), pred_cells.shape[1]), np.float32)
    for s in np.unique(test_strata):
        mt, mp = test_strata == s, pred_strata == s
        block = pred_cells[mp] if mp.sum() else pred_cells
        reps = int(np.ceil(mt.sum() / len(block)))
        aligned[mt] = np.tile(block, (reps, 1))[: mt.sum()]
    return aligned


def edist_clouds(pred_cells, pred_strata, test_cells, test_strata, fit_on, n_pca=50):
    """E-distance per stratum on the genuine pushed cloud vs observed cloud, macro-averaged.
    PCA basis = training cells (leak-safe, same as repo)."""
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import cdist
    pred_strata, test_strata = np.asarray(pred_strata), np.asarray(test_strata)
    k = int(min(n_pca, fit_on.shape[0] - 1, fit_on.shape[1]))
    pca = PCA(n_components=max(2, k), random_state=0).fit(fit_on)
    per = []
    for s in np.unique(test_strata):
        mt, mp = test_strata == s, pred_strata == s
        if mt.sum() < 2 or mp.sum() < 2:
            continue
        P, T = pca.transform(pred_cells[mp]), pca.transform(test_cells[mt])
        d = 2 * cdist(P, T).mean() - cdist(P, P).mean() - cdist(T, T).mean()
        per.append(float(d))
    return float(np.mean(per)) if per else float("nan")


if __name__ == "__main__":
    print("module ok; device", DEVICE)
