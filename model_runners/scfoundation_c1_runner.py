#!/usr/bin/env python
"""scFoundation runner for the CELL-CONTEXT tasks (C1 Kang LOCT / C5 LOCT) — `scfoundation` conda env.

Invoked by ivcbench.baselines.heavy.ScFoundationC1:
    <env python> scfoundation_c1_runner.py <in.npz> <out.npz>

Why a separate runner. The unseen-gene runner conditions on a one-hot perturbed-gene token, which is the
axis those models are published for. On a cell-context task the perturbation is SEEN and the held axis is
the cell type, so that interface degenerates. Here we keep scFoundation in exactly the regime it is
published for downstream use — a FROZEN encoder — and put the task in the head:

  1. frozen 19264-gene `cell` encoder -> 3072-d embedding for training cells and for the held unit's own
     control cells (identical embedding path to scfoundation_runner.py: get_embedding.py 'cell'/'all');
  2. latent shift delta = mean(emb[stimulated train]) - mean(emb[control train]), estimated on the
     TRAINING units only;
  3. a small MLP decoder head (the only trainable part) maps frozen embedding -> response-gene profile,
     trained on the training fold;
  4. prediction for the held unit: decode(emb(its own control cells) + delta), averaged over those cells.

This is the same latent-shift arithmetic as the scGen cell-context baseline, applied to the foundation
model's frozen representation, so the comparison isolates the value of that representation.

Leak-safety: the held unit's stimulated cells never enter training; only its control cells are input.
Epochs / cell cap / response panel via $IVCBENCH_SCF_{EPOCHS,MAXCELLS,NRESP}.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scfoundation_runner import _scf_dir, _ckpt_path  # noqa: E402  (same env discovery)


def main(in_path: str, out_path: str) -> None:
    import torch
    import torch.nn as nn
    import pandas as pd

    scf_dir = _scf_dir()
    sys.path.insert(0, str(scf_dir))
    from load import load_model_frommmf, gatherData

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = int(os.environ.get("IVCBENCH_SCF_EPOCHS", "15"))
    max_cells = int(os.environ.get("IVCBENCH_SCF_MAXCELLS", "4000"))
    n_resp_env = int(os.environ.get("IVCBENCH_SCF_NRESP", "512"))
    highres = os.environ.get("IVCBENCH_SCF_HIGHRES", "t4")

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if not test_perts:
        raise RuntimeError("scFoundation-C1: no held perturbation label in the payload")
    stim_label = test_perts[0]
    if not is_ctrl.any() or not (~is_ctrl).any():
        raise RuntimeError(
            "scFoundation-C1: training fold needs both control and stimulated cells"
        )

    model, cfg = load_model_frommmf(str(_ckpt_path()), "cell")
    model.eval().to(device)
    pad_id = cfg["pad_token_id"]
    gidx = pd.read_csv(
        scf_dir / "OS_scRNA_gene_index.19264.tsv", header=0, delimiter="\t"
    )
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
            gx = torch.tensor(
                full.tolist() + [hi, np.log10(total)],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            gids = torch.arange(n_scf + 2, device=device).repeat(gx.shape[0], 1)
            vl = gx > 0
            xx, xpad = gatherData(gx, vl, pad_id)
            pos, _ = gatherData(gids, vl, pad_id)
            e = model.token_emb(torch.unsqueeze(xx, 2).float(), output_weight=0)
            e = e + model.pos_emb(pos)
            g = model.encoder(e, xpad)
            emb = torch.concat(
                [
                    g[:, -1, :],
                    g[:, -2, :],
                    torch.max(g[:, :-2, :], dim=1)[0],
                    torch.mean(g[:, :-2, :], dim=1),
                ],
                axis=1,
            )
            out.append(emb.float().cpu().numpy()[0])
        return np.vstack(out).astype(np.float32)

    rng = np.random.default_rng(0)

    def cap(idx):
        return (
            np.sort(rng.choice(idx, max_cells, replace=False))
            if len(idx) > max_cells
            else idx
        )

    ctrl_rows, stim_rows = cap(np.where(is_ctrl)[0]), cap(np.where(~is_ctrl)[0])
    inf_rows = cap(np.arange(X_ctrl_inf.shape[0]))

    var = X.var(0)
    n_resp = min(n_resp_env, len(genes))
    resp_idx = np.argsort(var)[::-1][:n_resp]

    print(
        f"[scFoundation-C1] train ctrl={len(ctrl_rows)} stim={len(stim_rows)} "
        f"held-ctrl={len(inf_rows)} resp={n_resp} epochs={epochs}",
        flush=True,
    )

    E_ctrl, E_stim = embed(X[ctrl_rows]), embed(X[stim_rows])
    E_inf = embed(X_ctrl_inf[inf_rows])
    delta = E_stim.mean(0) - E_ctrl.mean(0)  # latent shift, training units only

    E_all = np.vstack([E_ctrl, E_stim])
    Y_all = np.vstack([X[ctrl_rows][:, resp_idx], X[stim_rows][:, resp_idx]])
    mu, sd = E_all.mean(0), E_all.std(0) + 1e-6
    head = nn.Sequential(
        nn.Linear(E_all.shape[1], 512), nn.ReLU(), nn.Linear(512, n_resp)
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Et = torch.tensor((E_all - mu) / sd, device=device)
    Yt = torch.tensor(Y_all, device=device)
    head.train()
    for ep in range(epochs):
        perm = torch.randperm(Et.shape[0], device=device)
        tot = 0.0
        for i in range(0, len(perm), 256):
            b = perm[i : i + 256]
            opt.zero_grad()
            loss = nn.functional.mse_loss(head(Et[b]), Yt[b])
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if ep in (0, epochs - 1):
            print(
                f"  [head] epoch {ep + 1}/{epochs} mse={tot / len(perm):.5f}",
                flush=True,
            )
    head.eval()

    with torch.no_grad():
        Ep = torch.tensor((E_inf + delta - mu) / sd, device=device, dtype=torch.float32)
        pred_resp = head(Ep).cpu().numpy().mean(0)

    profile = (
        X_ctrl_inf.mean(0).astype(np.float32).copy()
    )  # non-modelled genes keep the control mean
    profile[resp_idx] = pred_resp.astype(np.float32)
    np.savez(
        out_path,
        pred_perts=np.array([stim_label], dtype=object),
        pred_means=profile[None, :].astype(np.float32),
    )
    print(f"[scFoundation-C1] wrote prediction for '{stim_label}'", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
