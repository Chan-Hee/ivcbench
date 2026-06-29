#!/usr/bin/env python
"""scGPT runner — executed inside the `scgpt` conda env (scgpt 0.2.4 + torch).

Invoked by ivcbench.baselines.heavy.ScGPT:  <scgpt_python> scgpt_runner.py <in.npz> <out.npz>

Adapts the scGPT perturbation recipe (TransformerGenerator fine-tuned from the pretrained scGPT_human
checkpoint; ref: single_cell_fm/scripts/test_scgpt_perturbation.py) to the ivcbench leak-safe payload.
The model learns: a control cell's expression + a one-hot "perturbed gene" flag → the perturbed
profile. Training uses ONLY the payload's train cells (held genes already removed). Prediction: for
each held target gene, feed control cells with the flag set at that gene's position; the model output
is the predicted perturbed profile, averaged over controls and scattered back into the full HVG panel
(non-modelled genes keep the control mean). Leak-safe: held-gene expression never enters training.

Model dir (vocab.json/config.json/best_model.pt) from $IVCBENCH_SCGPT_MODEL_DIR.
Epochs/seq-len/cell-cap via $IVCBENCH_SCGPT_{EPOCHS,SEQLEN,MAXCELLS}.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def _model_dir() -> Path:
    p = os.environ.get("IVCBENCH_SCGPT_MODEL_DIR")
    if not p:
        raise FileNotFoundError("set $IVCBENCH_SCGPT_MODEL_DIR to the scGPT_human checkpoint directory")
    d = Path(p)
    if not (d / "vocab.json").exists():
        raise FileNotFoundError(f"scGPT model dir {d} missing vocab.json (set $IVCBENCH_SCGPT_MODEL_DIR)")
    return d


def main(in_path: str, out_path: str) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from scgpt.model import TransformerGenerator
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.utils import load_pretrained, set_seed

    set_seed(0)
    md = _model_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # default seqlen 2048 ≥ our 2000-HVG panel → no genes forced to Δ=0 (QC flagged 1536 dropped ~23%)
    seqlen = int(os.environ.get("IVCBENCH_SCGPT_SEQLEN", "2048"))
    epochs = int(os.environ.get("IVCBENCH_SCGPT_EPOCHS", "10"))
    max_cells = int(os.environ.get("IVCBENCH_SCGPT_MAXCELLS", "20000"))

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})

    # ---- vocab + gene selection (genes in vocab; force perturbed genes; top-variance fill) ----
    vocab = GeneVocab.from_file(str(md / "vocab.json"))
    for tok in ["<pad>", "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab["<pad>"])
    gene_ids_all = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=np.int64)
    valid = np.where(gene_ids_all != vocab["<pad>"])[0]
    train_pert_genes = sorted({p for p in pert_train[~is_ctrl]} if (~is_ctrl).any() else set())
    forced = [genes.index(g) for g in train_pert_genes if g in vocab and g in genes]
    var = X[:, valid].var(0)
    ranked = valid[np.argsort(var)[::-1]]
    sel, seen = [], set()
    for i in list(forced) + ranked.tolist():
        if i not in seen:
            sel.append(i); seen.add(i)
        if len(sel) >= seqlen:
            break
    sel = np.array(sel, dtype=np.int64)
    sel_names = [genes[i] for i in sel]
    pos_of = {g: p for p, g in enumerate(sel_names)}
    gene_ids = torch.tensor(gene_ids_all[sel], dtype=torch.long)

    ctrl_pool = X[is_ctrl][:, sel] if is_ctrl.any() else X_ctrl_inf[:, sel]
    if ctrl_pool.shape[0] == 0:
        raise RuntimeError("scGPT: no control cells to anchor predictions")

    class PertDS(Dataset):
        def __init__(self, seed=0):
            self.s = []
            for cond_gene in train_pert_genes:
                if cond_gene not in pos_of:
                    continue
                block = X[(~is_ctrl) & (pert_train == cond_gene)][:, sel].astype(np.float32)
                pp = pos_of[cond_gene]
                for row in block:
                    self.s.append((row, pp))
            if max_cells and len(self.s) > max_cells:
                rng = np.random.default_rng(0)
                self.s = [self.s[i] for i in rng.choice(len(self.s), max_cells, replace=False)]
            self.seed = seed

        def __len__(self):
            return len(self.s)

        def __getitem__(self, i):
            target, pp = self.s[i]
            inp = ctrl_pool[(i + self.seed) % len(ctrl_pool)]
            flags = np.zeros(len(sel), dtype=np.int64); flags[pp] = 1
            return (gene_ids, torch.tensor(inp), torch.tensor(target), torch.tensor(flags))

    def collate(b):
        return (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b]),
                torch.stack([x[2] for x in b]), torch.stack([x[3] for x in b]))

    cfg = json.load(open(md / "config.json"))
    model = TransformerGenerator(
        ntoken=len(vocab), d_model=cfg["embsize"], nhead=cfg["nhead"], d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"], nlayers_cls=3, n_cls=1, vocab=vocab,
        dropout=cfg.get("dropout", 0.1), pad_token="<pad>", pad_value=0, pert_pad_id=2,
        use_fast_transformer=False)
    model = load_pretrained(model, torch.load(md / "best_model.pt", map_location="cpu"), verbose=False)
    model.to(device)

    ds = PertDS()
    if len(ds) == 0:
        raise RuntimeError("scGPT: no trainable perturbations in the selected-gene panel")
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for _ in range(epochs):
        for gid, inp, tgt, fl in loader:
            gid, inp, tgt, fl = gid.to(device), inp.to(device), tgt.to(device), fl.to(device)
            mask = torch.zeros_like(inp, dtype=torch.bool, device=device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(gid, inp, fl, src_key_padding_mask=mask, CLS=False, CCE=False, MVC=False, ECS=False)
                loss = F.mse_loss(out["mlm_output"], tgt)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

    # ---- predict each held gene: control cells + flag at that gene → mean perturbed profile ----
    model.eval()
    n_ctrl = min(256, ctrl_pool.shape[0])
    base = torch.tensor(ctrl_pool[:n_ctrl], dtype=torch.float32, device=device)
    gid_b = gene_ids.to(device).unsqueeze(0).repeat(n_ctrl, 1)
    ctrl_full_mean = (X[is_ctrl].mean(0) if is_ctrl.any() else X_ctrl_inf.mean(0)).astype(np.float32)
    pred_perts, pred_means = [], []
    with torch.no_grad():
        for g in test_perts:
            if g not in pos_of:
                continue
            fl = torch.zeros((n_ctrl, len(sel)), dtype=torch.long, device=device); fl[:, pos_of[g]] = 1
            mask = torch.zeros_like(base, dtype=torch.bool, device=device)
            with torch.amp.autocast("cuda"):
                out = model(gid_b, base, fl, src_key_padding_mask=mask, CLS=False, CCE=False, MVC=False, ECS=False)
            prof_sel = out["mlm_output"].float().mean(0).cpu().numpy()
            full = ctrl_full_mean.copy()
            full[sel] = prof_sel                      # scatter modelled genes; others keep control mean
            pred_perts.append(g); pred_means.append(full.astype(np.float32))

    if not pred_perts:
        raise RuntimeError("scGPT: no held genes were in the modelled panel/vocab")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object),
             pred_means=np.vstack(pred_means).astype(np.float32))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
