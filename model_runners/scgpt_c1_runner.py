#!/usr/bin/env python
"""scGPT runner for the CELL-CONTEXT task (C1 Kang LOCT) — `scgpt` conda env.

Invoked by ivcbench.baselines.heavy.ScGPTC1:
    <env python> scgpt_c1_runner.py <in.npz> <out.npz>

Why a separate runner. The unseen-gene runner conditions on a one-hot perturbed-gene flag, which is the
interface scGPT publishes for CRISPR perturbation. On a cell-context task the perturbation is a cytokine
that is SEEN in training and the held axis is the cell type, so there is no held gene to flag. We keep
scGPT in the regime it is published for — the TransformerGenerator fine-tuned end-to-end from the
pretrained checkpoint, exactly as on the unseen-gene axis — and change only the conditioning: the
stimulus enters as a single global condition flag, and the held unit's identity is carried by its own
control cells, which are the inference input.

Leak-safe: the held unit's stimulated cells never enter training. Epochs / sequence length / cell cap
via $IVCBENCH_SCGPT_{EPOCHS,SEQLEN,MAXCELLS}.

EPOCH-BUDGET INSTRUMENTATION (all opt-in; unset -> the deposited path runs unchanged).
  $IVCBENCH_SCGPT_TRACE       append one JSON object per epoch (mean training MSE) to this file.
  $IVCBENCH_SCGPT_EVAL_EPOCHS comma-separated epoch numbers at which to ALSO emit a prediction,
                              written to $IVCBENCH_SCGPT_CKPT_DIR/$IVCBENCH_SCGPT_CKPT_TAG__ep<N>.npz
                              with the same schema as <out.npz>.
Neither changes the optimisation. The loss is read with .detach(); the checkpoint prediction runs
under model.eval() + no_grad, so no dropout is sampled and the RNG stream is untouched -- epochs
1..k of a 20-epoch run therefore see the same batch order as epochs 1..k of a k-epoch run. That is a
checkable invariant, not an assumption: the ep10 checkpoint must reproduce the deposited 10-epoch
score for the same unit.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scgpt_runner import _model_dir  # noqa: E402  (same checkpoint discovery)


def main(in_path: str, out_path: str) -> None:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from scgpt.model import TransformerGenerator
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.utils import load_pretrained, set_seed
    import json

    set_seed(0)
    md = _model_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqlen = int(os.environ.get("IVCBENCH_SCGPT_SEQLEN", "2048"))
    epochs = int(os.environ.get("IVCBENCH_SCGPT_EPOCHS", "10"))
    max_cells = int(os.environ.get("IVCBENCH_SCGPT_MAXCELLS", "8000"))
    trace_path = os.environ.get("IVCBENCH_SCGPT_TRACE", "")
    ckpt_dir = os.environ.get("IVCBENCH_SCGPT_CKPT_DIR", "")
    ckpt_tag = os.environ.get("IVCBENCH_SCGPT_CKPT_TAG", "")
    eval_epochs = sorted(
        {
            int(x)
            for x in os.environ.get("IVCBENCH_SCGPT_EVAL_EPOCHS", "").replace(",", " ").split()
        }
    )
    if eval_epochs and not (ckpt_dir and ckpt_tag):
        raise SystemExit(
            "IVCBENCH_SCGPT_EVAL_EPOCHS needs IVCBENCH_SCGPT_CKPT_DIR and "
            "IVCBENCH_SCGPT_CKPT_TAG (the runner's own out.npz lives in a tempdir that the "
            "caller deletes, so checkpoints must be written somewhere durable)"
        )

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if not test_perts:
        raise RuntimeError("scGPT-C1: no held perturbation label in the payload")
    stim_label = test_perts[0]
    if not is_ctrl.any() or not (~is_ctrl).any():
        raise RuntimeError(
            "scGPT-C1: training fold needs both control and stimulated cells"
        )

    vocab = GeneVocab.from_file(str(md / "vocab.json"))
    for tok in ["<pad>", "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab["<pad>"])
    gene_ids_all = np.array(
        [vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=np.int64
    )
    valid = np.where(gene_ids_all != vocab["<pad>"])[0]
    ranked = valid[np.argsort(X[:, valid].var(0))[::-1]]
    sel = np.array(ranked[:seqlen].tolist(), dtype=np.int64)
    gene_ids = torch.tensor(gene_ids_all[sel], dtype=torch.long)

    ctrl_tr = X[is_ctrl][:, sel]
    stim_tr = X[~is_ctrl][:, sel]
    rng = np.random.default_rng(0)
    if stim_tr.shape[0] > max_cells:
        stim_tr = stim_tr[
            np.sort(rng.choice(stim_tr.shape[0], max_cells, replace=False))
        ]
    print(
        f"[scGPT-C1] train ctrl={ctrl_tr.shape[0]} stim={stim_tr.shape[0]} "
        f"held-ctrl={X_ctrl_inf.shape[0]} genes={len(sel)} epochs={epochs}",
        flush=True,
    )

    class StimDS(Dataset):
        """Pair each stimulated training cell with a control cell; the condition flag is global."""

        def __len__(self):
            return stim_tr.shape[0]

        def __getitem__(self, i):
            inp = ctrl_tr[i % ctrl_tr.shape[0]]
            flags = np.ones(
                len(sel), dtype=np.int64
            )  # single seen condition, applied globally
            return (
                gene_ids,
                torch.tensor(inp),
                torch.tensor(stim_tr[i]),
                torch.tensor(flags),
            )

    def collate(b):
        return (
            torch.stack([x[0] for x in b]),
            torch.stack([x[1] for x in b]),
            torch.stack([x[2] for x in b]),
            torch.stack([x[3] for x in b]),
        )

    cfg = json.load(open(md / "config.json"))
    model = TransformerGenerator(
        ntoken=len(vocab),
        d_model=cfg["embsize"],
        nhead=cfg["nhead"],
        d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"],
        nlayers_cls=3,
        n_cls=1,
        vocab=vocab,
        dropout=cfg.get("dropout", 0.1),
        pad_token="<pad>",
        pad_value=0,
        pert_pad_id=2,
        use_fast_transformer=False,
    )
    model = load_pretrained(
        model, torch.load(md / "best_model.pt", map_location="cpu"), verbose=False
    )
    model.to(device)

    loader = DataLoader(
        StimDS(), batch_size=8, shuffle=True, num_workers=0, collate_fn=collate
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    def _write_prediction(path: str) -> None:
        """Predict the held unit's stimulated profile from ITS OWN control cells, and save it.

        Called once at the end of training, and additionally at each $IVCBENCH_SCGPT_EVAL_EPOCHS
        checkpoint. Restores train mode on the way out so a mid-training call is transparent.
        """
        was_training = model.training
        model.eval()
        n_ctrl = min(256, X_ctrl_inf.shape[0])
        base = torch.tensor(
            X_ctrl_inf[:n_ctrl][:, sel], dtype=torch.float32, device=device
        )
        gid_b = gene_ids.to(device).unsqueeze(0).repeat(n_ctrl, 1)
        fl = torch.ones((n_ctrl, len(sel)), dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = model(
                gid_b,
                base,
                fl,
                src_key_padding_mask=torch.zeros_like(base, dtype=torch.bool),
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
            )
        prof_sel = out["mlm_output"].float().mean(0).cpu().numpy()
        full = X_ctrl_inf.mean(0).astype(np.float32).copy()
        full[sel] = prof_sel
        np.savez(
            path,
            pred_perts=np.array([stim_label], dtype=object),
            pred_means=full[None, :].astype(np.float32),
        )
        if was_training:
            model.train()

    model.train()
    for ep in range(epochs):
        tot = n = 0.0
        for gid, inp, tgt, fl in loader:
            gid, inp, tgt, fl = (
                gid.to(device),
                inp.to(device),
                tgt.to(device),
                fl.to(device),
            )
            mask = torch.zeros_like(inp, dtype=torch.bool, device=device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(
                    gid,
                    inp,
                    fl,
                    src_key_padding_mask=mask,
                    CLS=False,
                    CCE=False,
                    MVC=False,
                    ECS=False,
                )
                loss = F.mse_loss(out["mlm_output"], tgt)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * inp.shape[0]
            n += inp.shape[0]
        mse = tot / max(n, 1)
        if ep in (0, epochs - 1) or trace_path or eval_epochs:
            print(f"  [scGPT-C1] epoch {ep+1}/{epochs} mse={mse:.5f}", flush=True)
        if trace_path:  # append-only, one object per epoch: a killed run still leaves a curve
            with open(trace_path, "a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "tag": ckpt_tag,
                            "epoch": ep + 1,
                            "epochs": epochs,
                            "train_mse": mse,
                            "n_train_cells": int(n),
                        }
                    )
                    + "\n"
                )
        if (ep + 1) in eval_epochs:
            ck = os.path.join(ckpt_dir, f"{ckpt_tag}__ep{ep + 1}.npz")
            os.makedirs(ckpt_dir, exist_ok=True)
            _write_prediction(ck)
            print(f"  [scGPT-C1] checkpoint prediction -> {ck}", flush=True)

    _write_prediction(out_path)
    print(f"[scGPT-C1] wrote prediction for '{stim_label}'", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
