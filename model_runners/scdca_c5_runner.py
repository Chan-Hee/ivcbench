#!/usr/bin/env python
"""scDCA runner (C5 / OP3) — drug-conditional adapter on a frozen scGPT backbone.

Invoked exactly like every other heavy runner:
    <env python> scdca_c5_runner.py <in_payload.npz> <out_pred.npz>

Method: Maleki et al., "Efficient Fine-Tuning of Single-Cell Foundation Models Enables Zero-Shot
Molecular Perturbation Prediction", arXiv:2412.13478v2 (ICLR-2025 MLGenX spotlight). The authors
released NO code -- this is a from-paper re-implementation of Appendix A.2 eq. (6)-(12); see
benchmark/vendor/scDCA/README.md for the search record and the one declared equation ambiguity.

Why one runner serves BOTH C5 splits: scDCA's own publication evaluates the two tasks this file has
to cover, and they differ only in what is held out of the training table.
  * paper task (d) "unseen cell line, zero-shot" == ivcbench T5c `C5_loct_<lineage>`: one context is
    reserved entirely for testing "except the negative control measurement" -- literally ivcbench's
    control_inference_only contract.
  * paper task (a) "unseen drug" == ivcbench T5u `C5_global_compound_holdout`.
The runner picks the mode from the payload: any test perturbation absent from `pert_train` means the
compound axis is held out (T5u); otherwise the held axis is the context (T5c).

Data model (paper 3.1): a training example is a PSEUDOBULK / metacell row
    input   X^(0)(c)  = mean DMSO/control profile of context c        (the ONLY context featurization
                        -- scDCA has no cell-line embedding, which is why it can go zero-shot)
    drug    emb_m(d)  = frozen molecular embedding
    target  X^(d)(c)  = mean profile of the cells of compound d in context c
OP3 supplies all three; it supplies no dose sweep and no cell-line identity, neither of which scDCA
consumes.

Molecular embedding, in order of preference:
  1. $IVCBENCH_SCDCA_MOLEMB -> .npz with `keys` (compound names) + `vals` (float matrix). Build it
     with scripts_onboarding/chemberta_embed.py; ChemBERTa (Chithrananda et al. 2020) is what the
     paper uses.
  2. the payload's own `fingerprint_keys` / `fingerprint_vals` (RDKit Morgan, from side_info). The
     paper states scDCA "is compatible with virtually all molecular embedding models", so this is a
     declared substitution, not an approximation of convenience.

scGPT checkpoint from $IVCBENCH_SCGPT_MODEL_DIR (vocab.json / config.json / best_model.pt), the same
one scgpt_runner.py uses. Knobs: $IVCBENCH_SCDCA_{EPOCHS,SEQLEN,BATCH,BOTTLENECK,LR,MINCELLS}.
Defaults are the paper's (A.2): 20 epochs, batch 16, Adam lr 1e-4. The paper does not state the
adapter bottleneck width; d_bottleneck=16 is the value at which the trainable-parameter count lands
at 463,841 / 50,742,241 = 0.91% of the backbone, i.e. it is the width that reproduces the paper's
own "less than 1% of the original foundation model" claim, so it is the default here.

Leak safety: only `X_train` (train fold) and the held unit's CONTROL cells (`X_ctrl_inf`) are read.
The held lineage's treated cells / the held compound's cells never enter a training row, and the
gene panel + every context control mean are computed from those same leak-safe arrays.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_VENDOR = Path("/data1/home/chlee/projects/immune virtual cell/benchmark/vendor/scDCA")
sys.path.insert(0, str(os.environ.get("IVCBENCH_SCDCA_VENDOR", _VENDOR)))


def _model_dir() -> Path:
    p = os.environ.get("IVCBENCH_SCGPT_MODEL_DIR")
    if not p:
        raise FileNotFoundError(
            "set $IVCBENCH_SCGPT_MODEL_DIR to the scGPT_human checkpoint directory"
        )
    d = Path(p)
    for f in ("vocab.json", "config.json", "best_model.pt"):
        if not (d / f).exists():
            raise FileNotFoundError(f"scGPT model dir {d} missing {f}")
    return d


def _mol_table(d) -> tuple[dict, int, str]:
    """compound -> molecular embedding vector."""
    ext = os.environ.get("IVCBENCH_SCDCA_MOLEMB")
    if ext and Path(ext).exists():
        e = np.load(ext, allow_pickle=True)
        tab = {str(k): np.asarray(v, np.float32) for k, v in zip(e["keys"], e["vals"])}
        return tab, len(next(iter(tab.values()))), f"ChemBERTa({Path(ext).name})"
    if "fingerprint_keys" not in d.files:
        raise RuntimeError(
            "scDCA: no molecular representation — set $IVCBENCH_SCDCA_MOLEMB or pass "
            "side_info['fingerprint'] so the payload carries fingerprint_keys/vals"
        )
    tab = {
        str(k): np.asarray(v, np.float32)
        for k, v in zip(d["fingerprint_keys"], d["fingerprint_vals"])
    }
    return tab, len(next(iter(tab.values()))), "Morgan-fingerprint(payload)"


def main(in_path: str, out_path: str) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from scgpt.model import TransformerGenerator
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.utils import load_pretrained, set_seed

    from scdca_model import ScDCA, param_report

    set_seed(0)
    md = _model_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqlen = int(os.environ.get("IVCBENCH_SCDCA_SEQLEN", "2000"))
    epochs = int(os.environ.get("IVCBENCH_SCDCA_EPOCHS", "20"))  # paper A.2
    batch = int(os.environ.get("IVCBENCH_SCDCA_BATCH", "16"))  # paper A.2
    lr = float(os.environ.get("IVCBENCH_SCDCA_LR", "1e-4"))  # paper A.2
    bottleneck = int(
        os.environ.get("IVCBENCH_SCDCA_BOTTLENECK", "16")
    )  # see note below
    min_cells = int(os.environ.get("IVCBENCH_SCDCA_MINCELLS", "3"))

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]])
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if not test_perts:
        raise RuntimeError("scDCA: payload carries no held perturbation label")

    ct_train = (
        np.array([str(c) for c in d["celltype_train"]])
        if "celltype_train" in d.files
        else np.array(["all"] * X.shape[0])
    )
    ct_inf = (
        np.array([str(c) for c in d["celltype_inf"]])
        if "celltype_inf" in d.files
        else np.array(["held"] * X_ctrl_inf.shape[0])
    )
    mol_tab, d_mol, mol_src = _mol_table(d)

    unseen = [p for p in test_perts if p not in set(pert_train)]
    mode = "T5u/unseen-compound" if unseen else "T5c/unseen-context(LOCT)"

    # ---------------- gene panel: genes in the scGPT vocabulary, ranked by training variance -------
    vocab = GeneVocab.from_file(str(md / "vocab.json"))
    for tok in ("<pad>", "<cls>", "<eoc>"):
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab["<pad>"])
    gid_all = np.array(
        [vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=np.int64
    )
    valid = np.where(gid_all != vocab["<pad>"])[0]
    if valid.size == 0:
        raise RuntimeError("scDCA: no dataset gene is in the scGPT vocabulary")
    sel = valid[np.argsort(X[:, valid].var(0))[::-1][:seqlen]]
    sel = np.sort(sel)
    gene_ids = torch.tensor(gid_all[sel], dtype=torch.long)

    # ---------------- pseudobulk table: X^(0)(c) inputs and X^(d)(c) targets ----------------------
    ctrl_mean = {}
    for c in np.unique(ct_train):
        m = is_ctrl & (ct_train == c)
        if m.sum() >= min_cells:
            ctrl_mean[c] = X[m][:, sel].mean(0)
    if not ctrl_mean:
        raise RuntimeError(
            "scDCA: no training context has enough control cells for X^(0)(c)"
        )

    rows_in, rows_tgt, rows_mol, n_skip = [], [], [], 0
    for c, cm in ctrl_mean.items():
        for p in sorted(set(pert_train[~is_ctrl])):
            if p not in mol_tab:
                n_skip += 1
                continue
            m = (~is_ctrl) & (pert_train == p) & (ct_train == c)
            if m.sum() < min_cells:
                continue
            rows_in.append(cm)
            rows_tgt.append(X[m][:, sel].mean(0))
            rows_mol.append(mol_tab[p])
    if not rows_in:
        raise RuntimeError(
            "scDCA: no (compound x context) pseudobulk training row could be built"
        )
    Xin = torch.tensor(np.asarray(rows_in, np.float32))
    Xtg = torch.tensor(np.asarray(rows_tgt, np.float32))
    Mol = torch.tensor(np.asarray(rows_mol, np.float32))

    print(
        "[scDCA]"
        f" mode={mode} contexts={sorted(ctrl_mean)} genes={len(sel)}/{len(genes)} "
        f"pseudobulk_rows={len(rows_in)} mol={mol_src} d_mol={d_mol} "
        f"epochs={epochs} batch={batch} bottleneck={bottleneck} device={device}",
        flush=True,
    )

    # ---------------- frozen scGPT backbone + drug-conditional adapters --------------------------
    cfg = json.load(open(md / "config.json"))
    backbone = TransformerGenerator(
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
    backbone = load_pretrained(
        backbone, torch.load(md / "best_model.pt", map_location="cpu"), verbose=False
    )
    model = ScDCA(
        backbone, d_model=cfg["embsize"], d_mol=d_mol, d_bottleneck=bottleneck
    ).to(device)
    print(
        f"[scDCA] {param_report(model)} literal_eq11={model.literal_eq11}", flush=True
    )

    opt = torch.optim.Adam(model.trainable_parameters(), lr=lr)  # paper A.2: Adam, 1e-4
    loader = DataLoader(TensorDataset(Xin, Xtg, Mol), batch_size=batch, shuffle=True)
    gid_dev = gene_ids.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for xi, xt, mm in loader:
            xi, xt, mm = xi.to(device), xt.to(device), mm.to(device)
            gid = gid_dev.unsqueeze(0).expand(xi.shape[0], -1)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = torch.nn.functional.mse_loss(model(gid, xi, mm), xt)  # eq (12)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss)
            nb += 1
        print(f"[scDCA] epoch {ep + 1}/{epochs} mse={tot / max(nb, 1):.5f}", flush=True)

    # ---------------- inference ------------------------------------------------------------------
    # T5c: the held context's OWN control cells are X^(0)(c_held) -> one prediction per compound.
    # T5u: the compound is unseen; every SEEN context is a legitimate X^(0)(c), so predict in each and
    #      combine with the lineage composition of the control-inference pool (control cells only --
    #      the harness contract keys a compound-split prediction by compound alone).
    if unseen:
        ctxs = list(ctrl_mean)
        comp = np.array([float((ct_inf == c).sum()) for c in ctxs])
        w = comp / comp.sum() if comp.sum() > 0 else np.ones(len(ctxs)) / len(ctxs)
        ctx_in = np.asarray([ctrl_mean[c] for c in ctxs], np.float32)
        base_full = np.asarray(
            [X[is_ctrl & (ct_train == c)].mean(0) for c in ctxs], np.float32
        )
    else:
        if X_ctrl_inf.shape[0] < 1:
            raise RuntimeError(
                "scDCA: LOCT mode needs the held context's control cells (X_ctrl_inf)"
            )
        ctxs, w = ["<held>"], np.array([1.0])
        ctx_in = X_ctrl_inf[:, sel].mean(0)[None, :].astype(np.float32)
        base_full = X_ctrl_inf.mean(0)[None, :].astype(np.float32)

    model.eval()
    pred_perts, pred_means, missing = [], [], []
    Xc = torch.tensor(ctx_in, device=device)
    gid_c = gid_dev.unsqueeze(0).expand(Xc.shape[0], -1)
    with torch.no_grad():
        for p in test_perts:
            if p not in mol_tab:
                missing.append(p)
                continue
            mm = torch.tensor(
                np.repeat(mol_tab[p][None, :], Xc.shape[0], 0), device=device
            )
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(gid_c, Xc, mm).float().cpu().numpy()
            full = base_full.copy()
            full[:, sel] = out  # unmodelled genes keep that context's control mean
            pred_perts.append(p)
            pred_means.append((w[:, None] * full).sum(0).astype(np.float32))

    if not pred_perts:
        raise RuntimeError(
            "scDCA: no held perturbation had a molecular embedding "
            f"(missing e.g. {missing[:3]})"
        )
    np.savez(
        out_path,
        pred_perts=np.array(pred_perts, dtype=object),
        pred_means=np.vstack(pred_means).astype(np.float32),
    )
    print(
        f"[scDCA] wrote {len(pred_perts)} profiles x {len(genes)} genes (skipped"
        f" {len(missing)} without a molecular embedding; {n_skip} train pairs skipped)",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
