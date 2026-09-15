#!/usr/bin/env python
"""scGPT runner for the COMPOUND cluster (C5/OP3) with a COMPOUND-CONDITIONED head — `scgpt` env.

Invoked by ivcbench.baselines.heavy.ScGPTC5Cond:
    <scgpt env python> scgpt_c5_cond_runner.py <in.npz> <out.npz>

WHY THIS RUNNER EXISTS
----------------------
The two existing scGPT entry points cannot represent a compound:
  * scgpt_c1_runner.py collapses every exposure into ONE global condition flag, so all 141 OP3
    compounds share a single predicted profile (that is the deposited C1-adapter T5c entry);
  * the fingerprint wrapper (baselines/fp_wrapper.py) fits Ridge(fingerprint -> the MODEL'S OWN
    predicted training-compound responses), so the encoder never sees the held compound and the
    cell is bounded above by FP-ridge. That is circular and is not an evaluation of scGPT.

This runner gives scGPT the same SHAPE chemCPA has: a frozen molecular vector feeds a trainable map
whose output is decoded through the model's own cell representation, and the training target is the
OBSERVED perturbed profile.

    prediction(compound c, held unit) = mean_control_profile
                                      + mean_{cells i in the held unit's own CONTROL cells}
                                            head([ E_frozen(x_i) || Morgan(c) ])

  * E_frozen  — scGPT_human's released encoder, FROZEN. Cell embeddings come from the released
    cell-embedding path (scgpt.tasks.cell_emb.get_batch_cell_embeddings: <cls> token, 51-bin value
    binning in the collator, L2-normalised output), with the model rebuilt from the checkpoint's own
    config.json because this checkpoint directory ships config.json rather than the args.json that
    scgpt.tasks.embed_data reads. Nothing else about the embedding path is changed.
  * head       — the ONLY trainable part: MLP([embedding ‖ fingerprint]) -> per-gene Δ, trained on
    the TRAIN fold only, against the OBSERVED (lineage x compound) mean response.
  * at inference the held compound's fingerprint goes through the SAME head.

Freezing the encoder is deliberate and is disclosed: it makes scGPT and scFoundation directly
comparable on this split, and the head design requires a fixed cell representation. scFoundation is
already in this shape (model_runners/scfoundation_c1_runner.py = frozen encoder + trained MLP head),
so this is an extension of an interface already written and already disclosed, not a new invention.

ONE RUNNER, BOTH C5 SPLITS
--------------------------
  * T5u (C5_global_compound_holdout): the held COMPOUNDS never appear in training; the inference
    input is the control-cell pool. One profile per held compound.
  * T5c (C5_loct_<lineage>): the held LINEAGE never appears in training; the inference input is that
    lineage's own DMSO cells. One profile per (seen) compound.
The regime is detected from the payload, and the corresponding leak assertion is enforced and
printed. Either way the runner emits ONE PROFILE PER COMPOUND keyed by the compound label, so the
adapter leaves pred_key_is_group False.

LEAK BOUNDARY (asserted at run time, see _assert_leak_free)
  * head training targets are group means over payload X_train only (held compound / held lineage
    already removed by the split builder);
  * at inference only X_ctrl_inf (control cells) and the held compound's fingerprint are used;
  * in the unseen-compound regime the held compound labels must not occur in pert_train, and no
    training target group may carry a held compound label.

Checkpoint dir via $IVCBENCH_SCGPT_MODEL_DIR. Head/embedding knobs via $IVCBENCH_FCOND_*.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scgpt_runner import _model_dir  # noqa: E402  (same checkpoint discovery)


# --------------------------------------------------------------------------------------- frozen encoder
def _frozen_cell_embeddings(
    X: np.ndarray, genes: list[str], device, *, batch_size: int, max_length: int
) -> np.ndarray:
    """scGPT_human cell embeddings, encoder FROZEN, via the released cell-embedding path.

    Mirrors scgpt.tasks.embed_data: build TransformerModel from the checkpoint config, load the
    released weights, then call the package's own get_batch_cell_embeddings (<cls> embedding,
    n_bins=51 binning inside DataCollator, L2 normalisation). embed_data itself is not called because
    it reads `args.json`, which this released checkpoint directory does not ship; every other element
    of the path is the library's.
    """
    import json
    import anndata as ad
    import torch
    from scgpt.model import TransformerModel
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.utils import load_pretrained
    from scgpt.tasks.cell_emb import get_batch_cell_embeddings

    md = _model_dir()
    cfg = json.load(open(md / "config.json"))
    vocab = GeneVocab.from_file(str(md / "vocab.json"))
    for tok in ["<pad>", "<cls>", "<eoc>"]:
        if tok not in vocab:
            vocab.append_token(tok)
    vocab.set_default_index(vocab["<pad>"])

    ids = np.array([vocab[g] if g in vocab else -1 for g in genes], dtype=np.int64)
    keep = np.where(ids >= 0)[0]
    print(f"[scGPT-C5cond] vocab match {len(keep)}/{len(genes)} genes", flush=True)
    if len(keep) < 50:
        raise RuntimeError(
            "scGPT-C5cond: fewer than 50 panel genes are in the scGPT vocabulary"
        )

    model_configs = {
        "embsize": int(cfg["embsize"]),
        "pad_token": "<pad>",
        # scGPT pretraining convention (mask_value -1 / pad_value -2); the <cls>
        # expression slot and the padding slots are filled with pad_value.
        "pad_value": int(os.environ.get("IVCBENCH_FCOND_PADVALUE", "-2")),
    }
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=int(cfg["embsize"]),
        nhead=int(cfg["nhead"]),
        d_hid=int(cfg["d_hid"]),
        nlayers=int(cfg["nlayers"]),
        nlayers_cls=3,
        n_cls=1,
        vocab=vocab,
        dropout=float(cfg.get("dropout", 0.0)),
        pad_token="<pad>",
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        input_emb_style=cfg.get("input_emb_style", "continuous"),
        cell_emb_style=cfg.get("cell_emb_style", "cls"),
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=(cfg.get("norm_scheme", "post") == "pre"),
    )
    sd = torch.load(md / "best_model.pt", map_location="cpu")
    model = load_pretrained(model, sd, verbose=False)
    # loud check that the released weights really landed (load_pretrained silently drops mismatches)
    msd = model.state_dict()
    ren = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items()}
    hit = sum(
        1 for k, v in ren.items() if k in msd and tuple(v.shape) == tuple(msd[k].shape)
    )
    print(
        f"[scGPT-C5cond] loaded {hit}/{len(msd)} encoder tensors from best_model.pt",
        flush=True,
    )
    if hit < 0.8 * len(msd):
        raise RuntimeError(
            f"scGPT-C5cond: only {hit}/{len(msd)} pretrained tensors matched; the "
            "encoder would be partly random"
        )
    model.to(device).eval()
    for p in model.parameters():  # FROZEN, explicitly
        p.requires_grad_(False)

    Xs = np.ascontiguousarray(X[:, keep].astype(np.float32))
    nnz = (Xs > 0).sum(1)
    print(
        f"[scGPT-C5cond] embedding {Xs.shape[0]} cells; nnz/cell"
        f" median={np.median(nnz):.0f} max={nnz.max()} (max_length={max_length})",
        flush=True,
    )
    adata = ad.AnnData(Xs)
    gene_ids = ids[keep]
    emb = get_batch_cell_embeddings(
        adata,
        cell_embedding_mode="cls",
        model=model,
        vocab=vocab,
        max_length=max_length,
        batch_size=batch_size,
        model_configs=model_configs,
        gene_ids=gene_ids,
        use_batch_labels=False,
    )
    del model
    torch.cuda.empty_cache()
    return np.asarray(emb, dtype=np.float32)


# ------------------------------------------------------------------------------------------ leak gate
def _assert_leak_free(
    *, unseen_cpd, test_cpds, train_cpds, lin_train, lin_inf, group_keys
):
    """Hard gate. Raises before a single head gradient step if the held entity could reach training."""
    tr, te = set(train_cpds), set(test_cpds)
    if unseen_cpd:
        bad = sorted(te & tr)
        if bad:
            raise RuntimeError(
                f"LEAK: {len(bad)} held compounds appear in the training labels, "
                f"e.g. {bad[:3]}"
            )
        bad_g = sorted({c for _, c in group_keys} & te)
        if bad_g:
            raise RuntimeError(
                f"LEAK: {len(bad_g)} head-training target groups carry a held "
                f"compound label, e.g. {bad_g[:3]}"
            )
        print(
            f"[leak-check] unseen-COMPOUND regime: {len(te)} held compounds, 0 of them"
            f" in the {len(tr)} training compounds and 0 in the"
            f" {len(group_keys)} head-training groups; only their fingerprints are used"
            " at inference. OK",
            flush=True,
        )
    else:
        bad = sorted(set(lin_inf) & set(lin_train))
        if bad:
            raise RuntimeError(
                f"LEAK: the held lineage(s) {bad} also occur in the training cells"
            )
        print(
            "[leak-check] held-LINEAGE regime: inference lineage(s)"
            f" {sorted(set(lin_inf))} absent from the {len(set(lin_train))} training"
            f" lineage(s) {sorted(set(lin_train))}; head trained on"
            f" {len(group_keys)} (lineage x compound) observed groups. OK",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------- main
def main(in_path: str, out_path: str) -> None:
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb_batch = int(os.environ.get("IVCBENCH_FCOND_EMB_BATCH", "32"))
    emb_maxlen = int(os.environ.get("IVCBENCH_FCOND_EMB_MAXLEN", "1200"))
    n_tr_ctrl = int(
        os.environ.get("IVCBENCH_FCOND_TRCTRL", "128")
    )  # ctrl cells/lineage -> head rows
    n_inf_ctrl = int(
        os.environ.get("IVCBENCH_FCOND_INFCTRL", "512")
    )  # ctrl cells averaged at predict
    min_cells = int(os.environ.get("IVCBENCH_FCOND_MINCELLS", "5"))
    hidden = int(os.environ.get("IVCBENCH_FCOND_HIDDEN", "512"))
    epochs = int(os.environ.get("IVCBENCH_FCOND_EPOCHS", "60"))
    lr = float(os.environ.get("IVCBENCH_FCOND_LR", "1e-3"))
    wd = float(os.environ.get("IVCBENCH_FCOND_WD", "1e-4"))
    bs = int(os.environ.get("IVCBENCH_FCOND_BATCH", "512"))
    val_frac = float(os.environ.get("IVCBENCH_FCOND_VALFRAC", "0.2"))

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    pert_train = np.array([str(p) for p in d["pert_train"]])
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if "fingerprint_keys" not in d.files:
        raise RuntimeError(
            "scGPT-C5cond: payload carries no fingerprint_* (compound side-rep)"
        )
    fp_by_cpd = {
        str(k): np.asarray(v, np.float32).ravel()
        for k, v in zip(d["fingerprint_keys"], d["fingerprint_vals"])
    }
    fp_dim = len(next(iter(fp_by_cpd.values())))
    lin_train = (
        np.array([str(c) for c in d["celltype_train"]])
        if "celltype_train" in d.files
        else np.array(["all"] * X.shape[0])
    )
    lin_inf = (
        np.array([str(c) for c in d["celltype_inf"]])
        if "celltype_inf" in d.files
        else np.array(["all"] * X_ctrl_inf.shape[0])
    )
    if len(lin_inf) != X_ctrl_inf.shape[0]:
        raise RuntimeError("scGPT-C5cond: celltype_inf does not align with X_ctrl_inf")

    train_cpds = sorted({c for c in pert_train[~is_ctrl] if c in fp_by_cpd})
    test_cpds = [c for c in test_perts if c in fp_by_cpd]
    dropped = [c for c in test_perts if c not in fp_by_cpd]
    if len(train_cpds) < 5:
        raise RuntimeError(
            f"scGPT-C5cond: only {len(train_cpds)} training compounds carry a "
            "fingerprint; the head is not estimable"
        )
    if not test_cpds:
        raise RuntimeError("scGPT-C5cond: no held compound carries a fingerprint")
    unseen_cpd = bool(set(test_cpds) - set(train_cpds))

    # ---- observed targets: mean Δ of each (training lineage x training compound) group ------------
    ctrl_mean_by_lin = {}
    for l in sorted(set(lin_train[is_ctrl])):
        ctrl_mean_by_lin[l] = X[is_ctrl & (lin_train == l)].mean(0)
    global_ctrl_mean = X[is_ctrl].mean(0)
    group_keys, group_Y, group_n = [], [], []
    for l in sorted(set(lin_train)):
        base = ctrl_mean_by_lin.get(l, global_ctrl_mean)
        for c in train_cpds:
            m = (~is_ctrl) & (lin_train == l) & (pert_train == c)
            n = int(m.sum())
            if n >= min_cells:
                group_keys.append((l, c))
                group_Y.append(X[m].mean(0) - base)
                group_n.append(n)
    if len(group_keys) < 20:
        raise RuntimeError(
            f"scGPT-C5cond: only {len(group_keys)} (lineage x compound) target groups"
        )
    group_Y = np.vstack(group_Y).astype(np.float32)

    _assert_leak_free(
        unseen_cpd=unseen_cpd,
        test_cpds=test_cpds,
        train_cpds=train_cpds,
        lin_train=lin_train,
        lin_inf=lin_inf,
        group_keys=group_keys,
    )

    print(
        "[scGPT-C5cond]"
        f" regime={'unseen-compound (T5u)' if unseen_cpd else 'held-lineage (T5c)'} "
        f"train_cpds={len(train_cpds)} test_cpds={len(test_cpds)}"
        + (f" (dropped, no fingerprint: {dropped})" if dropped else "")
        + f" groups={len(group_keys)} genes={len(genes)} fp_dim={fp_dim}",
        flush=True,
    )

    # ---- frozen scGPT embeddings for the CONTROL cells only ---------------------------------------
    rng = np.random.default_rng(0)
    tr_ctrl_rows = []
    for l in sorted(set(lin_train[is_ctrl])):
        idx = np.where(is_ctrl & (lin_train == l))[0]
        tr_ctrl_rows.append(
            np.sort(rng.choice(idx, n_tr_ctrl, replace=False))
            if len(idx) > n_tr_ctrl
            else idx
        )
    tr_ctrl_rows = np.concatenate(tr_ctrl_rows)
    inf_rows = (
        np.sort(rng.choice(X_ctrl_inf.shape[0], n_inf_ctrl, replace=False))
        if X_ctrl_inf.shape[0] > n_inf_ctrl
        else np.arange(X_ctrl_inf.shape[0])
    )

    E = _frozen_cell_embeddings(
        np.vstack([X[tr_ctrl_rows], X_ctrl_inf[inf_rows]]),
        genes,
        device,
        batch_size=emb_batch,
        max_length=emb_maxlen,
    )
    E_tr, E_inf = E[: len(tr_ctrl_rows)], E[len(tr_ctrl_rows) :]
    lin_tr_ctrl = lin_train[tr_ctrl_rows]
    print(
        f"[scGPT-C5cond] embeddings: train-ctrl {E_tr.shape} held-ctrl {E_inf.shape}",
        flush=True,
    )

    # ---- head training rows: (control-cell embedding, compound fingerprint) -> observed group Δ ----
    gkey_to_i = {k: i for i, k in enumerate(group_keys)}
    ci, pi, ti = [], [], []
    cpd_to_j = {c: j for j, c in enumerate(train_cpds)}
    for i_e, l in enumerate(lin_tr_ctrl):
        for c in train_cpds:
            g = gkey_to_i.get((l, c))
            if g is not None:
                ci.append(i_e)
                pi.append(cpd_to_j[c])
                ti.append(g)
    ci = np.asarray(ci, np.int64)
    pi = np.asarray(pi, np.int64)
    ti = np.asarray(ti, np.int64)
    print(
        f"[scGPT-C5cond] head rows={len(ci)} (= {len(E_tr)} ctrl cells x their"
        " lineage's groups)",
        flush=True,
    )

    FP_tr = np.vstack([fp_by_cpd[c] for c in train_cpds]).astype(np.float32)
    mu, sd = E_tr.mean(0), E_tr.std(0) + 1e-6  # standardise the embedding block only
    Et = torch.tensor((E_tr - mu) / sd, device=device)
    Ft = torch.tensor(FP_tr, device=device)
    Yt = torch.tensor(group_Y, device=device)
    cit = torch.tensor(ci, device=device)
    pit = torch.tensor(pi, device=device)
    tit = torch.tensor(ti, device=device)
    n_genes = group_Y.shape[1]
    d_in = E_tr.shape[1] + fp_dim

    def make_head(seed=0):
        torch.manual_seed(seed)
        return nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, n_genes)
        ).to(device)

    def fit(rows, epochs_, val_rows=None, val_groups=None):
        head = make_head(0)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        best = (-2.0, 0)
        for ep in range(epochs_):
            head.train()
            perm = rows[torch.randperm(len(rows), generator=g).to(rows.device)]
            tot = 0.0
            for k in range(0, len(perm), bs):
                b = perm[k : k + bs]
                opt.zero_grad()
                out = head(torch.cat([Et[cit[b]], Ft[pit[b]]], 1))
                loss = nn.functional.mse_loss(out, Yt[tit[b]])
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            if val_groups is not None:
                r = group_pearson(head, val_rows, val_groups)
                if r > best[0]:
                    best = (r, ep + 1)
                if ep % 10 == 9 or ep == 0:
                    print(
                        f"  [head] ep {ep+1}/{epochs_} mse={tot/len(perm):.5f} "
                        f"val-pearsonΔ={r:.4f} (best {best[0]:.4f} @ {best[1]})",
                        flush=True,
                    )
            elif ep % 10 == 9 or ep == 0:
                print(
                    f"  [head] ep {ep+1}/{epochs_} mse={tot/len(perm):.5f}", flush=True
                )
        return head, best

    @torch.no_grad()
    def group_pearson(head, rows, groups):
        """Mean over held-out-TRAIN-compound groups of corr(predicted Δ, observed Δ) -- the same
        quantity the benchmark scores, evaluated with the same averaging the prediction uses.
        """
        head.eval()
        rs = []
        for g_i in groups:
            m = rows[tit[rows] == g_i]
            if len(m) == 0:
                continue
            p = head(torch.cat([Et[cit[m]], Ft[pit[m]]], 1)).mean(0)
            o = Yt[g_i]
            p = p - p.mean()
            o = o - o.mean()
            den = p.norm() * o.norm()
            rs.append(float((p @ o) / den) if float(den) > 1e-12 else 0.0)
        return float(np.mean(rs)) if rs else 0.0

    # Inner model selection, entirely inside the TRAIN fold (never on the benchmark's held entity).
    # The inner holdout MIRRORS the outer regime so the epoch count is chosen under the same kind of
    # generalisation the cell is scored on: hold out TRAINING COMPOUNDS when the outer split holds out
    # compounds (T5u), hold out a TRAINING LINEAGE when the outer split holds out a lineage (T5c).
    rng2 = np.random.default_rng(0)
    all_rows = torch.arange(len(ci), device=device)
    row_lin = np.array([lin_tr_ctrl[i] for i in ci])
    if unseen_cpd:
        n_val = max(5, int(round(val_frac * len(train_cpds))))
        val_cpd_j = set(rng2.choice(len(train_cpds), n_val, replace=False).tolist())
        is_val_np = np.array([j in val_cpd_j for j in pi])
        inner_desc = f"{len(train_cpds)-n_val} train / {n_val} val COMPOUNDS"
    else:
        tr_lins = sorted(set(lin_tr_ctrl.tolist()))
        if len(tr_lins) < 2:
            raise RuntimeError(
                "scGPT-C5cond: need >=2 training lineages for the inner holdout"
            )
        val_lin = tr_lins[int(rng2.integers(len(tr_lins)))]
        is_val_np = row_lin == val_lin
        inner_desc = (
            f"{len(tr_lins)-1} train / 1 val LINEAGE (val={val_lin}, "
            f"train={[l for l in tr_lins if l != val_lin]})"
        )
    is_val = torch.tensor(is_val_np, device=device)
    inner_tr, inner_va = all_rows[~is_val], all_rows[is_val]
    val_groups = sorted(set(ti[is_val_np].tolist()))
    print(
        f"[scGPT-C5cond] inner split: {inner_desc} "
        f"({len(inner_tr)}/{len(inner_va)} rows, {len(val_groups)} val groups)",
        flush=True,
    )
    _, (best_r, best_ep) = fit(inner_tr, epochs, inner_va, val_groups)
    # Honour the inner selection exactly (textbook early stopping). An epoch FLOOR was tried first
    # and removed: it is an invented hyper-parameter and it overrode a validated, leak-safe choice.
    best_ep = max(1, best_ep)
    print(
        f"[scGPT-C5cond] inner selection: {best_ep} epochs (val pearsonΔ {best_r:.4f});"
        f" refitting on all {len(all_rows)} rows",
        flush=True,
    )
    head, _ = fit(all_rows, best_ep)

    # ---- predict one profile per compound ---------------------------------------------------------
    head.eval()
    Ei = torch.tensor((E_inf - mu) / sd, device=device)
    ctrl_profile = X_ctrl_inf.mean(0).astype(
        np.float32
    )  # == the adapter's control_mean
    pred_perts, pred_means = [], []
    with torch.no_grad():
        for c in test_cpds:
            f = torch.tensor(fp_by_cpd[c], device=device).expand(Ei.shape[0], fp_dim)
            delta = head(torch.cat([Ei, f], 1)).mean(0).cpu().numpy()
            pred_perts.append(c)
            pred_means.append(ctrl_profile + delta.astype(np.float32))
    P = np.vstack(pred_means).astype(np.float32)
    spread = float(np.mean(P.std(0)))  # per-gene SD across compounds
    print(
        f"[scGPT-C5cond] wrote {len(pred_perts)} per-compound profiles; "
        f"cross-compound spread (mean per-gene SD) = {spread:.5f}; "
        f"max |Δ| = {np.abs(P - ctrl_profile).max():.4f}",
        flush=True,
    )
    if spread < 1e-6:
        raise RuntimeError("scGPT-C5cond: predictions are identical across compounds")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object), pred_means=P)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
