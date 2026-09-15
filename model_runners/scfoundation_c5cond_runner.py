#!/usr/bin/env python
"""scFoundation on the OP3 COMPOUND cluster with a COMPOUND-CONDITIONED head — `scfoundation` env.

Invoked by ivcbench.baselines.heavy.ScFoundationC5Cond:
    <scfoundation python> scfoundation_c5cond_runner.py <in.npz> <out.npz>

WHY THIS RUNNER EXISTS
----------------------
The previously wired T5u path (`fp_wrapper.FPUnseenCompound`) is CIRCULAR: it regresses a Morgan
fingerprint onto the MODEL'S OWN predicted training-compound responses, so the frozen encoder never
sees the held compound and the cell is bounded above by the FP-ridge baseline. It is not an
evaluation of scFoundation. This runner replaces it with the same shape chemCPA uses, which is not
circular: a frozen molecular representation feeds a trainable map whose output enters the MODEL'S OWN
decoding path, and the training target is the OBSERVED response.

    prediction(compound c, held context) = mean_ctrl(held context)
                                         + mean_i head( [ E_frozen(control cell i) || fp(c) ] )

  * E_frozen — scFoundation's own released 19264-gene `cell` encoder, HELD FIXED. The embedding path
    (token_emb + pos_emb -> encoder -> [g[-1] ‖ g[-2] ‖ max ‖ mean] = 3072-d) is byte-identical to
    scfoundation_c1_runner.py / scfoundation_runner.py, batch-1 so no pad token enters the pooling.
  * head — the only trainable part (a two-branch MLP: cell branch on the frozen embedding, chemistry
    branch on the 1024-bit Morgan fingerprint, then a shared trunk to the FULL gene panel). Trained on
    the TRAIN FOLD ONLY, against the OBSERVED perturbed pseudobulk of each (compound, lineage) present
    in that fold, expressed as a difference from that lineage's own control mean; the control mean is
    a known quantity at inference and is added back there, exactly as chemCPA's decoder is applied on
    top of a basal state.
  * at inference the held compound's fingerprint goes through the SAME head, together with the
    embeddings of the held context's own CONTROL cells.

This is strictly more information than Ridge(fingerprint -> observed response): the head also sees
the cell representation, which is what carries the lineage identity on the held-lineage split.

This is an extension of an interface already written and already disclosed for this model
(scfoundation_c1_runner.py is a frozen encoder + a trained MLP head); the cells it produces are
reported as ADAPTED, never as native.

TWO REGIMES, ONE RUNNER (detected from the payload, never from a flag)
  * held-LINEAGE (T5c, compounds seen): the held lineage is absent from every training cell; the
    inference input is its own control cells. One profile per compound (141).
  * held-COMPOUND (T5u, lineages seen): the held compounds are absent from every training cell; the
    inference input is the control pool. One profile per held compound (28).
In BOTH regimes the runner emits ONE PROFILE PER COMPOUND, keyed by the compound label, so the
adapter's `pred_key_is_group` stays False and no held stratum falls back to the control mean.

LEAK SAFETY (asserted at run time, printed, and fatal on violation)
  * unseen-compound regime: no training cell may carry a held compound label; the head's targets are
    built only from training compounds; the held compound enters only as its fingerprint.
  * held-lineage regime: the held lineage may not appear among the training cells; only its control
    cells are read, as `X_ctrl_inf`.
Everything data-adaptive (control means, the target table, the embedding standardisation, the
inner-validation split, the epoch count) is estimated on the training fold alone.

EPOCH SELECTION is an inner split of the TRAIN fold that mirrors the held axis (unseen-compound ->
hold out 15% of training compounds; held-lineage -> hold out one training lineage). The epoch with the
best inner-validation mean per-target Pearson correlation is chosen, then the head is refit on the
whole train fold for that many epochs. No held-out quantity is involved.

Knobs: $IVCBENCH_SCFC_{EPOCHS,K,CTRLCAP,MINCELLS,HIDDEN,LR,WD,SEED}; encoder resolution token
$IVCBENCH_SCF_HIGHRES (shared with the other scFoundation runners).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scfoundation_runner import _scf_dir, _ckpt_path  # noqa: E402  (same env discovery)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 0 else 0.0


class _Tee:
    """Mirror the runner's own stdout to $IVCBENCH_SCFC_LOG. The adapter runs this script with
    capture_output=True and discards stdout on success, so without this the leak assertions, the
    embedding cost and the epoch selection would be invisible in the sweep log."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def main(in_path: str, out_path: str) -> None:
    log_path = os.environ.get("IVCBENCH_SCFC_LOG")
    if log_path:
        _fh = open(log_path, "a", buffering=1)
        sys.stdout = _Tee(sys.stdout, _fh)
        print(
            "\n===== scfoundation_c5cond_runner"
            f" {time.strftime('%F %T')} in={in_path} =====",
            flush=True,
        )
    import torch
    import torch.nn as nn
    import pandas as pd

    scf_dir = _scf_dir()
    sys.path.insert(0, str(scf_dir))
    from load import load_model_frommmf, gatherData

    seed = int(os.environ.get("IVCBENCH_SCFC_SEED", "0"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_epochs = int(os.environ.get("IVCBENCH_SCFC_EPOCHS", "60"))
    k_ctrl = int(os.environ.get("IVCBENCH_SCFC_K", "128"))
    ctrl_cap = int(os.environ.get("IVCBENCH_SCFC_CTRLCAP", "1600"))
    min_cells = int(os.environ.get("IVCBENCH_SCFC_MINCELLS", "10"))
    hidden = int(os.environ.get("IVCBENCH_SCFC_HIDDEN", "512"))
    lr = float(os.environ.get("IVCBENCH_SCFC_LR", "1e-3"))
    wd = float(os.environ.get("IVCBENCH_SCFC_WD", "1e-4"))
    highres = os.environ.get("IVCBENCH_SCF_HIGHRES", "t4")

    # ------------------------------------------------------------------ payload
    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)  # log1p-normalised HVG, TRAIN fold only
    genes = [str(g) for g in d["genes"]]
    n_genes = len(genes)
    is_ctrl = d["is_control_train"].astype(bool)
    pert_train = np.asarray([str(p) for p in d["pert_train"]])
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)  # inference-input CONTROL cells
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if "fingerprint_keys" not in d.files:
        raise RuntimeError(
            "scFoundation-C5cond: payload carries no fingerprint_* (compound side-rep);"
            " the adapter must set requires_compound_side = True"
        )
    fp_by_cpd = {
        str(k): np.asarray(v, np.float32).ravel()
        for k, v in zip(d["fingerprint_keys"], d["fingerprint_vals"])
    }
    fp_dim = int(len(next(iter(fp_by_cpd.values()))))

    ct_train = (
        np.asarray([str(c) for c in d["celltype_train"]])
        if "celltype_train" in d.files
        else np.full(X.shape[0], "all")
    )
    held_label = str(d["held_label"]) if "held_label" in d.files else ""

    train_cpds_all = sorted({p for p in pert_train[~is_ctrl]})
    train_cpds = [c for c in train_cpds_all if c in fp_by_cpd]
    pred_cpds = [c for c in test_perts if c in fp_by_cpd]
    if not pred_cpds:
        raise RuntimeError(
            "scFoundation-C5cond: none of the held compound labels has a fingerprint"
        )
    unseen_regime = len(set(test_perts) & set(train_cpds_all)) == 0

    # ------------------------------------------------------------- LEAK ASSERTIONS
    if unseen_regime:
        n_leak = int(np.isin(pert_train, np.asarray(test_perts)).sum())
        if n_leak:
            raise AssertionError(
                f"LEAK: {n_leak} training cells carry one of the {len(test_perts)} "
                "held compound labels"
            )
        assert not (
            set(test_perts) & set(train_cpds_all)
        ), "LEAK: held compound in the head's target table"
        print(
            f"[leak] held-COMPOUND regime: 0 of {X.shape[0]} training cells carry any"
            f" of the {len(test_perts)} held compound labels; the head's targets come"
            f" from {len(train_cpds)} training compounds only; the held compounds enter"
            f" as a {fp_dim}-bit fingerprint and nothing else",
            flush=True,
        )
    else:
        lin_seen = set(ct_train.tolist())
        if not held_label or held_label in lin_seen:
            raise AssertionError(
                f"LEAK: held lineage '{held_label}' is present among the training "
                f"cells (training lineages: {sorted(lin_seen)})"
            )
        print(
            f"[leak] held-LINEAGE regime: lineage '{held_label}' absent from all"
            f" {X.shape[0]} training cells (training lineages {sorted(lin_seen)});"
            f" inference input is its own {X_ctrl_inf.shape[0]} control cells;"
            f" {len(pred_cpds)} seen compounds to predict",
            flush=True,
        )
    if not is_ctrl.any():
        raise RuntimeError(
            "scFoundation-C5cond: the training fold has no control cells"
        )

    # ------------------------------------------------------------- frozen encoder
    t0 = time.time()
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
    print(
        f"[scFoundation-C5cond] frozen `cell` encoder loaded in {time.time()-t0:.0f}s; "
        f"{int(have.sum())}/{n_genes} panel genes map onto the 19264 index",
        flush=True,
    )

    @torch.no_grad()
    def embed(rows: np.ndarray, tag: str) -> np.ndarray:
        """FROZEN 3072-d cell embedding — identical path to scfoundation_c1_runner.py (batch-1, so
        `gatherData` pads nothing and the max/mean pooling never sees a pad token)."""
        out, t = [], time.time()
        for i, r in enumerate(rows):
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
            if (i + 1) % 250 == 0:
                print(
                    f"  [embed:{tag}] {i+1}/{len(rows)} cells  {time.time()-t:.0f}s",
                    flush=True,
                )
        return np.vstack(out).astype(np.float32)

    rng = np.random.default_rng(seed)
    ctrl_rows = np.where(is_ctrl)[0]
    if len(ctrl_rows) > ctrl_cap:
        ctrl_rows = np.sort(rng.choice(ctrl_rows, ctrl_cap, replace=False))
    inf_rows = np.arange(X_ctrl_inf.shape[0])
    if len(inf_rows) > ctrl_cap:
        inf_rows = np.sort(rng.choice(inf_rows, ctrl_cap, replace=False))

    # On the held-COMPOUND split the inference-input controls ARE the train fold's controls (no
    # lineage is held out), so the same frozen embeddings serve both and are computed once.
    _tr_ctrl_block = X[np.where(is_ctrl)[0]]
    same_pool = _tr_ctrl_block.shape == X_ctrl_inf.shape and np.array_equal(
        _tr_ctrl_block, X_ctrl_inf
    )
    t0 = time.time()
    E_tr = embed(X[ctrl_rows], "train-ctrl")
    d_emb = E_tr.shape[1]
    if same_pool:
        _pos = {int(r): i for i, r in enumerate(np.where(is_ctrl)[0])}
        inf_rows = np.asarray([_pos[int(r)] for r in ctrl_rows])
        E_inf = E_tr
        print(
            "[embed] inference-control pool is identical to the training-control pool "
            "(held-compound regime) -> embeddings reused, not recomputed",
            flush=True,
        )
    else:
        E_inf = embed(X_ctrl_inf[inf_rows], "inference-ctrl")
    print(
        f"[embed] {len(ctrl_rows)} training-control + {len(inf_rows)} inference-control"
        f" cells -> {d_emb}-d in {time.time()-t0:.0f}s",
        flush=True,
    )

    # ------------------------------------------------- target table (TRAIN fold only)
    lin_tr = ct_train[ctrl_rows]  # lineage of each embedded train control
    lineages = sorted(set(ct_train.tolist()))
    ctrl_mu = {}
    for l in lineages:
        m = is_ctrl & (ct_train == l)
        if m.sum():
            ctrl_mu[l] = X[m].mean(0)
    cpd_list = sorted(set(train_cpds) | set(pred_cpds))
    cpd_ix = {c: i for i, c in enumerate(cpd_list)}
    FP = np.zeros((len(cpd_list), fp_dim), np.float32)
    for c, i in cpd_ix.items():
        FP[i] = fp_by_cpd[c]

    Y, meta, pairs = [], [], []
    for c in train_cpds:
        for l in sorted(ctrl_mu):
            m = (~is_ctrl) & (pert_train == c) & (ct_train == l)
            n = int(m.sum())
            if n < min_cells:
                continue
            ti = len(Y)
            Y.append(X[m].mean(0) - ctrl_mu[l])  # OBSERVED response, train fold
            meta.append((c, l, n))
            pool = np.where(lin_tr == l)[0]
            if len(pool) == 0:
                Y.pop()
                meta.pop()
                continue
            take = (
                pool if len(pool) <= k_ctrl else rng.choice(pool, k_ctrl, replace=False)
            )
            for e in take:
                pairs.append((int(e), cpd_ix[c], ti))
    if len(Y) < 20:
        raise RuntimeError(
            f"scFoundation-C5cond: only {len(Y)} (compound, lineage) targets in the "
            "train fold; the head is not estimable"
        )
    Yn = np.vstack(Y).astype(np.float32)
    P = np.asarray(pairs, dtype=np.int64)
    tgt_cpd = np.asarray([m[0] for m in meta])
    tgt_lin = np.asarray([m[1] for m in meta])
    print(
        f"[head] {Yn.shape[0]} (compound x lineage) observed targets over"
        f" {len(set(tgt_cpd))} training compounds x {len(set(tgt_lin))} lineages;"
        f" {P.shape[0]} (control cell, compound) training pairs; target dim ="
        f" {n_genes} (FULL panel)",
        flush=True,
    )

    # inner-validation split of the TRAIN fold, mirroring the held axis
    if unseen_regime:
        vc_all = sorted(set(tgt_cpd.tolist()))
        n_val = max(3, int(round(0.15 * len(vc_all))))
        vc = set(
            rng.choice(np.asarray(vc_all, dtype=object), n_val, replace=False).tolist()
        )
        val_t = np.asarray([c in vc for c in tgt_cpd])
        print(
            f"[head] inner validation = {n_val} held-back TRAINING compounds "
            "(mirrors the unseen-compound axis)",
            flush=True,
        )
    else:
        vls = sorted(set(tgt_lin.tolist()))
        vl_pick = vls[0]
        val_t = tgt_lin == vl_pick
        print(
            f"[head] inner validation = training lineage '{vl_pick}' held back "
            "(mirrors the unseen-lineage axis)",
            flush=True,
        )

    mu, sd = E_tr.mean(0), E_tr.std(0) + 1e-6  # standardisation: TRAIN controls only
    Et = torch.tensor((E_tr - mu) / sd, dtype=torch.float32, device=device)
    Ei = torch.tensor((E_inf - mu) / sd, dtype=torch.float32, device=device)
    Ft = torch.tensor(FP, dtype=torch.float32, device=device)
    Yt = torch.tensor(Yn, dtype=torch.float32, device=device)

    class CondHead(nn.Module):
        """[frozen cell embedding] and [Morgan fingerprint] each get their own projection, then a
        shared trunk emits the full-panel response. The compound representation passes THROUGH the
        head that also carries the cell representation — the property fp_wrapper lacked.
        """

        def __init__(self):
            super().__init__()
            self.cell = nn.Sequential(nn.Linear(d_emb, 256), nn.ReLU())
            self.chem = nn.Sequential(
                nn.Linear(fp_dim, 256), nn.ReLU(), nn.Dropout(0.1)
            )
            self.trunk = nn.Sequential(
                nn.Linear(512, hidden), nn.ReLU(), nn.Linear(hidden, n_genes)
            )

        def forward(self, e, f):
            return self.trunk(torch.cat([self.cell(e), self.chem(f)], dim=1))

    def run_head(idx_pairs, epochs, track_val=None, tag=""):
        torch.manual_seed(seed)
        head = CondHead().to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
        ei = torch.tensor(idx_pairs[:, 0], device=device)
        ci = torch.tensor(idx_pairs[:, 1], device=device)
        ti = torch.tensor(idx_pairs[:, 2], device=device)
        n = len(ei)
        curve = []
        for ep in range(epochs):
            head.train()
            perm = torch.randperm(n, device=device)
            tot = 0.0
            for i in range(0, n, 512):
                b = perm[i : i + 512]
                opt.zero_grad()
                loss = nn.functional.mse_loss(head(Et[ei[b]], Ft[ci[b]]), Yt[ti[b]])
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            if track_val is not None:
                curve.append((ep + 1, _val_score(head, track_val)))
                if (ep + 1) % 5 == 0 or ep == 0:
                    print(
                        f"  [head{tag}] epoch {ep+1}/{epochs} mse={tot/n:.5f} "
                        f"val_r={curve[-1][1]:.4f}",
                        flush=True,
                    )
            elif (ep + 1) % 10 == 0 or ep == 0:
                print(
                    f"  [head{tag}] epoch {ep+1}/{epochs} mse={tot/n:.5f}", flush=True
                )
        return head, curve

    @torch.no_grad()
    def _val_score(head, vpairs):
        """Mean per-target Pearson between the head's averaged prediction and the OBSERVED response —
        the same shape as the reported metric, computed inside the train fold."""
        head.eval()
        ei = torch.tensor(vpairs[:, 0], device=device)
        ci = torch.tensor(vpairs[:, 1], device=device)
        out = np.zeros((Yn.shape[0], n_genes), np.float64)
        cnt = np.zeros(Yn.shape[0])
        for i in range(0, len(ei), 4096):
            sl = slice(i, i + 4096)
            p = head(Et[ei[sl]], Ft[ci[sl]]).double().cpu().numpy()
            for j, t in enumerate(vpairs[sl, 2]):
                out[t] += p[j]
                cnt[t] += 1
        ts = np.where(cnt > 0)[0]
        return (
            float(np.mean([_pearson(out[t] / cnt[t], Yn[t]) for t in ts]))
            if len(ts)
            else 0.0
        )

    val_pair_mask = val_t[P[:, 2]]
    tr_pairs, va_pairs = P[~val_pair_mask], P[val_pair_mask]
    print(
        f"[head] inner split: {len(tr_pairs)} train pairs / {len(va_pairs)} validation"
        " pairs",
        flush=True,
    )
    t0 = time.time()
    _, curve = run_head(tr_pairs, max_epochs, track_val=va_pairs, tag=":select")
    # smooth the validation curve over a centred 3-epoch window before taking the argmax: a single
    # epoch's inner-validation score is noisy and picking its raw maximum overfits the selection
    vals = [v for _, v in curve]
    smooth = [float(np.mean(vals[max(0, i - 1) : i + 2])) for i in range(len(vals))]
    bi = int(np.argmax(smooth))
    best_ep, best_val = curve[bi][0], curve[bi][1]
    print(
        f"[head] best inner-validation epoch = {best_ep}/{max_epochs} "
        f"(val_r={best_val:.4f}, 3-epoch mean {smooth[bi]:.4f}; raw max "
        f"{max(vals):.4f} at epoch {curve[int(np.argmax(vals))][0]}); "
        f"selection took {time.time()-t0:.0f}s",
        flush=True,
    )

    t0 = time.time()
    head, _ = run_head(P, best_ep, tag=":refit")
    print(
        f"[head] refit on all {P.shape[0]} pairs for {best_ep} epochs in"
        f" {time.time()-t0:.0f}s",
        flush=True,
    )

    # --------------------------------------------------------------- prediction
    ctrl_ref = X_ctrl_inf.mean(0).astype(np.float64)  # == the harness's control_mean
    head.eval()
    preds = []
    with torch.no_grad():
        for c in pred_cpds:
            f = Ft[cpd_ix[c]].unsqueeze(0).expand(Ei.shape[0], -1)
            dl = head(Ei, f).double().cpu().numpy().mean(0)
            preds.append(ctrl_ref + dl)
    Pm = np.vstack(preds).astype(np.float32)

    spread = float(np.mean(Pm.std(0))) if Pm.shape[0] > 1 else 0.0
    same_as_ctrl = int(
        (np.abs(Pm.astype(np.float64) - ctrl_ref[None, :]).max(1) <= 1e-5).sum()
    )
    print(
        f"[scFoundation-C5cond] emitted {Pm.shape[0]}/{len(test_perts)} held compound"
        f" profiles over {n_genes}/{n_genes} genes (emitted-gene fraction 1.000);"
        f" {same_as_ctrl} identical to the control mean; compound specificity (mean"
        f" over genes of the SD across compounds) = {spread:.4f}",
        flush=True,
    )
    np.savez(out_path, pred_perts=np.array(pred_cpds, dtype=object), pred_means=Pm)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
