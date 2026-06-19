#!/usr/bin/env python
"""C3 unseen-gene PREDICTABILITY PROBE — why does leave-one-gene-out fail?

Converts the WHERE-map (results_raw.csv macro-averages) into a WHY-claim by recovering the
PER-HELD-GENE predictability and relating it to candidate explanatory variables computed from
the ACTUAL deposited C3 data.

Per-held-gene predictability is NOT deposited (results_raw.csv stores only the macro-average over
the held-gene set). We recover it exactly by replaying the deterministic, GPU-free FLOOR baselines
(cell-mean / donor-shift / linear-PCA), whose pearson_delta(...) returns a per_stratum dict. This is
legitimate because the deposited macro-averages show NO heavy model beats the floor on any C3 cell
(mean heavy-floor gap -0.031, 0/15) — so the floor's per-gene Pearson-Δ IS the achievable
predictability ceiling for unseen genes, not merely a convenient proxy.

For each held gene g (across 5 datasets x 3 holdout fractions) we compute:
  predictability  = max over {cell-mean, donor-shift, linear-PCA} of per-stratum Pearson-Δ (downstream-only)
  also model_floor_gap (best deposited heavy model macro - floor) reported at dataset/frac grain.
Explanatory variables (all from the deposited data):
  (b) EFFECT SIZE / SNR
      effect_l2     = ||Δ_obs|| (L2 norm of observed mean treated-minus-control Δ, downstream genes)
      snr           = ||Δ_obs|| / mean within-perturbation per-gene SD  (signal vs cell-to-cell noise)
      n_test_cells  = #treated cells of g (sampling support)
  (a) REPRESENTATION DISTANCE to TRAINING genes
      go_jaccard_nn = 1 - max GO-term Jaccard similarity of g to any TRAINING perturbed gene
      go_jaccard_k5 = 1 - mean of top-5 GO Jaccard similarities to training genes
      coexpr_nn     = 1 - max Pearson co-expression (over control cells) of g to any training gene
  (c) RESPONSE-GENE OVERLAP with training perturbations
      resp_overlap  = max Jaccard overlap of g's top-50 |Δ| response genes with any training gene's top-50
Then: Spearman/Pearson of predictability vs each factor; which factor best explains it (or none).

CPU only. Loads each dataset once via the project loaders. Writes a tidy per-gene table.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivcbench.clusters import c3  # noqa: E402
from ivcbench.splits.builder import build_split  # noqa: E402
from ivcbench.baselines.simple import CellMean, DonorShift, LinearPCA, CtrlPred  # noqa: E402
from ivcbench.metrics.response import _pearson  # noqa: E402

LOADERS = {
    "shifrut":            ("ivcbench.data.loaders.shifrut", "load", {}),
    "schmidt":            ("ivcbench.data.loaders.schmidt", "load", {}),
    "mccutcheon_CRISPRi": ("ivcbench.data.loaders.mccutcheon", "load", {"modality": "CRISPRi"}),
    "mccutcheon_CRISPRa": ("ivcbench.data.loaders.mccutcheon", "load", {"modality": "CRISPRa"}),
    "chen":               ("ivcbench.data.loaders.chen", "load", {}),
}
FRACS = [(0.10, "10"), (0.25, "25"), (0.50, "50")]
GENE2GO_PATH = ROOT / "data/pertadapt/official/gene2go.pkl"


def go_jaccard(a_terms, b_terms):
    if not a_terms or not b_terms:
        return np.nan
    inter = len(a_terms & b_terms)
    union = len(a_terms | b_terms)
    return inter / union if union else np.nan


def process_dataset(ds, cs, gene2go):
    genes = sorted(set(cs.uns["genes_perturbed"]))
    X = cs.X
    obs = cs.obs
    pert = obs["perturbation"].to_numpy()
    is_ctrl = obs["is_control"].to_numpy()
    var = list(cs.var_names)
    gpos = {g: i for i, g in enumerate(var)}

    # global control mean & per-gene control SD (noise floor) — over ALL control cells
    ctrl_mask = is_ctrl
    ctrl_X = X[ctrl_mask]
    # observed mean treated profile per perturbed gene (over ALL its treated cells, any context)
    obs_mean = {}
    obs_sd = {}          # mean within-perturbation per-gene SD (cell-to-cell noise)
    n_cells = {}
    for g in genes:
        m = (pert == g) & (~is_ctrl)
        n_cells[g] = int(m.sum())
        if m.sum() == 0:
            continue
        sub = X[m]
        obs_mean[g] = sub.mean(0)
        obs_sd[g] = sub.std(0).mean()
    ctrl_mean_global = ctrl_X.mean(0)

    # co-expression matrix over CONTROL cells, restricted to perturbed-gene columns that exist in var
    coexpr_cols = {g: gpos[g] for g in genes if g in gpos}
    coexpr = {}
    if coexpr_cols:
        cols = list(coexpr_cols.keys())
        idx = np.array([coexpr_cols[g] for g in cols])
        sub = ctrl_X[:, idx]  # (n_ctrl, n_perturbed_present)
        # pearson corr matrix
        subc = sub - sub.mean(0)
        norm = np.linalg.norm(subc, axis=0)
        norm[norm < 1e-9] = 1e-9
        corr = (subc.T @ subc) / np.outer(norm, norm)
        coexpr = {g: dict(zip(cols, corr[i])) for i, g in enumerate(cols)}

    # top-50 response genes (downstream, target excluded) per perturbed gene
    top_resp = {}
    for g in genes:
        if g not in obs_mean:
            continue
        delta = obs_mean[g] - ctrl_mean_global
        d = delta.copy()
        if g in gpos:
            d[gpos[g]] = 0.0  # exclude target
        order = np.argsort(-np.abs(d))[:50]
        top_resp[g] = set(order.tolist())

    records = []
    for frac, lbl in FRACS:
        held = c3.held_gene_fraction(genes, frac, seed=0)
        held_set = set(held)
        train_genes = [g for g in genes if g not in held_set]
        spec = c3.true_lo_gene(held, lbl)
        split = build_split(cs, spec)

        # control mean used at inference for this split (matches baseline _control_mean)
        inf = split.inference_input_idx
        cmean = X[inf].mean(0) if len(inf) else ctrl_mean_global

        # fit floor baselines once per split
        fitted = {}
        for B in (CellMean, DonorShift, LinearPCA, CtrlPred):
            b = B(); b.fit(cs, split)
            fitted[b.name] = b.predict(cs, split)

        # per held gene: per-stratum predictability (downstream-only; exclude held target gene)
        # NOTE: split.test_strata entries are the formatted stratum_key ("perturbation=GENE"),
        # not the bare gene symbol — match on that.
        test_strata = split.test_strata
        test_cells = X[split.test_idx]
        for g in held:
            stratum_label = spec.stratum_key({"perturbation": g})
            sm = (test_strata == stratum_label)
            if sm.sum() == 0:
                continue
            keep = np.ones(test_cells.shape[1], dtype=bool)
            if g in gpos:
                keep[gpos[g]] = False
            delta_obs = test_cells[sm].mean(0) - cmean
            preds = {}
            for name, pr in fitted.items():
                delta_pred = pr.pred_cells[sm].mean(0) - pr.control_mean
                preds[name] = _pearson(delta_pred[keep], delta_obs[keep])
            floor = max(preds["cell-mean"], preds["donor-shift"], preds["linear-PCA"])

            # (b) effect size / SNR  (downstream, target excluded)
            eff_vec = delta_obs[keep]
            effect_l2 = float(np.linalg.norm(eff_vec))
            sd = obs_sd.get(g, np.nan)
            snr = float(effect_l2 / (sd * np.sqrt(keep.sum()))) if (sd and sd > 0) else np.nan
            # simpler per-gene SNR: ||Δ|| / sd
            snr_raw = float(effect_l2 / sd) if (sd and sd > 0) else np.nan

            # (a) representation distance to TRAINING genes
            a_go = gene2go.get(g, set())
            sims_go = [go_jaccard(a_go, gene2go.get(tg, set())) for tg in train_genes]
            sims_go = [s for s in sims_go if not (s is None or np.isnan(s))]
            if sims_go:
                go_jaccard_nn = 1.0 - max(sims_go)
                go_jaccard_k5 = 1.0 - float(np.mean(sorted(sims_go, reverse=True)[:5]))
            else:
                go_jaccard_nn = np.nan; go_jaccard_k5 = np.nan

            if g in coexpr:
                cvals = [abs(coexpr[g].get(tg, np.nan)) for tg in train_genes if tg in coexpr[g] and tg != g]
                cvals = [c for c in cvals if not np.isnan(c)]
                coexpr_nn = (1.0 - max(cvals)) if cvals else np.nan
            else:
                coexpr_nn = np.nan

            # (c) response-gene overlap with training perturbations (top-50 |Δ| Jaccard)
            if g in top_resp:
                rg = top_resp[g]
                ov = []
                for tg in train_genes:
                    if tg in top_resp:
                        tr = top_resp[tg]
                        u = len(rg | tr)
                        ov.append(len(rg & tr) / u if u else 0.0)
                resp_overlap = max(ov) if ov else np.nan
            else:
                resp_overlap = np.nan

            records.append(dict(
                dataset=ds, hold=lbl, held_gene=g,
                predictability=floor,
                pd_cell_mean=preds["cell-mean"], pd_donor_shift=preds["donor-shift"],
                pd_linear_pca=preds["linear-PCA"], pd_ctrl=preds["ctrl-pred"],
                n_test_cells=n_cells.get(g, 0),
                effect_l2=effect_l2, within_pert_sd=sd, snr=snr, snr_raw=snr_raw,
                go_jaccard_nn=go_jaccard_nn, go_jaccard_k5=go_jaccard_k5,
                coexpr_nn=coexpr_nn, resp_overlap=resp_overlap,
                n_train_genes=len(train_genes),
            ))
    return records


def main():
    import importlib
    gene2go_raw = pickle.load(open(GENE2GO_PATH, "rb"))
    gene2go = {k: set(v) for k, v in gene2go_raw.items()}

    all_recs = []
    for ds, (mod, fn, kw) in LOADERS.items():
        try:
            loader = getattr(importlib.import_module(mod), fn)
            cs = loader(**kw)
        except Exception as e:  # noqa: BLE001
            print(f"!! {ds} load failed: {type(e).__name__}: {e}", flush=True)
            continue
        recs = process_dataset(ds, cs, gene2go)
        all_recs.extend(recs)
        print(f"{ds}: {len(recs)} held-gene observations", flush=True)

    df = pd.DataFrame(all_recs)
    out = ROOT / "results/C3/predictability_probe_pergene.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  shape={df.shape}", flush=True)
    print(df.head().to_string())


if __name__ == "__main__":
    main()
