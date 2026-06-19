#!/usr/bin/env python
"""Defensive statistics for the reviewer-driven Results rewrite: cluster (hierarchical) bootstrap CIs that
respect non-independence (resampling the biological unit — lineage / dataset / donor), leave-one-dataset-out
robustness for the C3 negative result, a pre-specified-primary-floor check (so 'best floor' is not a
test-set oracle), and the model-applicability audit. Deposits results/_paper/defensive_stats.json.
All inputs from results/{C1,C3,C4,C5}/results_raw.csv (+ results/C5/loct_fine6.csv). Deterministic (seeded).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
SIMPLE = ["ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"]
NT = ["cell-mean", "donor-shift", "linear-PCA"]      # non-trivial floors (exclude control-as-prediction)
DEEP = {"latent", "graph", "foundation", "hybrid"}
rng = np.random.default_rng(0)
B = 10000


def _ran(c):
    d = pd.read_csv(R / c / "results_raw.csv"); return d[d.ran == True]  # noqa: E712


def cluster_boot(unit_vals, fn=np.mean):
    """Cluster bootstrap: resample the UNITS (each a scalar gap) with replacement, B times."""
    u = np.asarray(unit_vals, float); n = len(u)
    bs = np.array([fn(u[rng.integers(0, n, n)]) for _ in range(B)])
    return float(fn(u)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int((bs > 0).sum())


out = {}

# ---------- C5 cell-context: FP-ridge vs best floor, cluster bootstrap over 6 fine lineages ----------
f6 = pd.read_csv(R / "C5" / "loct_fine6.csv"); f6 = f6[f6.ran == True]
lin_gap, prim_gap = [], []
for s in sorted(f6.split.unique()):
    sub = f6[f6.split == s]
    fp = sub[sub.baseline == "FP-ridge"].pearson_delta
    fl = sub[sub.baseline.isin(SIMPLE)].pearson_delta
    prim = sub[sub.baseline.isin(["cell-mean", "donor-shift"])].pearson_delta  # pre-specified primary
    if len(fp) and len(fl):
        lin_gap.append(float(fp.iloc[0] - fl.max()))
        if len(prim):
            prim_gap.append(float(fp.iloc[0] - prim.max()))
m, lo, hi, npos = cluster_boot(lin_gap)
out["C5_cellcontext"] = dict(n_lineages=len(lin_gap), mean_gap=m, ci=[lo, hi], boot_pos_frac=npos / B,
                             per_lineage=lin_gap, vs_primary_floor_mean=float(np.mean(prim_gap)))

# ---------- C3 perturbation: best-deep vs best-floor, cluster bootstrap over 5 datasets ----------
d3 = _ran("C3"); M = "pearson_delta_ontarget"
cell_gap = {}   # dataset -> [gap per holdout]
floor_winner = []
prim_gap_c3 = []   # using cell-mean/donor-shift as the pre-specified primary floor
for ds in sorted(d3.dataset.unique()):
    cell_gap[ds] = []
    for h in ["10", "25", "50"]:
        sub = d3[(d3.dataset == ds) & (d3.split == f"C3_true_lo_gene_{h}")]
        if not len(sub):
            continue
        simp = sub[sub.baseline.isin(SIMPLE)]
        best_floor = simp[M].max()
        floor_winner.append(simp.loc[simp[M].idxmax(), "baseline"])
        best_deep = sub[sub.family.isin(DEEP)][M].max()
        prim_floor = sub[sub.baseline.isin(["cell-mean", "donor-shift"])][M].max()
        cell_gap[ds].append(best_deep - best_floor)
        prim_gap_c3.append(float(best_deep - prim_floor))
# cluster bootstrap over datasets: each draw resamples 5 datasets, pools their cell gaps, takes the mean
ds_list = list(cell_gap)
allcells = [g for ds in ds_list for g in cell_gap[ds]]
def _c3_draw():
    pick = rng.integers(0, len(ds_list), len(ds_list))
    pooled = [g for i in pick for g in cell_gap[ds_list[i]]]
    return np.mean(pooled)
bs = np.array([_c3_draw() for _ in range(B)])
out["C3_perturbation"] = dict(n_datasets=len(ds_list), n_cells=len(allcells), mean_gap=float(np.mean(allcells)),
                              cluster_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                              boot_neg_frac=float((bs < 0).mean()),
                              cells_cond_wins=int(sum(g > 0 for g in allcells)),
                              vs_primary_floor_mean=float(np.mean(prim_gap_c3)),
                              floor_winner_counts={b: floor_winner.count(b) for b in set(floor_winner)})
# leave-one-dataset-out: drop each dataset, mean gap over remaining
lodo_ds = {}
for drop in ds_list:
    rem = [g for ds in ds_list if ds != drop for g in cell_gap[ds]]
    lodo_ds[drop] = dict(mean_gap=float(np.mean(rem)), all_negative=bool(all(g < 0 for g in rem)))
out["C3_leave_one_dataset_out"] = lodo_ds

# ---------- Donor: random-split inflation, cluster bootstrap over 8 donors ----------
d1 = _ran("C1")
lo_ = d1[d1.split.str.startswith("C1_lodo")]; rn_ = d1[d1.split.str.startswith("C1_randsplit")]
donors = sorted(s.replace("C1_lodo_", "") for s in lo_.split.unique())
per_donor = []
for dn in donors:
    infl = []
    for b in NT:
        a = lo_[(lo_.baseline == b) & (lo_.split == f"C1_lodo_{dn}")].pearson_delta
        c = rn_[(rn_.baseline == b) & (rn_.split == f"C1_randsplit_f{dn}")].pearson_delta
        if len(a) and len(c):
            infl.append(float(c.iloc[0] - a.iloc[0]))
    if infl:
        per_donor.append(np.mean(infl))
m, lo, hi, npos = cluster_boot(per_donor)
out["donor_inflation"] = dict(n_donors=len(per_donor), mean=m, cluster_ci=[lo, hi], boot_pos_frac=npos / B,
                              per_donor=[float(x) for x in per_donor])

# ---------- pre-specified primary floor: is training-mean/donor-shift the top floor? ----------
out["primary_floor_note"] = dict(
    rule="Universal two-member floor = {cell-mean, linear-PCA}; a model must beat BOTH members. "
         "Donor shift and training-mean shift are descriptive context comparators, not universal-floor "
         "members; the per-cell best-of-four-floors is reported descriptively as an upper floor only.",
    C3_floor_winner_counts=out["C3_perturbation"]["floor_winner_counts"])

# ---------- applicability audit (native/adapted/floor per cluster from the 'action' column) ----------
audit = {}
for c in ["C1", "C3", "C4", "C5"]:
    d = _ran(c)
    col = "action" if "action" in d.columns else None
    if col:
        sub = d.drop_duplicates(["baseline"])[["baseline", "family", col]]
        audit[c] = {r.baseline: r[col] for _, r in sub.iterrows()}
out["applicability_audit"] = audit

(R / "_paper" / "defensive_stats.json").write_text(json.dumps(out, indent=2))
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "per_lineage" and kk != "per_donor"}) for k, v in out.items()}, indent=2))
print("\nwrote results/_paper/defensive_stats.json")
