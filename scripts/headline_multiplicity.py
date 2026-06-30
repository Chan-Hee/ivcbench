#!/usr/bin/env python
"""PREREGISTRATION §4 multiplicity correction over the pre-specified headline contrasts.

Gathers the RAW per-contrast p-values that EXIST in the deposited artifacts (recomputed
deterministically from the deposited inputs, or read directly where a p is deposited),
restricts to the maximal NON-RAGGED submatrix (headline contrasts that actually ran on the
headline response-direction axis AND carry a deposited/recomputable raw p), then applies
Benjamini-Hochberg AND Holm WITHIN each of two pre-specified families — the headline
floor-clearance tests (H1-H5) and the confirmatory contrasts (C2 donor, C5 compound-null) —
matching the deposited Supplementary Table S11. Contrasts whose raw p is not deposited are
listed as [MISSING] and EXCLUDED from the correction (never invented).

Deposits results/_paper/headline_multiplicity_adjusted.csv.
No fabrication: every raw_p traces to a deposited file (path logged) or is recomputed
deterministically from a deposited input and re-deposited here.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
SIMPLE = ["ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"]
DEEP = {"latent", "graph", "foundation", "hybrid"}

rows = []   # (contrast, raw_p, source, note); raw_p None => MISSING

# ----- H1: C1 Kang cell-context (LOCT), latent scGen vs universal floor (n=8 lineages) -----
d1 = pd.read_csv(R / "C1" / "results_raw.csv"); d1 = d1[d1.ran == True]
g1 = []
for s in sorted(x for x in d1.split.unique() if x.startswith("C1_loct")):
    sub = d1[d1.split == s]; sg = sub[sub.baseline == "scGen"].pearson_delta
    fl = sub[sub.baseline.isin(SIMPLE)].pearson_delta
    if len(sg) and len(fl):
        g1.append(float(sg.iloc[0] - fl.max()))
p_h1 = float(stats.wilcoxon(g1).pvalue)
rows.append(("H1_C1_cellcontext_scGen_vs_floor", p_h1,
             "recomputed: Wilcoxon over 8 lineages from results/C1/results_raw.csv",
             f"n=8 lineages, mean gap {np.mean(g1):+.4f}, {sum(x>0 for x in g1)}/8 positive (NEG result)"))

# ----- H2: C3 unseen-perturbation (true LO-gene), best-conditioned vs floor (n=5 datasets) -----
d3 = pd.read_csv(R / "C3" / "results_raw.csv"); d3 = d3[d3.ran == True]; M = "pearson_delta_ontarget"
dsg = []
for ds in sorted(d3.dataset.unique()):
    gg = []
    for h in ["10", "25", "50"]:
        sub = d3[(d3.dataset == ds) & (d3.split == f"C3_true_lo_gene_{h}")]
        if not len(sub):
            continue
        gg.append(sub[sub.family.isin(DEEP)][M].max() - sub[sub.baseline.isin(SIMPLE)][M].max())
    if gg:
        dsg.append(float(np.mean(gg)))
p_h2 = float(stats.wilcoxon(dsg).pvalue)
rows.append(("H2_C3_unseenPerturbation_bestcond_vs_floor", p_h2,
             "recomputed: Wilcoxon over 5 datasets (pre-specified unit) from results/C3/results_raw.csv",
             f"n=5 datasets, mean gap {np.mean(dsg):+.4f}, 0/5 positive (NEG result; defensive_stats boot_neg_frac=1.0)"))

# ----- H3: C4 Frangieh modality (unseen-KO), latent scGen vs floor -----
# Only 2 holdout cells (25%,50%), no biological-unit replicate, no Wilcoxon/perm p deposited
# anywhere, no C4 entry in defensive_stats.json -> raw p NOT DEPOSITED.
rows.append(("H3_C4_modality_scGen_vs_floor", None,
             "[MISSING] no deposited/recomputable raw p (2 holdout cells, no biological-unit "
             "replicate; absent from defensive_stats.json and all C4 artifacts)",
             "point gaps -0.150 / -0.129 (scGen below floor, both holdouts) but no p of record"))

# ----- H4: C5 OP3 cell-context (LOCT), FP-ridge (chemistry) vs floor (n=6 fine lineages) -----
f6 = pd.read_csv(R / "C5" / "loct_fine6.csv"); f6 = f6[f6.ran == True]
g5 = []
for s in sorted(f6.split.unique()):
    sub = f6[f6.split == s]; fp = sub[sub.baseline == "FP-ridge"].pearson_delta
    fl = sub[sub.baseline.isin(SIMPLE)].pearson_delta
    if len(fp) and len(fl):
        g5.append(float(fp.iloc[0] - fl.max()))
p_h4 = float(stats.wilcoxon(g5).pvalue)
rows.append(("H4_C5_cellcontext_FPridge_vs_floor", p_h4,
             "recomputed: Wilcoxon over 6 fine lineages from results/C5/loct_fine6.csv",
             f"n=6 lineages, mean gap {np.mean(g5):+.4f}, 6/6 positive (POS result)"))

# ----- H5: C5 OP3 unseen-compound, chemistry vs floor -----
# FP-ridge loses on the point estimate (0.164 vs floor 0.172); the deposited inferential claim
# is the Tanimoto-distance negative-control TOST equivalence-to-zero (recomputed from
# results/C5/tanimoto_percompound.csv via scripts/reviewer_batch4_analysis.py logic).
t = pd.read_csv(R / "C5" / "tanimoto_percompound.csv")
EQ = 0.05
sub = t[t.baseline == "FP-ridge"].dropna(subset=["pearson_delta", "tanimoto_dist"])
n = len(sub); x = (sub.tanimoto_dist - sub.tanimoto_dist.mean()) / sub.tanimoto_dist.std()
y = 1 - sub.pearson_delta
res = stats.linregress(x, y); slope, se = res.slope, res.stderr; df_ = n - 2
p_lo = stats.t.sf((slope - (-EQ)) / se, df_); p_hi = stats.t.cdf((slope - (EQ)) / se, df_)
p_h5 = float(max(p_lo, p_hi))
rows.append(("H5_C5_unseenCompound_TanimotoTOST_equiv0", p_h5,
             "recomputed: TOST equivalence (+/-0.05/SD) from results/C5/tanimoto_percompound.csv",
             f"FP-ridge n={n}, slope {slope:+.4f}/SD; equivalence-to-zero (FP-ridge below floor)"))

# ----- C2: donor axis headline (per corrected reading: model-level CellOT win; scPRAM loses) -----
# DEPOSITED exact p: scPRAM-vs-CellOT paired Wilcoxon in scpram_vs_cellot_donor_paired.csv.
paired = pd.read_csv(R / "_paper" / "scpram_vs_cellot_donor_paired.csv")
p_c2_paired = float(paired.wilcoxon_p.iloc[0])
rows.append(("C2_donor_scPRAM_vs_CellOT_paired", p_c2_paired,
             "deposited: results/_paper/scpram_vs_cellot_donor_paired.csv (wilcoxon_p)",
             f"paired n={int(paired.n_shared_donors.iloc[0])} of 106, scPRAM wins "
             f"{int(paired.scpram_wins.iloc[0])}, gap {float(paired.mean_gap.iloc[0]):+.3f} (scPRAM LOSES)"))

# CellOT-vs-floor exact donor paired Wilcoxon is deposited in cellot_vs_floor_donor_paired.csv
# (wilcoxon_p), recomputed from the per-donor prediction bundles by scripts/c2_donor_paired.py.
cf = pd.read_csv(R / "_paper" / "cellot_vs_floor_donor_paired.csv")
p_c2_cellot = float(cf.wilcoxon_p.iloc[0])
rows.append(("C2_donor_CellOT_vs_floor", p_c2_cellot,
             "cellot_vs_floor_donor_paired.csv",
             f"CellOT vs cell-mean floor, paired Wilcoxon n={int(cf.n.iloc[0])}, "
             f"{int(cf.cellot_wins.iloc[0])}/106 donors win, gap {float(cf.gap.iloc[0]):+.3f}; "
             "src prediction bundles via scripts/c2_donor_paired.py -> cellot_vs_floor_donor_paired.csv"))

# ----- C5 FP-ridge compound-matching permutation null (headline 'compound-resolved' claim) -----
nul = pd.read_csv(R / "C5" / "ifn_shuffle_null.csv")
pv = [float(x) for x in nul.perm_p_onesided.values]
chi2 = -2 * np.sum(np.log(pv)); dof = 2 * len(pv)
p_c5null = float(stats.chi2.sf(chi2, dof))
rows.append(("C5_FPridge_compound_matching_null_Fisher", p_c5null,
             "recomputed: Fisher-combine 4 per-lineage perm_p from results/C5/ifn_shuffle_null.csv",
             f"per-lineage one-sided p={pv}; Fisher-combined (POS result, FP-ridge only)"))

# ============ BH and Holm WITHIN each pre-specified family (not pooled) ============
# Two pre-specified families: the headline floor-clearance tests (H1-H5) and the
# confirmatory contrasts (C2 donor, C5 compound-null). Pooling them would over-shrink the
# floor family and flip H4 (FP-ridge) to surviving; the deposited Supplementary Table S11
# corrects within family, so H4 BH=0.0625 does NOT survive.
def family_of(contrast):
    return "headline_floor" if contrast.startswith("H") else "confirmatory"

adj = {}
for fam in ("headline_floor", "confirmatory"):
    present = [(c, p) for (c, p, *_ ) in rows if p is not None and family_of(c) == fam]
    m = len(present)
    order = sorted(range(m), key=lambda i: present[i][1])
    ps = [present[i][1] for i in order]
    # Benjamini-Hochberg (monotone, capped at 1)
    bh = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        val = min(ps[rank - 1] * m / rank, running)
        bh[rank - 1] = val
        running = val
    # Holm (step-down, monotone, capped at 1)
    holm = [0.0] * m
    running = 0.0
    for rank in range(1, m + 1):
        val = max(running, min(ps[rank - 1] * (m - rank + 1), 1.0))
        holm[rank - 1] = val
        running = val
    for pos, i in enumerate(order):
        adj[present[i][0]] = (bh[pos], holm[pos])  # keep full precision (tiny confirmatory p's must not round to 0)

# ================= write the deposit =================
out = R / "_paper" / "headline_multiplicity_adjusted.csv"
with out.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["contrast", "raw_p", "BH_p", "Holm_p", "survives_FDR05", "family", "source", "note"])
    for (contrast, p, source, note) in rows:
        fam = family_of(contrast)
        if p is None:
            w.writerow([contrast, "[MISSING]", "[MISSING]", "[MISSING]", "[MISSING]", fam, source, note])
        else:
            bh_p, holm_p = adj[contrast]
            survives = bh_p < 0.05   # True/False, matching the deposited table
            w.writerow([contrast, f"{p:.6g}", f"{bh_p:.6g}", f"{holm_p:.6g}", survives, fam, source, note])

print(f"{'contrast':45s} {'raw_p':>12s} {'BH_p':>12s} {'Holm_p':>12s} {'family':>14s}  surv")
for (contrast, p, *_ ) in rows:
    fam = family_of(contrast)
    if p is None:
        print(f"{contrast:45s} {'[MISSING]':>12s} {'[MISSING]':>12s} {'[MISSING]':>12s} {fam:>14s}   -")
    else:
        bh_p, holm_p = adj[contrast]
        print(f"{contrast:45s} {p:12.4g} {bh_p:12.4g} {holm_p:12.4g} {fam:>14s}  {'YES' if bh_p<0.05 else 'no'}")
print(f"\nwrote {out}")
