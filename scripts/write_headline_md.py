#!/usr/bin/env python3
"""Render the cross-cluster HEADLINE and WITHIN-FAMILY CONSISTENCY markdown tables
from the assembled CSVs. NEW files only; does NOT touch results_section.md."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
OUT = os.path.join(ROOT, "results", "_paper")

h = pd.read_csv(os.path.join(OUT, "cross_cluster_headline.csv"))
c = pd.read_csv(os.path.join(OUT, "within_family_consistency.csv"))

def fmt(x, nd=4):
    if pd.isna(x):
        return "—"
    s = f"{x:.{nd}f}"
    return s.replace("-", "−")  # unicode minus

# ---------------- HEADLINE markdown ----------------
lines = []
lines.append("# Cross-cluster HEADLINE table — response-direction delta vs the UNIVERSAL floor\n")
lines.append("Generated mechanically from `cross_cluster_headline.csv` "
             "(`scripts/assemble_cross_cluster.py`). **Real, already-computed results only.**\n")
lines.append("**Metric:** response-direction = Pearson-Δ (`pearson_delta`), PREREG axis 1 (headline).  ")
lines.append("**Reference:** the UNIVERSAL simple floor = {`cell-mean`, `linear-PCA`} (PREREG §2), "
             "NOT cluster-specific floors (donor-shift / FP-ridge are context-only and excluded here).  ")
lines.append("`delta_vs_floor_mean` = family Pearson-Δ − mean(cell-mean, linear-PCA).  ")
lines.append("`beats_both` = point estimate exceeds BOTH floor members. This is the **point-estimate "
             "direction/magnitude** read; the CI-gated fit verdict (CI_low>0 on the gap, PREREG §5) "
             "is the separate descriptive fit-matrix and is NOT asserted here.  ")
lines.append("Biological unit macro-averaged per cluster (PREREG §7): C1 lineage, C2 donor, C3 dataset, "
             "C4 modality-fold (RNA), C5 lineage / compound.\n")

for (cl, task, split), g in h.groupby(["cluster", "task", "split"], sort=False):
    fcm = g["floor_cell_mean"].iloc[0]; flp = g["floor_linear_PCA"].iloc[0]; fm = g["floor_mean"].iloc[0]
    lines.append(f"## {cl} — {task} — {split}")
    lines.append(f"Universal floor: cell-mean = {fmt(fcm)}, linear-PCA = {fmt(flp)}, "
                 f"floor-mean = {fmt(fm)}  (unit = {g['unit'].iloc[0]}, n = {g['n_unit'].iloc[0]})\n")
    lines.append("| family | model | Pearson-Δ | Δ vs floor-mean | Δ vs cell-mean | Δ vs linear-PCA | beats both? |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in g.sort_values("delta_vs_floor_mean", ascending=False).itertuples():
        lines.append(f"| {r.family} | {r.model} | {fmt(r.pearson_delta)} | "
                     f"{fmt(r.delta_vs_floor_mean)} | {fmt(r.delta_vs_cell_mean)} | "
                     f"{fmt(r.delta_vs_linear_PCA)} | {'**yes**' if r.beats_both_floor_members else 'no'} |")
    lines.append("")

# headline narrative (mechanical)
lines.append("## Read (mechanical)\n")
beat_rows = h[h["beats_both_floor_members"]]
lines.append(f"- Conditioned models that beat BOTH universal-floor members (point estimate): "
             f"{len(beat_rows)} of {len(h)} (family,model)×task cells — "
             + ("; ".join(f"{r.model}@{r.cluster}/{r.task.split('/')[0]} "
                         f"({r.split.split('(')[0].strip()}, +{r.delta_vs_floor_mean:.3f})"
                         for r in beat_rows.itertuples()) if len(beat_rows) else "none") + ".")
# This bullet was the last hardcoded string in the file. Written for the 35-cell census, it said
# "C5 unseen-compound: chemCPA/scGen below floor" -- and after the tables were regenerated from the
# 58-cell census neither model has a row in that split, while scFoundation IS printed there above
# the binding floor, which the bullet's own sibling on the line above lists by name. Derive the two
# extrapolation splits so the sentence can only say what the table says.
def _below(cluster, split_contains):
    m = h[(h["cluster"] == cluster) & (h["split"].str.contains(split_contains, case=False))]
    return m, int((~m["beats_both_floor_members"]).sum()), len(m)

_c3, _c3_below, _c3_n = _below("C3", "gene")
_c5, _c5_below, _c5_n = _below("C5", "compound")
def _phrase(tag, m, below, n):
    if not n:
        return f"{tag}: no rows"
    if below == n:
        return f"{tag}: all {n} conditioned entries below floor"
    beat = "; ".join(sorted(r.model for r in m[m["beats_both_floor_members"]].itertuples()))
    return (f"{tag}: {below} of {n} below floor, with {beat} above it on the point estimate "
            f"(see the uncertainty column before reading that as a clearance)")
lines.append("- Pattern matches the integrated finding: conditioning helps on **cell/donor-context "
             "transfer** (C2 CellOT donor-LODO; C5 FP-ridge LOCT) but largely **fails on "
             "unseen-perturbation extrapolation** — "
             + _phrase("C3 LO-gene", _c3, _c3_below, _c3_n) + "; "
             + _phrase("C5 unseen-compound", _c5, _c5_below, _c5_n) + ".")
with open(os.path.join(OUT, "cross_cluster_headline.md"), "w") as fh:
    fh.write("\n".join(lines) + "\n")

# ---------------- CONSISTENCY markdown ----------------
cl2 = []
cl2.append("# Within-family CONSISTENCY table\n")
cl2.append("Generated from `within_family_consistency.csv`. For each family with ≥2 models on a task: "
           "do the members **agree** on the beat-floor verdict, and how correlated are their per-unit "
           "Pearson-Δ vectors (Spearman ρ)?  Floor = universal {cell-mean, linear-PCA}.\n")
cl2.append("| cluster | task | split | family | models | n beat both floor | n models | "
           "verdict agreement | Spearman ρ (per-unit) | flag |")
cl2.append("|---|---|---|---|---|---|---|---|---|---|")
for r in c.itertuples():
    rho = fmt(r.spearman_rho_pair, 3) if not pd.isna(r.spearman_rho_pair) else "— (n<3 units)"
    flag = (r.flag if isinstance(r.flag, str) and r.flag and r.flag != "nan" else "")
    cl2.append(f"| {r.cluster} | {r.task} | {r.split} | {r.family} | {r.models} | "
               f"{r.n_beat_both_floor} | {r.n_models} | {r.verdict_agreement} | {rho} | {flag} |")
cl2.append("")
cl2.append("## Notes\n")
# This note used to assert "agree in every family/task cell ... no within-family verdict split
# anywhere in the matrix" as a hardcoded string. It was true of the 35-cell census it was written
# for; on the current roster four family/task cells split, and the sentence contradicted the table
# printed directly above it. Derive it, so the note can only ever say what the data says.
_agree = int((c["verdict_agreement"] == "agree").sum())
_split = int((c["verdict_agreement"] == "split").sum())
if _split == 0:
    cl2.append("- **Verdict agreement = `agree` in every family/task cell**: paired members of the "
               "same family reach the SAME beat-floor verdict (all beat, or none beat). No "
               "within-family verdict split anywhere in the matrix.")
else:
    _rows = c[c["verdict_agreement"] == "split"]
    _which = "; ".join(f"{r.cluster} {r.family} ({r.models})" for r in _rows.itertuples())
    cl2.append(f"- **Verdict agreement: {_agree} of {_agree + _split} family/task cells agree**, "
               f"{_split} split — {_which}. In a split cell one member of the family clears both "
               "floor members and the other does not, so family membership does not by itself "
               "predict the beat-floor verdict.")
# These three bullets used to carry hardcoded rho values from the 35-cell census. Every one of
# them had drifted: C3 Foundation was printed as 1.00 against a current 0.80, C3 Graph as 0.90
# against 1.00, and C1 Latent as "0.48 ... agree directionally" against a current -0.05, which is
# no agreement at all and the opposite sign. Derive them from the same frame the table is built
# from, and print nothing where rho is undefined.
def _rho(cluster, family):
    m = c[(c["cluster"] == cluster) & (c["family"] == family)]
    m = m[m["spearman_rho_pair"].notna()]
    return None if m.empty else float(m["spearman_rho_pair"].iloc[0])

_c3 = [(f, _rho("C3", f)) for f in ("Foundation", "Graph", "Hybrid", "Flow")]
_c3 = [(f, v) for f, v in _c3 if v is not None]
if _c3:
    cl2.append("- **C3 within-family rho** (" + ", ".join(f"{f} {fmt(v, 2)}" for f, v in _c3) +
               "): where rho is high the family's members rank datasets the same way even though "
               "both sit below the floor, which is consistent failure rather than noise.")
_lat = [(cl, _rho(cl, "Latent")) for cl in ("C1", "C2", "C5")]
_lat = [(cl, v) for cl, v in _lat if v is not None]
if _lat:
    cl2.append("- **Latent-family rho by cluster**: " +
               ", ".join(f"{cl} {fmt(v, 2)}" for cl, v in _lat) +
               ". These are two-model rank correlations over the units of one split; read them as "
               "indicative, not as an estimate with an interval.")
# The old version of this bullet carried an internal "FLAGGED for re-run" TODO about CellOT and
# scPRAM on Frangieh. Neither is in the T4 panel any more, so the flag named work that no longer
# applies, in a file a reader downloads. State the structural reason rho is undefined instead.
_undef = sorted({r.family for r in c[(c["cluster"] == "C4") &
                                     (c["spearman_rho_pair"].isna())].itertuples()})
if _undef:
    cl2.append("- **C4 rho is undefined** for " + ", ".join(_undef) +
               ": the modality split has only two folds (LO-KO 25% and 50%), fewer than the three "
               "units a rank correlation needs. The census reports those cells by their observed "
               "unit range rather than an interval, for the same reason.")
with open(os.path.join(OUT, "within_family_consistency.md"), "w") as fh:
    fh.write("\n".join(cl2) + "\n")

print("wrote cross_cluster_headline.md and within_family_consistency.md")
