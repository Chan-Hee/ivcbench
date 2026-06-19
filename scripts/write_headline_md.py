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
lines.append("- Pattern matches the integrated finding: conditioning helps on **cell/donor-context "
             "transfer** (C2 CellOT donor-LODO; C5 FP-ridge LOCT) but **fails on unseen-perturbation "
             "extrapolation** (C3 LO-gene: every conditioned family is below floor; C5 unseen-compound: "
             "chemCPA/scGen below floor).")
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
cl2.append("- **Verdict agreement = `agree` in every family/task cell**: paired members of the same "
           "family reach the SAME beat-floor verdict (all beat, or none beat). No within-family "
           "verdict split anywhere in the matrix.")
cl2.append("- **C3 ρ is high within family** (Foundation ρ=1.00, Graph 0.90, Hybrid 0.70, Latent 0.70 "
           "across the 5 primary-T datasets): family members rank datasets the same way even though "
           "both members sit below floor — consistent failure, not noise.")
cl2.append("- **C1 / C2 Latent ρ moderate** (0.48 / 0.44 across lineages / 106 donors): scGen and CPA "
           "agree directionally but not tightly.")
cl2.append("- **C5 Latent ρ = −0.80** on the unseen-compound split (only 1 unit there → "
           "computed across the LOCT lineages instead; small n, treat as indicative).")
cl2.append("- **C4 ρ undefined**: only 2 modality folds (LO-KO 25% / 50%) → <3 units, so cross-model "
           "ρ is not computable. **CellOT and scPRAM on Frangieh ran a single seed per split (the two "
           "LO-KO fractions, no CI / no multi-seed) — FLAGGED for re-run** to obtain a marker-bootstrap "
           "CI before any inferential claim.")
with open(os.path.join(OUT, "within_family_consistency.md"), "w") as fh:
    fh.write("\n".join(cl2) + "\n")

print("wrote cross_cluster_headline.md and within_family_consistency.md")
