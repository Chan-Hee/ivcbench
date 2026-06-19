#!/usr/bin/env python
"""Part 2 - reproduce/load the current manuscript anchor values before adding CellOT/chemCPA.

Recomputes anchors from per-unit source rows where feasible (status PASS); loads from the
deposited aggregate summary where exact recomputation is non-trivial (status PASS_loaded).
Emits outputs/additional_models/anchor_reproduction_report.{csv,md}.

Run: ./.venv/bin/python scripts/reproduce_anchor_results.py
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "outputs" / "additional_models"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
def rec(name, expected, observed, tol, source, notes, loaded=False):
    if observed is None:
        status = "FAIL"; diff = None
    else:
        diff = abs(observed - expected)
        if diff <= tol:
            status = "PASS_loaded" if loaded else "PASS"
        elif diff <= 2 * tol:
            status = "WARN"
        else:
            status = "FAIL"
    rows.append(dict(anchor_name=name, expected_value=expected,
                     observed_value=None if observed is None else round(observed, 6),
                     tolerance=tol, absolute_difference=None if diff is None else round(diff, 6),
                     status=status, source_file_used=source, notes=notes))

js = json.loads((R / "_paper" / "defensive_stats.json").read_text())

# --- Anchor 1+2: OP3 leave-one-lineage-out FP-ridge gap (recompute from per-lineage rows) ---
f = pd.read_csv(R / "C5" / "loct_fine6.csv")
SIMPLE = ["ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"]
piv = f.pivot_table(index="split", columns="baseline", values="pearson_delta")
gap_floor = (piv["FP-ridge"] - piv[SIMPLE].max(axis=1)).mean()
gap_prim = (piv["FP-ridge"] - piv[["cell-mean", "donor-shift"]].max(axis=1)).mean()
rec("OP3 LOLO FP-ridge vs best-of-four floor (mean gap)", 0.119, float(gap_floor), 0.005,
    "results/C5/loct_fine6.csv", f"recomputed over {len(piv)} fine lineages; defensive_stats C5_cellcontext.mean_gap={js['C5_cellcontext']['mean_gap']:.4f}")
rec("OP3 LOLO FP-ridge vs primary (training-mean/donor) comparator", 0.123, float(gap_prim), 0.005,
    "results/C5/loct_fine6.csv", f"recomputed; defensive_stats vs_primary_floor_mean={js['C5_cellcontext'].get('vs_primary_floor_mean')}")

# --- Anchor 3+4: Kang LOLO scGen 8-lineage avg + CD14 monocyte ---
k = pd.read_csv(R / "C1" / "loct_scgen_forest.csv")
rec("Kang LOLO scGen 8-lineage average gap", -0.016, float(k["gap"].mean()), 0.005,
    "results/C1/loct_scgen_forest.csv", f"mean of per-lineage gap over {len(k)} lineages; {(k['gap']>0).sum()}/{len(k)} positive")
cd14 = k[k["lineage"].astype(str).str.contains("CD14|Mono_CD14|monocyte", case=False, regex=True)]
cd14_gap = float(cd14["gap"].max()) if len(cd14) else None
rec("Kang LOLO scGen CD14+ monocyte gap (seed-0)", 0.098, cd14_gap, 0.012,
    "results/C1/loct_scgen_forest.csv", f"row(s): {cd14['lineage'].tolist() if len(cd14) else 'NOT FOUND'}")

# --- Anchor 5+6: OP3 unseen compound FP-ridge vs no-chemistry baseline ---
c5 = pd.read_csv(R / "C5" / "results_raw.csv")
c5c = c5[c5["split"].astype(str).str.contains("compound", case=False)]
def val(b):
    s = c5c[c5c["baseline"] == b]["pearson_delta"]
    return float(s.iloc[0]) if len(s) else None
fp, base = val("FP-ridge"), val("cell-mean")
rec("OP3 unseen-compound FP-ridge score", 0.164, fp, 0.003, "results/C5/results_raw.csv", "C5_global_compound_holdout")
rec("OP3 unseen-compound no-chemistry baseline score", 0.172, base, 0.003, "results/C5/results_raw.csv", "cell-mean=donor-shift")
if fp is not None and base is not None:
    rec("OP3 unseen-compound FP-ridge minus no-chemistry gap", -0.008, fp - base, 0.003, "results/C5/results_raw.csv", "recomputed FP-ridge - baseline")

# --- Anchor 7: CRISPR leave-one-gene-out gap (load aggregate) ---
rec("CRISPR LOGO best-conditioned minus training-mean gap", -0.241, float(js["C3_perturbation"]["mean_gap"]), 0.005,
    "results/_paper/defensive_stats.json", f"C3_perturbation; cluster_ci={js['C3_perturbation']['cluster_ci']}; cells_cond_wins={js['C3_perturbation']['cells_cond_wins']}", loaded=True)

# --- Anchor 8+9+10: Soskic LODO (recompute Pearson; load E-dist/AUCell) ---
s = pd.read_csv(R / "C2" / "soskic_donor_axis.csv")
pv = s.pivot_table(index="donor", columns="model", values="pearson_delta")
prim = pv[["cell-mean", "donor-shift"]].max(axis=1)
soskic_pearson = float((pv["scGen"] - prim).mean())
pct_pos = float(((pv["scGen"] - prim) > 0).mean() * 100)
rec("Soskic LODO scGen minus primary baseline (Pearson-delta)", -0.123, soskic_pearson, 0.003,
    "results/C2/soskic_donor_axis.csv", f"recomputed over {pv.shape[0]} donors; {pct_pos:.1f}% donors positive")
bs = pd.read_csv(R / "C2" / "soskic_donor_bootstrap_summary.csv").set_index("metric")
rec("Soskic LODO E-distance donor-bootstrap gap (oriented)", -0.056, float(bs.loc["e_distance", "mean"]), 0.003,
    "results/C2/soskic_donor_bootstrap_summary.csv", "lower-better, sign-oriented so positive favours conditioning", loaded=True)
rec("Soskic LODO AUCell-delta-MAE donor-bootstrap gap (oriented)", -0.012, float(bs.loc["aucell_delta_mae", "mean"]), 0.003,
    "results/C2/soskic_donor_bootstrap_summary.csv", "lower-better, sign-oriented", loaded=True)

# --- Anchor 11: Kang random-split inflation (load aggregate) ---
rec("Kang random-split optimism inflation (mean)", 0.017, float(js["donor_inflation"]["mean"]), 0.003,
    "results/_paper/defensive_stats.json", f"donor_inflation; cluster_ci={js['donor_inflation']['cluster_ci']}; boot_pos_frac={js['donor_inflation']['boot_pos_frac']}; n={js['donor_inflation']['n_donors']}", loaded=True)

# --- Anchor 12+13: Frangieh protein-CITE (multiseed scGen; baseline from results_raw) ---
ms = pd.read_csv(R / "_paper" / "multiseed_scgen_summary.csv")
prot = ms[(ms["cluster"] == "C4") & (ms["modality"] == "protein-CITE")]
ps25 = float(prot[prot["frac"] == 25.0]["pearson_mean"].iloc[0])
ps50 = float(prot[prot["frac"] == 50.0]["pearson_mean"].iloc[0])
c4 = pd.read_csv(R / "C4" / "results_raw.csv")
pb25 = float(c4[(c4["modality"] == "protein-CITE") & (c4["baseline"] == "cell-mean")
                & (c4["split"].astype(str).str.contains("ko_25"))]["pearson_delta"].iloc[0])
rec("Frangieh CITE-protein scGen score (unseen KO, 25% multiseed)", 0.09, ps25, 0.05,
    "results/_paper/multiseed_scgen_summary.csv", f"protein-CITE 3-seed mean: 25%={ps25:.4f}, 50%={ps50:.4f} (seed-robust collapse)", loaded=True)
rec("Frangieh CITE-protein scGen minus baseline gap (25%)", -0.59, ps25 - pb25, 0.08,
    "results/_paper/multiseed_scgen_summary.csv + results/C4/results_raw.csv",
    f"protein scGen 25% {ps25:.4f} - baseline {pb25:.4f}", loaded=True)

# --- write reports ---
df = pd.DataFrame(rows)
df.to_csv(OUT / "anchor_reproduction_report.csv", index=False)

n_pass = (df["status"].isin(["PASS", "PASS_loaded"])).sum()
n_fail = (df["status"] == "FAIL").sum()
n_warn = (df["status"] == "WARN").sum()
gate = "PASS" if n_fail == 0 else "BLOCKED"
md = ["# Anchor reproduction report (Part 2)", "",
      f"Gate: **{gate}** ({n_pass}/{len(df)} PASS/PASS_loaded, {n_warn} WARN, {n_fail} FAIL).",
      "Integration into main manuscript tables/conclusions is permitted only if no anchor FAILs.", "",
      "| anchor | expected | observed | tol | abs_diff | status | source | notes |",
      "|---|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['anchor_name']} | {r['expected_value']} | {r['observed_value']} | {r['tolerance']} | "
              f"{r['absolute_difference']} | {r['status']} | `{r['source_file_used']}` | {r['notes']} |")
(OUT / "anchor_reproduction_report.md").write_text("\n".join(md) + "\n")

print(f"GATE: {gate}  ({n_pass} pass/loaded, {n_warn} warn, {n_fail} fail)")
print(df[["anchor_name", "expected_value", "observed_value", "status"]].to_string(index=False))
