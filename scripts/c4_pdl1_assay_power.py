#!/usr/bin/env python
"""C4 PD-L1 assay-power + RNA-vs-surface decoupling (hardens the flagship surface claim).

Every value computed from the ACTUAL deposited C4 data:
  - results/C4/cite_marker_recovery.csv  (per-surface-marker observed Δ mean/sd over held-KO strata)
  - data/C4/frangieh/*.h5ad via src/ivcbench loaders (matched RNA + surface CITE on the SAME cells)

PART 1 (assay power, surface only):
  Each marker's obsDelta is the per-held-KO observed surface shift (mean of held-KO cells - ctrl_mean),
  with n_held_KO strata (62 at 25% holdout, 124 at 50%). obsDelta_mean / obsDelta_sd are the mean/SD
  over those strata. So a Normal CI on the MEAN observed surface Δ is mean ± 1.96 * sd/sqrt(n).
  We compute that CI for every marker, flag whether it straddles 0 (effect indistinguishable from
  zero -> at/below the assay floor) and contrast PD-L1 (CD274) vs PD-1 (CD279) vs the recovered panel.
  We also quantify the assay-floor confound: a near-zero, low-|effect| marker has sign-match at chance
  (~0.5) by construction, demonstrated by regressing sign_match_frac on standardized effect size.

PART 2 (RNA-vs-surface decoupling, post-transcriptional buffering AS A RESULT):
  CD274 exists in BOTH the surface CITE panel AND the RNA matrix. We re-run the IDENTICAL leave-one-
  KO-gene-out split in RNA space and compare, per held-KO stratum, the CD274 mRNA Δ vs the CD274
  surface Δ on the same KOs. If mRNA is recoverable (moves, distinguishable from 0) while surface is
  not, the post-transcriptional buffering claim is demonstrated in-data, not cited.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
ROOT = Path(__file__).resolve().parents[1]
OUT_PAPER = ROOT / "results" / "_paper"
OUT_PAPER.mkdir(parents=True, exist_ok=True)

Z = 1.959963984540054  # 95% normal quantile

# ---------------------------------------------------------------------------------------------------
# PART 1 — surface-marker assay power from the deposited recovery table
# ---------------------------------------------------------------------------------------------------
rec = pd.read_csv(ROOT / "results" / "C4" / "cite_marker_recovery.csv")

# CI on the MEAN observed surface Δ over held-KO strata: mean ± Z * sd/sqrt(n)
rec["sem"] = rec["obsDelta_sd"] / np.sqrt(rec["n_held_KO"])
rec["ci_lo"] = rec["obsDelta_mean"] - Z * rec["sem"]
rec["ci_hi"] = rec["obsDelta_mean"] + Z * rec["sem"]
rec["straddles_zero"] = (rec["ci_lo"] <= 0) & (rec["ci_hi"] >= 0)
# standardized effect size of the per-KO surface shift (|mean| in SD-of-strata units)
rec["effect_sd_units"] = rec["obsDelta_mean"].abs() / rec["obsDelta_sd"]
# z-stat of the mean shift vs 0 (how many SEM from zero)
rec["z_vs_zero"] = rec["obsDelta_mean"] / rec["sem"]

rec.to_csv(OUT_PAPER / "c4_surface_marker_CIs.csv", index=False)


def grab(marker, frac):
    r = rec[(rec["marker"] == marker) & (rec["held_frac_pct"] == frac)].iloc[0]
    return r


summary = {}
for frac in (25, 50):
    sub = rec[rec["held_frac_pct"] == frac]
    pdl1 = grab("CD274", frac)
    pd1 = grab("CD279", frac)
    summary[f"frac{frac}"] = {
        "PD-L1_CD274": {
            "obs_mean": float(pdl1["obsDelta_mean"]), "sd": float(pdl1["obsDelta_sd"]),
            "sem": float(pdl1["sem"]), "ci": [float(pdl1["ci_lo"]), float(pdl1["ci_hi"])],
            "straddles_zero": bool(pdl1["straddles_zero"]),
            "effect_sd_units": float(pdl1["effect_sd_units"]), "z_vs_zero": float(pdl1["z_vs_zero"]),
            "sign_match_frac": float(pdl1["sign_match_frac"]), "n_held_KO": int(pdl1["n_held_KO"]),
        },
        "PD-1_CD279": {
            "obs_mean": float(pd1["obsDelta_mean"]), "sd": float(pd1["obsDelta_sd"]),
            "sem": float(pd1["sem"]), "ci": [float(pd1["ci_lo"]), float(pd1["ci_hi"])],
            "straddles_zero": bool(pd1["straddles_zero"]),
            "effect_sd_units": float(pd1["effect_sd_units"]), "z_vs_zero": float(pd1["z_vs_zero"]),
            "sign_match_frac": float(pd1["sign_match_frac"]), "n_held_KO": int(pd1["n_held_KO"]),
        },
        "n_markers_total": int(len(sub)),
        "n_markers_straddle_zero": int(sub["straddles_zero"].sum()),
        "markers_straddle_zero": sub[sub["straddles_zero"]]["marker"].tolist(),
    }

# Assay-floor confound: sign-match vs standardized effect size.
# Pearson correlation across all markers (both fracs pooled) between |effect| and sign_match_frac.
from scipy.stats import pearsonr
m = rec.dropna(subset=["sign_match_frac"])
r_eff, p_eff = pearsonr(m["effect_sd_units"], m["sign_match_frac"])
# chance band: markers whose CI straddles zero -> sign match should sit near 0.5
near_floor = rec[rec["straddles_zero"]]
recovered = rec[~rec["straddles_zero"]]
summary["assay_floor_confound"] = {
    "pearson_effect_vs_signmatch": float(r_eff), "p": float(p_eff), "n": int(len(m)),
    "mean_signmatch_near_floor": float(near_floor["sign_match_frac"].mean()),
    "mean_signmatch_recovered": float(recovered["sign_match_frac"].mean()),
    "mean_effect_near_floor": float(near_floor["effect_sd_units"].mean()),
    "mean_effect_recovered": float(recovered["effect_sd_units"].mean()),
}

print("=== PART 1: surface-marker assay power (CI on observed Δ) ===")
for frac in (25, 50):
    s = summary[f"frac{frac}"]
    print(f"\n-- holdout {frac}% (n_held_KO={s['PD-L1_CD274']['n_held_KO']}) --")
    for nm in ("PD-L1_CD274", "PD-1_CD279"):
        d = s[nm]
        print(f"  {nm:12s} Δ={d['obs_mean']:+.4f}  95%CI=[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]"
              f"  straddles0={d['straddles_zero']}  |eff|={d['effect_sd_units']:.3f}sd"
              f"  z={d['z_vs_zero']:+.2f}  signmatch={d['sign_match_frac']:.3f}")
    print(f"  markers w/ CI straddling 0 ({s['n_markers_straddle_zero']}/{s['n_markers_total']}):"
          f" {s['markers_straddle_zero']}")
af = summary["assay_floor_confound"]
print(f"\n  ASSAY-FLOOR CONFOUND: corr(|effect|, sign_match) = {af['pearson_effect_vs_signmatch']:.3f}"
      f" (p={af['p']:.2e}, n={af['n']})")
print(f"    near-floor markers: mean sign_match={af['mean_signmatch_near_floor']:.3f}"
      f" at mean |eff|={af['mean_effect_near_floor']:.3f}sd")
print(f"    recovered markers:  mean sign_match={af['mean_signmatch_recovered']:.3f}"
      f" at mean |eff|={af['mean_effect_recovered']:.3f}sd")

# ---------------------------------------------------------------------------------------------------
# PART 2 — RNA-vs-surface decoupling on the SAME cells, IDENTICAL held-KO split
# ---------------------------------------------------------------------------------------------------
from ivcbench.data.loaders.frangieh import load
from ivcbench.clusters import c4
from ivcbench.splits.builder import build_split

# surface markers that have an unambiguous RNA gene-symbol counterpart
SURFACE_TO_GENE = {
    "CD274": "CD274", "HLA_A": "HLA-A", "HLA_E": "HLA-E", "CD58": "CD58", "CD59": "CD59",
    "CD47": "CD47", "CD119": "IFNGR1", "CD44": "CD44", "CD29": "ITGB1", "CD117": "KIT",
    "CD9": "CD9", "CD61": "ITGB3", "CD49f": "ITGA6", "CD184": "CXCR4", "CD172a": "SIRPA",
    "CD140a": "PDGFRA", "CD140b": "PDGFRB", "CD202b": "TEK", "CD309": "KDR",
    # CD279 (PD-1) gene PDCD1 is a T-cell receptor, not expressed by melanoma -> handled separately
}

print("\n=== PART 2: RNA-vs-surface decoupling (same cells, identical split) ===")

# Load both modalities (SAME loader, SAME condition, SAME seed -> matched cells per perturbation)
cs_prot = load(modality="protein")
cs_rna = load(modality="rna")
prot_markers = list(cs_prot.var_names)
rna_genes = list(cs_rna.var_names)
print(f"  protein CellSet: {cs_prot.n_cells} cells x {cs_prot.n_genes} markers")
print(f"  RNA CellSet:     {cs_rna.n_cells} cells x {cs_rna.n_genes} HVGs")

g_prot = cs_prot.uns["genes_perturbed"]
g_rna = cs_rna.uns["genes_perturbed"]


def per_ko_deltas(cs, frac_label, frac):
    """Replicate c4_marker_deposit exactly: per-held-KO Δ = mean(KO cells) - ctrl_mean."""
    held = c4.held_ko_fraction(cs.uns["genes_perturbed"], frac, seed=0)
    spec = c4.modality_lo_ko(held, frac_label)
    sp_ = build_split(cs, spec)
    ctrl_mean = (cs.X[sp_.inference_input_idx].mean(0) if len(sp_.inference_input_idx)
                 else cs.X[sp_.train_idx[cs.obs.iloc[sp_.train_idx]["is_control"].to_numpy()]].mean(0))
    strata = sp_.test_strata
    test_X = cs.X[sp_.test_idx]
    uniq = np.unique(strata)
    obs_delta = np.vstack([test_X[strata == s].mean(0) - ctrl_mean for s in uniq])  # (n_KO, n_feat)
    return uniq, obs_delta, ctrl_mean


decouple_rows = []
panel_summary = {}
for frac, lbl in [(0.25, "25"), (0.50, "50")]:
    fp = int(lbl)
    uniq_p, dprot, _ = per_ko_deltas(cs_prot, lbl, frac)
    uniq_r, drna, _ = per_ko_deltas(cs_rna, lbl, frac)
    # held KOs should be identical (same seed, same gene set) — align on stratum label
    assert list(uniq_p) == list(uniq_r), "held-KO strata differ between modalities!"
    n = len(uniq_p)

    for surf, gene in SURFACE_TO_GENE.items():
        if surf not in prot_markers or gene not in rna_genes:
            continue
        jp = prot_markers.index(surf)
        jr = rna_genes.index(gene)
        sp_d = dprot[:, jp]
        rn_d = drna[:, jr]
        sp_mean, sp_sd = float(sp_d.mean()), float(sp_d.std())
        rn_mean, rn_sd = float(rn_d.mean()), float(rn_d.std())
        sp_sem, rn_sem = sp_sd / np.sqrt(n), rn_sd / np.sqrt(n)
        row = {
            "held_frac_pct": fp, "surface_marker": surf, "rna_gene": gene, "n_held_KO": n,
            "surf_mean": sp_mean, "surf_sd": sp_sd,
            "surf_ci_lo": sp_mean - Z * sp_sem, "surf_ci_hi": sp_mean + Z * sp_sem,
            "surf_straddles0": (sp_mean - Z * sp_sem <= 0 <= sp_mean + Z * sp_sem),
            "surf_z": sp_mean / sp_sem if sp_sem else np.nan,
            "rna_mean": rn_mean, "rna_sd": rn_sd,
            "rna_ci_lo": rn_mean - Z * rn_sem, "rna_ci_hi": rn_mean + Z * rn_sem,
            "rna_straddles0": (rn_mean - Z * rn_sem <= 0 <= rn_mean + Z * rn_sem),
            "rna_z": rn_mean / rn_sem if rn_sem else np.nan,
            # paired across KOs: correlation of the two Δ vectors over the SAME held KOs
            "paired_pearson": float(np.corrcoef(sp_d, rn_d)[0, 1]),
        }
        decouple_rows.append(row)

    # focused CD274 print
    cd = [r for r in decouple_rows if r["surface_marker"] == "CD274" and r["held_frac_pct"] == fp][0]
    print(f"\n-- holdout {fp}% (n_held_KO={n}) — CD274 (PD-L1) surface vs mRNA --")
    print(f"   surface CD274 Δ = {cd['surf_mean']:+.4f}  95%CI=[{cd['surf_ci_lo']:+.4f},"
          f"{cd['surf_ci_hi']:+.4f}]  straddles0={cd['surf_straddles0']}  z={cd['surf_z']:+.2f}")
    print(f"   mRNA   CD274 Δ = {cd['rna_mean']:+.4f}  95%CI=[{cd['rna_ci_lo']:+.4f},"
          f"{cd['rna_ci_hi']:+.4f}]  straddles0={cd['rna_straddles0']}  z={cd['rna_z']:+.2f}")
    print(f"   paired Pearson(surface Δ, mRNA Δ) over the {n} held KOs = {cd['paired_pearson']:+.3f}")

dec = pd.DataFrame(decouple_rows)
dec.to_csv(OUT_PAPER / "c4_rna_vs_surface_decoupling.csv", index=False)

# Panel-wide: how many markers are mRNA-recoverable (CI off 0) but surface-not?
for fp in (25, 50):
    sub = dec[dec["held_frac_pct"] == fp]
    both_off = sub[~sub["surf_straddles0"] & ~sub["rna_straddles0"]]
    rna_only = sub[sub["surf_straddles0"] & ~sub["rna_straddles0"]]
    surf_only = sub[~sub["surf_straddles0"] & sub["rna_straddles0"]]
    panel_summary[f"frac{fp}"] = {
        "n_markers_with_rna_match": int(len(sub)),
        "n_surface_moves": int((~sub["surf_straddles0"]).sum()),
        "n_rna_moves": int((~sub["rna_straddles0"]).sum()),
        "n_both_move": int(len(both_off)),
        "n_rna_only_decoupled": int(len(rna_only)),
        "rna_only_markers": rna_only["surface_marker"].tolist(),
        "n_surface_only": int(len(surf_only)),
        "surface_only_markers": surf_only["surface_marker"].tolist(),
    }

print("\n=== PANEL-WIDE decoupling summary ===")
for fp in (25, 50):
    p = panel_summary[f"frac{fp}"]
    print(f"  {fp}%: of {p['n_markers_with_rna_match']} markers w/ RNA counterpart -> "
          f"surface moves {p['n_surface_moves']}, mRNA moves {p['n_rna_moves']}, "
          f"mRNA-only-decoupled {p['n_rna_only_decoupled']}: {p['rna_only_markers']}")

# dump combined summary json
def _pyify(o):
    if isinstance(o, dict):
        return {k: _pyify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_pyify(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o

with open(OUT_PAPER / "c4_pdl1_assay_power_summary.json", "w") as fh:
    json.dump(_pyify({"part1_surface_power": summary, "part2_decoupling": panel_summary,
                      "cd274_decouple_rows": [r for r in decouple_rows
                                              if r["surface_marker"] == "CD274"]}),
              fh, indent=2)
print(f"\nWROTE: {OUT_PAPER/'c4_surface_marker_CIs.csv'}")
print(f"WROTE: {OUT_PAPER/'c4_rna_vs_surface_decoupling.csv'}")
print(f"WROTE: {OUT_PAPER/'c4_pdl1_assay_power_summary.json'}")
