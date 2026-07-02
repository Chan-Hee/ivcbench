#!/usr/bin/env python
"""Supplementary figure: PD-L1 assay power + RNA-vs-surface decoupling (C4 Frangieh CITE).

Navy editorial style (src/ivcbench/report/style.py). Three panels:
  (a) FOREST — per-surface-marker observed Δ (50% holdout) with 95% CIs over held-KO strata, ranked by
      effect; the assay-floor band (markers whose CI straddles 0) shaded; PD-1 (CD279) recovered (CI
      cleanly negative) vs PD-L1 (CD274) at/below the assay floor (CI straddles 0).
  (b) ASSAY-FLOOR CONFOUND — sign-match vs standardized effect size across all markers/holdouts;
      chance line at 0.5; near-zero markers sit at chance BY CONSTRUCTION (PD-L1 there, PD-1 at ceiling).
  (c) DECOUPLING — CD274 surface Δ vs mRNA Δ (same cells, identical split), each with 95% CI; surface
      straddles 0 while mRNA moves cleanly = post-transcriptional buffering shown in-data.

All numbers read from results/_paper/c4_*.csv (produced by scripts/c4_pdl1_assay_power.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, "src")
from ivcbench.report.style import (set_pub_style, despine, panel_title, style_legend,
                                   CITE_COLORS, NAVY, NAVY_DARK, GREY_MID, GREY_LITE, NULL_GREY,
                                   INK, SIMPLE_GREY, LEGEND_EC)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results" / "_paper"
Z = 1.959963984540054

C_REC = CITE_COLORS["PD-1"]    # PD-1 (CD279) recovered — teal-green
C_LIG = CITE_COLORS["PD-L1"]   # PD-L1 (CD274) at/below floor — rose
C_FLOOR = SIMPLE_GREY          # other markers — floor grey
FS_LAB = 7.0


def _u(x):
    """unicode-minus a numeric string."""
    return x.replace("-", "−")


def fmt(v, d=2):
    return _u(f"{v:.{d}f}")


def main():
    set_pub_style()
    rec = pd.read_csv(PAPER / "c4_surface_marker_CIs.csv")
    dec = pd.read_csv(PAPER / "c4_rna_vs_surface_decoupling.csv")

    fig = plt.figure(figsize=(11.2, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.0, 0.9], wspace=0.42,
                          left=0.075, right=0.985, top=0.84, bottom=0.135)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # ============================ (a) FOREST of observed surface Δ + 95% CI ======================
    # 50% holdout (larger n=124, more stringent) as deposited.
    t = rec[rec.held_frac_pct == 50].copy()
    t["lab"] = t["alias"].map(lambda a: a.split(" (")[0])
    t = t.sort_values("obsDelta_mean").reset_index(drop=True)  # most negative at bottom
    yA = np.arange(len(t))

    def mcol(lab):
        return C_REC if lab == "PD-1" else (C_LIG if lab == "PD-L1" else C_FLOOR)

    axA.axvline(0.0, color=GREY_MID, lw=0.7, zorder=1)
    # shade the assay-floor zone: markers whose CI straddles 0
    floor_rows = [i for i, r in enumerate(t.itertuples()) if r.straddles_zero]
    if floor_rows:
        # contiguous-or-not, draw one faint span per row to mark the floor cluster
        for i in floor_rows:
            axA.axhspan(i - 0.42, i + 0.42, color=NULL_GREY, alpha=0.16, zorder=0)

    for yi, r in zip(yA, t.itertuples()):
        col = mcol(r.lab)
        is_chk = r.lab in ("PD-1", "PD-L1")
        lw = 2.0 if is_chk else 1.1
        ms = 5.4 if is_chk else 3.0
        a = 0.97 if is_chk else 0.55
        # 95% CI whisker
        axA.plot([r.ci_lo, r.ci_hi], [yi, yi], color=col, lw=lw, alpha=a,
                 solid_capstyle="round", zorder=2)
        # cap ticks
        for xc in (r.ci_lo, r.ci_hi):
            axA.plot([xc, xc], [yi - 0.16, yi + 0.16], color=col, lw=lw * 0.7, alpha=a, zorder=2)
        # point estimate
        axA.plot(r.obsDelta_mean, yi, "o", color=col, ms=ms, mec="white", mew=0.7, zorder=4)

    axA.set_yticks(yA)
    axA.set_yticklabels(t["lab"], fontsize=FS_LAB)
    for tick, lab in zip(axA.get_yticklabels(), t["lab"]):
        if lab == "PD-1":
            tick.set_color(C_REC); tick.set_fontweight("bold")
        elif lab == "PD-L1":
            tick.set_color(C_LIG); tick.set_fontweight("bold")
    axA.set_xlim(-0.60, 0.34)
    axA.set_ylim(-0.9, len(t) - 0.3)
    xa = [-0.6, -0.4, -0.2, 0.0, 0.2]
    axA.set_xticks(xa)
    axA.set_xticklabels([_u(f"{x:.1f}") for x in xa], fontsize=FS_LAB)
    axA.set_xlabel(r"observed surface $\Delta$  (mean $\pm$ 95% CI over held KOs)",
                   fontsize=7.4, color=INK)
    axA.tick_params(colors=INK, labelsize=FS_LAB)
    despine(axA)
    panel_title(axA, "a", "Per-marker surface response", sub="Frangieh CITE, 50% held-KO (n=124)")

    # checkpoint callouts
    yrec = int(np.where(t["lab"].values == "PD-1")[0][0])
    ylig = int(np.where(t["lab"].values == "PD-L1")[0][0])
    rrec = t[t.lab == "PD-1"].iloc[0]
    rlig = t[t.lab == "PD-L1"].iloc[0]
    axA.annotate(f"receptor recovered\n95% CI {fmt(rrec.ci_lo)}..{fmt(rrec.ci_hi)}  (z={fmt(rrec.z_vs_zero,1)})",
                 xy=(rrec.obsDelta_mean, yrec), xytext=(-0.27, yrec + 1.05),
                 fontsize=6.2, color=C_REC, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=C_REC, lw=0.7, shrinkA=2, shrinkB=3))
    axA.annotate(f"ligand at floor\n95% CI {fmt(rlig.ci_lo)}..{fmt(rlig.ci_hi)}  (crosses 0)",
                 xy=(rlig.obsDelta_mean, ylig), xytext=(0.085, ylig - 0.05),
                 fontsize=6.2, color=C_LIG, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=C_LIG, lw=0.7, shrinkA=3, shrinkB=2))

    # ============================ (b) assay-floor confound ======================================
    m = rec.dropna(subset=["sign_match_frac"]).copy()
    r_eff, p_eff = pearsonr(m["effect_sd_units"], m["sign_match_frac"])
    axB.axhline(0.5, color=GREY_MID, lw=0.8, ls=(0, (4, 3)), zorder=1)
    axB.text(m["effect_sd_units"].max() * 0.99, 0.5, "  chance (0.5)", fontsize=6.4,
             color=GREY_MID, va="bottom", ha="right")

    def pcol(mk):
        return C_LIG if mk == "CD274" else (C_REC if mk == "CD279" else C_FLOOR)

    for r in m.itertuples():
        is_chk = r.marker in ("CD274", "CD279")
        axB.plot(r.effect_sd_units, r.sign_match_frac, "o",
                 color=pcol(r.marker), ms=6.0 if is_chk else 3.4,
                 mec="white" if is_chk else "none", mew=0.7,
                 alpha=0.98 if is_chk else 0.5, zorder=4 if is_chk else 3)
    # trend line
    xs = np.linspace(0, m["effect_sd_units"].max(), 50)
    b1, b0 = np.polyfit(m["effect_sd_units"], m["sign_match_frac"], 1)
    axB.plot(xs, b0 + b1 * xs, color=NAVY, lw=1.3, alpha=0.85, zorder=2)
    axB.set_xlabel(r"standardized effect size  $|\Delta| / \mathrm{SD}_{\mathrm{KO}}$",
                   fontsize=7.4, color=INK)
    axB.set_ylabel("direction recovery  (sign-match frac)", fontsize=7.4, color=INK)
    axB.set_ylim(0.30, 1.02)
    axB.set_xlim(-0.05, m["effect_sd_units"].max() * 1.06)
    axB.tick_params(colors=INK, labelsize=FS_LAB)
    despine(axB)
    panel_title(axB, "b", "Assay-floor confound",
                sub="near-zero markers match sign at chance")
    # annotate the two checkpoints. PD-L1's lowest point (sign_match_frac ~0.355) sits close to
    # the panel floor (ylim=0.30), so its label is placed ABOVE the marker (not below, which used
    # to push the text past the axis and clip it); PD-1 keeps its existing above offset.
    for mk, lab, c, dy in [("CD274", "PD-L1", C_LIG, 0.05), ("CD279", "PD-1", C_REC, 0.04)]:
        rr = m[m.marker == mk].sort_values("held_frac_pct").iloc[0]
        axB.annotate(lab, xy=(rr.effect_sd_units, rr.sign_match_frac),
                     xytext=(rr.effect_sd_units + 0.06, rr.sign_match_frac + dy),
                     fontsize=6.6, fontweight="bold", color=c, va="center")
    axB.text(0.96, 0.06, f"r = {fmt(r_eff)},  p = {p_eff:.0e}".replace("e-", "e−"),
             transform=axB.transAxes, fontsize=6.6, color=NAVY_DARK, ha="right", va="bottom")

    # ============================ (c) RNA-vs-surface decoupling (CD274) ==========================
    cd = dec[dec.surface_marker == "CD274"].sort_values("held_frac_pct")
    rows = []  # (label, mean, lo, hi, color)
    for r in cd.itertuples():
        tag = f"{r.held_frac_pct}%"
        rows.append((f"surface  {tag}", r.surf_mean, r.surf_ci_lo, r.surf_ci_hi, C_LIG, r.surf_z))
        rows.append((f"mRNA  {tag}", r.rna_mean, r.rna_ci_lo, r.rna_ci_hi, NAVY, r.rna_z))
    # order: surface 25, mRNA 25, surface 50, mRNA 50 -> plot top-down
    yC = np.arange(len(rows))[::-1]
    axC.axvline(0.0, color=GREY_MID, lw=0.7, zorder=1)
    for yi, (lab, mean, lo, hi, col, zz) in zip(yC, rows):
        axC.plot([lo, hi], [yi, yi], color=col, lw=2.2, solid_capstyle="round", zorder=2)
        for xc in (lo, hi):
            axC.plot([xc, xc], [yi - 0.13, yi + 0.13], color=col, lw=1.5, zorder=2)
        axC.plot(mean, yi, "o", color=col, ms=6.0, mec="white", mew=0.8, zorder=4)
        axC.text(0.995, yi, f"z={fmt(zz,1)}", transform=axC.get_yaxis_transform(),
                 fontsize=6.3, color=col, ha="right", va="center", zorder=5)
    axC.set_yticks(yC)
    axC.set_yticklabels([r[0] for r in rows], fontsize=FS_LAB)
    for tick, r in zip(axC.get_yticklabels(), rows):
        tick.set_color(r[4]);
        if r[0].startswith("surface"): tick.set_fontweight("bold")
    axC.set_ylim(-0.7, len(rows) - 0.3)
    # xlim right bound widened past the widest CI whisker (surface 25%, hi=0.079) so the
    # per-row "z=" label (anchored near the right spine) never overprints the whisker tick.
    axC.set_xlim(-0.10, 0.115)
    xc_t = [-0.08, -0.04, 0.0, 0.04]
    axC.set_xticks(xc_t)
    axC.set_xticklabels([_u(f"{x:.2f}") for x in xc_t], fontsize=FS_LAB)
    axC.set_xlabel(r"CD274 response  ($\Delta$, same cells)", fontsize=7.4, color=INK)
    axC.tick_params(colors=INK, labelsize=FS_LAB)
    despine(axC)
    panel_title(axC, "c", "PD-L1: mRNA moves, surface does not",
                sub="post-transcriptional buffering, in-data")

    fig.suptitle("PD-L1 assay power and RNA-vs-surface decoupling (T4 Frangieh Perturb-CITE-seq)",
                 fontsize=10.5, fontweight="bold", color=NAVY_DARK, x=0.075, ha="left", y=0.965)

    out = PAPER / "figS_c4_pdl1_assay_power.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.10, facecolor="white")
    print("WROTE", out)
    print("WROTE", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
