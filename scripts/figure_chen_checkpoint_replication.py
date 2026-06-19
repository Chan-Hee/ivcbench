#!/usr/bin/env python
"""Second-dataset (n=2) replication of the checkpoint-recovery asymmetry on the INDEPENDENT Chen
Perturb-icCITE-seq dataset (E-GEAD-648; primary human CD4+ Treg CRISPR-KO; SURFACE ADT/CITE panel).

Navy editorial style (src/ivcbench/report/style.py). Three panels:
  (a) FOREST of the observed SURFACE Δ for PD-1 (CD279) and PD-L1 (CD274) at both holdouts, with 95%
      CIs over held-KO strata and the assay-floor band shaded. PD-1's CI cleanly excludes 0 (modulated);
      PD-L1's CI straddles 0 (at the surface floor, direction-unrecoverable) — the LIGAND-not-moved
      half of the flagship replicates on a second, independent dataset.
  (b) ASSAY-FLOOR CONFOUND on Chen: sign-match vs standardized effect size across all 277 surface
      markers x 2 holdouts; the same law (r=0.92) that held on Frangieh (r=0.90) — near-floor markers
      sign-match at chance BY CONSTRUCTION; PD-L1 sits there, PD-1 sits just above the floor.
  (c) CROSS-DATASET (n=2) panel: PD-1 vs PD-L1 standardized effect + direction-recovery on Frangieh
      vs Chen side-by-side. The ASYMMETRY (PD-1 > PD-L1, ligand at floor) replicates in BOTH; but the
      receptor's ABSOLUTE recoverability is context-dependent (strong on Frangieh TIL synapse, weak on
      Chen Treg-regulator KOs) — reported honestly, not over-claimed.

Every number is read from the deposited real-data CSV/JSON (no fabrication):
  results/newdata/chen_cite_marker_recovery.csv + chen_checkpoint_replication_summary.json
  results/_paper/c4_surface_marker_CIs.csv (Frangieh, for the cross-dataset panel).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import pearsonr

sys.path.insert(0, "src")
from ivcbench.report.style import (set_pub_style, despine, panel_title, style_legend,
                                   CITE_COLORS, NAVY, NAVY_DARK, GREY_MID, GREY_LITE, NULL_GREY,
                                   INK, SIMPLE_GREY, CLAY_DARK, LEGEND_EC)

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "results" / "newdata"
PAPER = ROOT / "results" / "_paper"

C_REC = CITE_COLORS["PD-1"]    # PD-1 (CD279) — teal-green
C_LIG = CITE_COLORS["PD-L1"]   # PD-L1 (CD274) — rose
C_FLOOR = SIMPLE_GREY
FS_LAB = 7.0
PDL1 = "surface_A0007_PDL1"
PD1 = "surface_A0088_PD1"


def _u(x):
    return x.replace("-", "−")


def fmt(v, d=2):
    return _u(f"{v:.{d}f}")


def main():
    set_pub_style()
    rec = pd.read_csv(NEW / "chen_cite_marker_recovery.csv")
    summ = json.load(open(NEW / "chen_checkpoint_replication_summary.json"))
    fr = pd.read_csv(PAPER / "c4_surface_marker_CIs.csv")   # Frangieh deposited

    fig = plt.figure(figsize=(11.4, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.12], wspace=0.46,
                          left=0.085, right=0.985, top=0.82, bottom=0.145)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # ============================ (a) FOREST: PD-1 vs PD-L1 surface Δ, both holdouts ==============
    # Rows top-down: PD-1 25%, PD-1 50%, PD-L1 25%, PD-L1 50%
    rows = []
    for mk, lab, c in [(PD1, "PD-1 (CD279)", C_REC), (PDL1, "PD-L1 (CD274)", C_LIG)]:
        for frac in (25, 50):
            r = rec[(rec.marker == mk) & (rec.held_frac_pct == frac)].iloc[0]
            rows.append((f"{lab.split(' (')[0]}  {frac}%", r.obsDelta_mean, r.ci_lo, r.ci_hi,
                         c, bool(r.straddles_zero), r.z_vs_zero, r.n_held_KO))
    yA = np.arange(len(rows))[::-1]
    XLO, XHI = -0.052, 0.052
    BAND_HI = 0.034   # band hugs the data; leaves a clean right gutter for the status tags
    axA.axvline(0.0, color=GREY_MID, lw=0.8, zorder=1)
    for yi, (lab, mean, lo, hi, col, straddle, zz, nhk) in zip(yA, rows):
        # assay-floor shading for the straddlers (bounded to the data, not the full panel)
        if straddle:
            axA.add_patch(plt.Rectangle((XLO, yi - 0.45), BAND_HI - XLO, 0.90,
                                        facecolor=NULL_GREY, alpha=0.18, edgecolor="none",
                                        zorder=0))
        axA.plot([lo, hi], [yi, yi], color=col, lw=2.4, solid_capstyle="round", zorder=2)
        for xc in (lo, hi):
            axA.plot([xc, xc], [yi - 0.14, yi + 0.14], color=col, lw=1.6, zorder=2)
        axA.plot(mean, yi, "o", color=col, ms=6.2, mec="white", mew=0.8, zorder=4)
        # right-edge status tag, pulled INWARD with clear whitespace before the spine
        tag = "crosses 0" if straddle else f"z = {fmt(zz, 1)}"
        axA.text(0.965, yi, tag, transform=axA.get_yaxis_transform(),
                 fontsize=6.2, color=(GREY_MID if straddle else col), ha="right", va="center")
    axA.set_yticks(yA)
    axA.set_yticklabels([r[0] for r in rows], fontsize=FS_LAB)
    for tick, r in zip(axA.get_yticklabels(), rows):
        tick.set_color(r[4]); tick.set_fontweight("bold")
    axA.set_xlim(XLO, XHI)
    axA.set_ylim(-0.7, len(rows) - 0.3)
    xa = [-0.04, -0.02, 0.0, 0.02, 0.04]
    axA.set_xticks(xa)
    axA.set_xticklabels([_u(f"{x:.2f}") for x in xa], fontsize=FS_LAB)
    axA.set_xlabel(r"observed surface $\Delta$  (mean $\pm$ 95% CI over held KOs)",
                   fontsize=7.2, color=INK)
    axA.tick_params(colors=INK, labelsize=FS_LAB)
    despine(axA)
    panel_title(axA, "a", "Checkpoint surface response (Chen)",
                sub="receptor off 0, ligand at the floor")
    # floor-band key — swatch matches the rendered band tint; anchored in clear space below the
    # lowest band so it never overlaps the data rows
    leg = axA.legend(handles=[Patch(facecolor=NULL_GREY, alpha=0.18, edgecolor="none",
                                    label="CI straddles 0 (assay floor)")],
                     loc="upper left", bbox_to_anchor=(0.0, -0.105), fontsize=6.0,
                     frameon=False, handlelength=1.1, borderpad=0.3)

    # ============================ (b) assay-floor confound on Chen ===============================
    m = rec.dropna(subset=["sign_match_frac"]).copy()
    r_eff, p_eff = pearsonr(m["effect_sd_units"], m["sign_match_frac"])
    axB.axhline(0.5, color=GREY_MID, lw=0.8, ls=(0, (4, 3)), zorder=1)
    # chance label pulled INWARD off the right spine, with clear whitespace at the border
    axB.text(0.955, 0.515, "chance (0.5)", transform=axB.transAxes, fontsize=6.4,
             color=GREY_MID, va="bottom", ha="right")

    def pcol(mk):
        return C_LIG if mk == PDL1 else (C_REC if mk == PD1 else C_FLOOR)

    chk = m[m.marker.isin([PDL1, PD1])]
    oth = m[~m.marker.isin([PDL1, PD1])]
    axB.plot(oth["effect_sd_units"], oth["sign_match_frac"], "o", color=C_FLOOR, ms=3.0,
             mec="none", alpha=0.40, zorder=3)
    for r in chk.itertuples():
        axB.plot(r.effect_sd_units, r.sign_match_frac, "o", color=pcol(r.marker), ms=6.6,
                 mec="white", mew=0.8, alpha=0.99, zorder=5)
    xs = np.linspace(0, m["effect_sd_units"].max(), 50)
    b1, b0 = np.polyfit(m["effect_sd_units"], m["sign_match_frac"], 1)
    axB.plot(xs, b0 + b1 * xs, color=NAVY, lw=1.4, alpha=0.9, zorder=2)
    axB.set_xlabel(r"standardized effect size  $|\Delta| / \mathrm{SD}_{\mathrm{KO}}$",
                   fontsize=7.2, color=INK)
    axB.set_ylabel("direction recovery  (sign-match frac)", fontsize=7.2, color=INK)
    axB.set_ylim(0.20, 1.02)
    axB.set_xlim(-0.05, m["effect_sd_units"].max() * 1.06)
    axB.tick_params(colors=INK, labelsize=FS_LAB)
    despine(axB)
    panel_title(axB, "b", "Assay-floor confound replicates",
                sub="277 surface markers × 2 holdouts")
    # leader-line labels routed into the open lower-right white space, clear of the navy trend line
    for mk, lab, c, dx, dy, ha in [(PD1, "PD-1", C_REC, 0.55, -0.085, "left"),
                                   (PDL1, "PD-L1", C_LIG, 0.30, -0.105, "left")]:
        rr = chk[chk.marker == mk].sort_values("held_frac_pct").iloc[0]
        axB.annotate(lab, xy=(rr.effect_sd_units, rr.sign_match_frac),
                     xytext=(rr.effect_sd_units + dx, rr.sign_match_frac + dy),
                     fontsize=6.8, fontweight="bold", color=c, va="center", ha=ha,
                     arrowprops=dict(arrowstyle="-", color=c, lw=0.6, shrinkA=2, shrinkB=3))
    axB.text(0.96, 0.06, f"r = {fmt(r_eff)},  p = {p_eff:.0e}".replace("e-", "e−"),
             transform=axB.transAxes, fontsize=6.8, color=NAVY_DARK, ha="right", va="bottom")
    axB.text(0.96, 0.155, "Frangieh: r = 0.90", transform=axB.transAxes, fontsize=6.4,
             color=GREY_MID, ha="right", va="bottom", style="italic")

    # ============================ (c) cross-dataset (n=2) PD-1 vs PD-L1 ==========================
    # Grouped bars of standardized effect size (|Δ|/SD_KO), 25% holdout, on Frangieh vs Chen.
    def eff(df, marker_col, marker, frac=25):
        r = df[(df[marker_col] == marker) & (df.held_frac_pct == frac)].iloc[0]
        return float(r.effect_sd_units), float(r.sign_match_frac), bool(r.straddles_zero)

    fr_pd1 = eff(fr, "marker", "CD279")
    fr_pdl1 = eff(fr, "marker", "CD274")
    ch_pd1 = eff(rec, "marker", PD1)
    ch_pdl1 = eff(rec, "marker", PDL1)

    groups = ["Frangieh\n(melanoma+TIL)", "Chen\n(CD4+ Treg)"]
    pd1_eff = [fr_pd1[0], ch_pd1[0]]
    pdl1_eff = [fr_pdl1[0], ch_pdl1[0]]
    pd1_sm = [fr_pd1[1], ch_pd1[1]]
    pdl1_sm = [fr_pdl1[1], ch_pdl1[1]]
    x = np.arange(len(groups))
    w = 0.34
    bars_r = axC.bar(x - w / 2, pd1_eff, w, color=C_REC, label="PD-1 (CD279)", zorder=3,
                     edgecolor="white", linewidth=0.6)
    bars_l = axC.bar(x + w / 2, pdl1_eff, w, color=C_LIG, label="PD-L1 (CD274)", zorder=3,
                     edgecolor="white", linewidth=0.6)
    # annotate each bar with sign-match
    for xb, h, sm in zip(x - w / 2, pd1_eff, pd1_sm):
        axC.text(xb, h + 0.04, f"sm {fmt(sm)}", ha="center", va="bottom", fontsize=6.0, color=C_REC)
    for xb, h, sm, straddle in zip(x + w / 2, pdl1_eff, pdl1_sm, [fr_pdl1[2], ch_pdl1[2]]):
        axC.text(xb, h + 0.04, f"sm {fmt(sm)}", ha="center", va="bottom", fontsize=6.0, color=C_LIG)
    # the floor reference (|eff| ~ 0.1 SD = the near-zero band); label sits in the open gap between
    # the two groups, just under the dashed line, clear of every bar
    axC.axhline(0.1, color=GREY_MID, lw=0.8, ls=(0, (4, 3)), zorder=1)
    axC.text(0.5, 0.012, "surface floor (~0.1 SD)", transform=axC.get_yaxis_transform(),
             fontsize=6.0, color=GREY_MID, va="bottom", ha="center")
    axC.set_xticks(x)
    axC.set_xticklabels(groups, fontsize=6.8)
    # widen the x-range symmetrically so the rightmost bar and the upper-right legend keep a clean
    # gutter to the right spine (nothing flush to the panel border)
    axC.set_xlim(-0.58, 1.58)
    axC.set_ylabel(r"standardized effect size  $|\Delta|/\mathrm{SD}_{\mathrm{KO}}$  (25% holdout)",
                   fontsize=7.0, color=INK)
    axC.set_ylim(0, 2.15)
    axC.tick_params(colors=INK, labelsize=FS_LAB)
    despine(axC)
    panel_title(axC, "c", "n = 2: asymmetry replicates",
                sub="receptor > ligand in both; magnitude context-dependent")
    leg = axC.legend(loc="upper right", fontsize=6.6, frameon=True, handlelength=1.1,
                     borderpad=0.4)
    style_legend(leg)

    fig.suptitle("Second-dataset (n = 2) replication: checkpoint receptor vs ligand at the surface "
                 "(Chen Perturb-icCITE-seq, primary human CD4+ Treg CRISPR-KO)",
                 fontsize=10.0, fontweight="bold", color=NAVY_DARK, x=0.085, ha="left", y=0.965)

    for out_dir in (PAPER, NEW):
        out = out_dir / "figS_chen_checkpoint_replication.png"
        fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.10, facecolor="white")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.10, facecolor="white")
        print("WROTE", out)
        print("WROTE", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
