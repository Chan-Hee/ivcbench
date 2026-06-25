#!/usr/bin/env python
"""Figure: program-recovery vs program-dimensionality (tests "only low-rank programs survive").

Panel A — the ASSERTED law: recovery (AUCell-Δ corr, deposited) vs program low-rank-ness (FVE-PC1 of
          the program gene block, recomputed from the actual cells). type-I IFN highlighted.
Panel B — the REAL mechanism: recovery vs the observed across-stratum program-shift SD (the learnable
          signal magnitude, recomputed from the actual cells).
Navy editorial style (src/ivcbench/report/style.py). Reads only the two recomputed CSVs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.report.style import (set_pub_style, NAVY, NAVY_DARK, CLAY_DARK, INK, GREY_MID,
                                   GREY_LITE, panel_title, despine)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DF = pd.read_csv(ROOT / "results/_paper/program_recovery_vs_dimensionality.csv")
OUT = ROOT / "results/_paper/program_recovery_vs_dimensionality.png"


def scatter_panel(ax, letter, title, sub, xcol, xlabel, df, law_text):
    fit = df[df.recovery.notna() & df[xcol].notna()].copy()
    is_ifn = fit["program"].eq("type_I_IFN") & fit["cluster"].isin(["C5"])
    other = fit[~is_ifn]; ifn = fit[is_ifn]
    # other programs (navy hollow, cluster-shaped)
    markers = {"C5": "o", "C3": "s", "C1": "^"}
    for cl, mk in markers.items():
        s = other[other.cluster == cl]
        if len(s):
            ax.scatter(s[xcol], s.recovery, s=58, marker=mk, facecolor="white",
                       edgecolor=NAVY, linewidth=1.3, zorder=3,
                       label={"C5": "T5 program", "C3": "T3 program"}.get(cl))
    # type-I IFN highlighted (filled navy, larger)
    ax.scatter(ifn[xcol], ifn.recovery, s=150, marker="*", facecolor=CLAY_DARK,
               edgecolor=NAVY_DARK, linewidth=1.2, zorder=5, label="type-I IFN (T5)")
    # label each point with its program name
    span = fit[xcol].max() - fit[xcol].min()
    for _, r in fit.iterrows():
        nm = {"type_I_IFN": "type-I IFN", "inflammatory_NFkB": "infl-NFκB",
              "effector_lymphocyte": "effector-lymph", "effector_cytokine": "effector-cyt",
              "TCR_activation": "TCR-act", "IL2_STAT5": "IL2-STAT5",
              "proliferation": "prolif", "Treg_exhaustion": "Treg-exh"}.get(r.program, r.program)
        if r.program == "type_I_IFN":
            ax.annotate(nm, (r[xcol], r.recovery), xytext=(r[xcol] - 0.018 * span, r.recovery - 0.055),
                        fontsize=6.4, color=CLAY_DARK, fontweight="bold", va="top", ha="right", zorder=6)
        else:
            ax.annotate(nm, (r[xcol], r.recovery), xytext=(r[xcol] + 0.018 * span, r.recovery + 0.030),
                        fontsize=6.0, color=GREY_MID, va="bottom", ha="left", zorder=4)
    # regression line + correlation
    pr, pp = stats.pearsonr(fit[xcol], fit.recovery)
    sr, sp = stats.spearmanr(fit[xcol], fit.recovery)
    xs = np.linspace(fit[xcol].min(), fit[xcol].max(), 50)
    b, a = np.polyfit(fit[xcol], fit.recovery, 1)
    ax.plot(xs, b * xs + a, color=NAVY, lw=1.4, ls="--", alpha=0.65, zorder=2)
    # stats box
    ax.text(0.035, 0.97, f"Pearson r = {pr:+.2f} (p = {pp:.3f})\nSpearman ρ = {sr:+.2f} (p = {sp:.3f})\nn = {len(fit)} scored programs",
            transform=ax.transAxes, fontsize=6.6, va="top", ha="left", color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY_LITE, lw=0.6))
    ax.text(0.5, -0.205, law_text, transform=ax.transAxes, fontsize=6.4, va="top", ha="center",
            color=NAVY_DARK, style="italic")
    ax.set_xlabel(xlabel); ax.set_ylabel("program recovery\n(AUCell-Δ corr, deposited)")
    ax.axhline(0, color=GREY_LITE, lw=0.6, ls=":", zorder=1)
    ax.set_ylim(-0.06, 0.92)
    despine(ax)
    panel_title(ax, letter, title, sub)


def main():
    set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))
    scatter_panel(
        axes[0], "a", "Recovery vs low-rank-ness (the asserted law)",
        "x = fraction of variance on PC1 of the program gene block",
        "fve_pc1", "program low-rank-ness  (FVE on PC1) →",
        DF,
        "Law predicts a POSITIVE slope (more low-rank → recovered).  Observed: flat / non-significant;\n"
        "the most low-rank program (effector-lymph, FVE-PC1=0.66) is essentially NOT recovered.")
    scatter_panel(
        axes[1], "b", "Recovery vs program-shift signal (signal magnitude)",
        "x = SD of observed program-Δ across perturbation strata",
        "obs_shift_sd", "across-stratum program-shift SD  (signal) →",
        DF,
        "Recovery tracks the learnable SIGNAL, not dimensionality:  type-I IFN survives because it has the\n"
        "largest cross-perturbation program shift, not because it is the most low-dimensional.")
    # shared legend
    handles = [
        Line2D([0],[0], marker="*", color="none", markerfacecolor=CLAY_DARK, markeredgecolor=NAVY_DARK,
               markersize=13, label="type-I IFN (T5)"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor="white", markeredgecolor=NAVY,
               markersize=8, label="T5 program (OP3)"),
        Line2D([0],[0], marker="s", color="none", markerfacecolor="white", markeredgecolor=NAVY,
               markersize=8, label="T3 program (5×CRISPR)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.045), ncol=3,
               frameon=False, fontsize=7, handletextpad=0.4, columnspacing=1.6)
    fig.suptitle("Program recovery is set by signal magnitude, not by low-dimensionality",
                 y=1.10, fontsize=9.5, fontweight="bold", color=NAVY_DARK)
    fig.subplots_adjust(wspace=0.32, top=0.82, bottom=0.20, left=0.085, right=0.985)
    fig.savefig(OUT, dpi=350, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print("WROTE", OUT)
    print("WROTE", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
