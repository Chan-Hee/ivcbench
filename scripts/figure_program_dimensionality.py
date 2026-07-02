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


def _u(s: str) -> str:
    """True unicode minus on a preformatted numeric string (never touches hyphenated words)."""
    return s.replace("-", "−")


PROG_NAMES = {"type_I_IFN": "type-I IFN", "inflammatory_NFkB": "infl-NFκB",
              "effector_lymphocyte": "effector-lymph", "effector_cytokine": "effector-cyt",
              "TCR_activation": "TCR-act", "IL2_STAT5": "IL2-STAT5",
              "proliferation": "prolif", "Treg_exhaustion": "Treg-exh"}

# Per-panel explicit label placement (axes-fraction coordinates for the label anchor) so the
# lower-left cluster is fanned into free whitespace and every label reads separately. Each entry:
# program -> (label_x_axesfrac, label_y_axesfrac, ha, va). A thin leader line connects the point
# (converted to axes-fraction) to the label when they are not adjacent. Points/statistics unchanged.
LABEL_PLACEMENT = {
    "fve_pc1": {  # panel (a): x-range ~0.20..0.66 (data-frac), y-range recovery
        "type_I_IFN":          (0.410, 0.720, "left",  "center"),
        "inflammatory_NFkB":   (0.175, 0.315, "left",  "center"),
        "effector_cytokine":   (0.470, 0.235, "left",  "center"),
        "proliferation":       (0.690, 0.088, "left",  "center"),
        "effector_lymphocyte": (0.905, 0.180, "right", "center"),
        # TCR-act's old anchor (y=0.215) sat almost exactly on the near-flat regression line
        # (line yfrac ~0.20-0.21 across this whole x-range) — raised clear of it.
        "TCR_activation":      (0.045, 0.270, "left",  "center"),
        "IL2_STAT5":           (0.045, 0.125, "left",  "center"),
        "Treg_exhaustion":     (0.420, 0.055, "left",  "center"),
    },
    "obs_shift_sd": {  # panel (b): x-range ~0.003..0.042 (data-frac), y-range recovery
        "type_I_IFN":          (0.905, 0.905, "right", "center"),
        "inflammatory_NFkB":   (0.360, 0.320, "left",  "center"),
        "effector_cytokine":   (0.360, 0.235, "left",  "center"),
        "effector_lymphocyte": (0.680, 0.160, "left",  "center"),
        # prolif / IL2-STAT5 / TCR-act sit almost on top of one another in x (all <0.007 SD) and
        # the regression line rises steeply through this corner. Right-aligning the labels (text
        # grows LEFTWARD from the anchor) keeps every glyph on the left, upper side of the line
        # instead of growing rightward into it; anchor x is staggered to roughly track each
        # point's own x-order (prolif < IL2-STAT5 < TCR-act) so the three leaders fan out without
        # a lower leader re-crossing a label sitting above it.
        "proliferation":       (0.270, 0.340, "right", "center"),
        "IL2_STAT5":           (0.190, 0.260, "right", "center"),
        "TCR_activation":      (0.230, 0.185, "right", "center"),
        "Treg_exhaustion":     (0.235, 0.055, "left",  "center"),
    },
}


def scatter_panel(ax, letter, title, sub, xcol, xlabel, df, law_text, stats_ha="left"):
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
    # label each point with its program name (explicit non-overlapping placement + leader lines)
    from matplotlib.transforms import blended_transform_factory  # noqa: F401
    placement = LABEL_PLACEMENT[xcol]
    # x/y limits are set below (set_ylim) and via autoscale on x; force them now so the
    # data->axes-fraction conversion for leaders is correct, without changing the ranges.
    ax.autoscale(enable=True, axis="x")
    ax.set_ylim(-0.06, 0.92)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    for _, r in fit.iterrows():
        nm = PROG_NAMES.get(r.program, r.program)
        lx, ly, ha, va = placement[r.program]
        is_ifn_pt = (r.program == "type_I_IFN")
        color = CLAY_DARK if is_ifn_pt else GREY_MID
        fw = "bold" if is_ifn_pt else "normal"
        fs = 6.6 if is_ifn_pt else 6.2
        # point position in axes-fraction, for the leader line
        px = (r[xcol] - x0) / (x1 - x0); py = (r.recovery - y0) / (y1 - y0)
        # draw a thin leader from a point on the label side toward the marker
        pad = 0.012 * (len(nm) if ha == "left" else -len(nm))  # small gap from text to leader
        # leader end near the text (a touch off the label anchor, on the marker side)
        leadx = lx + (0.010 if ha == "left" else -0.010)
        dist = ((leadx - px) ** 2 + (ly - py) ** 2) ** 0.5
        if dist > 0.045:  # only draw a leader when the label is genuinely offset from its point
            ax.plot([leadx, px], [ly, py], transform=ax.transAxes, color=GREY_LITE,
                    lw=0.5, alpha=0.75, zorder=2, solid_capstyle="round")
        ax.text(lx, ly, nm, transform=ax.transAxes, fontsize=fs, color=color, fontweight=fw,
                ha=ha, va=va, zorder=6)
    # regression line + correlation
    pr, pp = stats.pearsonr(fit[xcol], fit.recovery)
    sr, sp = stats.spearmanr(fit[xcol], fit.recovery)
    xs = np.linspace(fit[xcol].min(), fit[xcol].max(), 50)
    b, a = np.polyfit(fit[xcol], fit.recovery, 1)
    ax.plot(xs, b * xs + a, color=NAVY, lw=1.4, ls="--", alpha=0.65, zorder=2)
    # stats box (right-anchored in panel a to clear the type-I IFN star; left-anchored in panel b)
    box_x = 0.965 if stats_ha == "right" else 0.035
    ax.text(box_x, 0.97, _u(f"Pearson r = {pr:+.2f} (p = {pp:.3f})\nSpearman ρ = {sr:+.2f} (p = {sp:.3f})\nn = {len(fit)} scored programs"),
            transform=ax.transAxes, fontsize=6.6, va="top", ha=stats_ha, color=INK,
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
        "the most low-rank program (effector-lymph, FVE-PC1=0.66) is essentially NOT recovered.",
        stats_ha="right")
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
