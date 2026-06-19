#!/usr/bin/env python
"""Supplementary figure — the NOISE / RELIABILITY CEILING per benchmark cluster.

For each cluster we plot, on the Axis-1 Pearson-Delta scale:
  * the OBSERVED-EFFECT RELIABILITY CEILING (split-half pseudo-replicate r of the observed effect
    vector; Spearman-Brown full-sample value is the ceiling a model that sees all cells could reach),
    with its unit-bootstrap 95% CI;
  * the UNIVERSAL FLOOR (best simple baseline) and the BEST MODEL's Pearson-Delta (deposited headline);
The gap between the best model and the ceiling is the headroom that is NOT measurement noise: when the
best model sits far below a HIGH ceiling, the floor-failure is real, not just an unreliable target.

Everything is read from deposited artefacts:
  results/_paper/immune_novelty/reliability_ceiling.csv      (computed by reliability_ceiling.py)
  results/_paper/cross_cluster_headline.csv                  (best model + floor per cluster)
Nothing is hardcoded. Navy editorial style (report/style.py).
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.report.style import (set_pub_style, despine, panel_title, style_legend,  # noqa: E402
                                   NAVY, NAVY_DARK, SLATE_BAND, SIMPLE_GREY, NULL_GREY,
                                   CLAY_DARK, INK, GREY_MID, GREY_LITE)

PAPER = ROOT / "results" / "_paper"
REL = pd.read_csv(PAPER / "immune_novelty" / "reliability_ceiling.csv")
HEAD = pd.read_csv(PAPER / "cross_cluster_headline.csv")

# headline task per cluster aligned to the reliability unit definition
TASK_SPLIT = {
    "C1": "cell-context (LOCT)",
    "C2": "donor (LODO)",
    "C3": "unseen-perturbation (LO-gene 10%)",
    "C4": "unseen-KO (modality, RNA)",
    "C5": "unseen-compound",
}
CL_LABEL = {
    "C1": "T1  cytokine\nKang IFN-β / LOCT",
    "C2": "T2  donor\nSoskic CD4 / LODO",
    "C3": "T3  gene\nCRISPR / unseen-gene",
    "C4": "T4  modality\nFrangieh / unseen-KO",
    "C5": "T5  small-mol\nOP3 / unseen-cpd",
}


def best_model_and_floor(cl: str):
    split = TASK_SPLIT[cl]
    g = HEAD[(HEAD["cluster"] == cl) & (HEAD["split"] == split)]
    if g.empty:
        return None
    best = g.loc[g["pearson_delta"].idxmax()]
    return dict(best_model=str(best["model"]), best_pd=float(best["pearson_delta"]),
                floor=float(best["floor_mean"]))


def main():
    set_pub_style()
    clusters = ["C1", "C2", "C3", "C4", "C5"]
    rel = REL.set_index("cluster")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(clusters))[::-1]  # C1 at top

    C_CEIL = NAVY            # reliability ceiling (observed-effect, full-sample SB)
    C_CEIL_HALF = SLATE_BAND  # half-sample split-half point
    C_FLOOR = SIMPLE_GREY    # universal floor
    C_BEST = CLAY_DARK       # best model

    for yi, cl in zip(y, clusters):
        r = rel.loc[cl]
        ceil_full = float(r["reliability_fullsample_SB"])
        ceil_half = float(r["reliability_halfsample"])
        lo, hi = float(r["ci_lo_halfsample"]), float(r["ci_hi_halfsample"])
        # Spearman-Brown the CI bounds too (monotone transform) for the full-sample band
        sb = lambda x: 2 * x / (1 + x)
        lo_f, hi_f = sb(lo), sb(hi)
        bm = best_model_and_floor(cl)

        # ceiling band: from floor area to ceiling, light navy wash to show "reachable & real" zone
        # ceiling CI (full-sample) as a horizontal bar
        ax.plot([lo_f, hi_f], [yi, yi], color=C_CEIL, lw=6, alpha=0.22, solid_capstyle="round",
                zorder=1)
        # ceiling point (full-sample, SB)
        ax.scatter([ceil_full], [yi], s=120, marker="D", color=C_CEIL, edgecolor="white",
                   linewidth=0.8, zorder=5)
        # half-sample point (smaller, slate) just below
        ax.scatter([ceil_half], [yi], s=34, marker="d", color=C_CEIL_HALF, edgecolor="white",
                   linewidth=0.5, zorder=4)
        # floor
        ax.scatter([bm["floor"]], [yi], s=70, marker="s", color=C_FLOOR, edgecolor="white",
                   linewidth=0.7, zorder=4)
        # best model
        ax.scatter([bm["best_pd"]], [yi], s=90, marker="o", color=C_BEST, edgecolor="white",
                   linewidth=0.8, zorder=5)

        # connect best-model -> ceiling with a thin "headroom" arrow. When the deposited best model
        # sits ABOVE the per-unit reliability ceiling (C3/C4: deposited score is macro-averaged on a
        # COARSER grain — dataset / modality-fold — than the per-unit ceiling), the arrow reverses and
        # the gap is negative; we annotate that case explicitly rather than hide it.
        gap = ceil_full - bm["best_pd"]
        x0, x1 = bm["best_pd"], ceil_full
        ax.annotate("", xy=(x1, yi + 0.0), xytext=(x0, yi + 0.0),
                    arrowprops=dict(arrowstyle="-|>", color=GREY_MID, lw=0.9,
                                    shrinkA=6, shrinkB=8, alpha=0.85), zorder=2)
        midx = (x0 + x1) / 2
        glab = f"headroom {gap:+.2f}" if gap >= 0 else f"model > ceiling ({gap:+.2f})†"
        ax.text(midx, yi + 0.20, glab, ha="center", va="bottom",
                fontsize=6.2, color=(GREY_MID if gap >= 0 else CLAY_DARK), style="italic")
        # best-model name label below its marker
        ax.text(bm["best_pd"], yi - 0.27, bm["best_model"], ha="center", va="top",
                fontsize=6.0, color=C_BEST)
        # floor value label (small, above the floor square)
        ax.text(bm["floor"], yi + 0.18, f"floor {bm['floor']:.2f}", ha="center", va="bottom",
                fontsize=5.4, color=GREY_MID)
        # ceiling value label
        ax.text(ceil_full, yi - 0.27, f"ceiling {ceil_full:.2f}", ha="center", va="top",
                fontsize=6.0, color=NAVY_DARK, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([CL_LABEL[c] for c in clusters], fontsize=7.0)
    ax.set_xlabel("Axis-1 response-direction recovery  (Pearson-Δ across genes)", fontsize=8)
    ax.set_xlim(-0.05, 1.06)
    ax.set_ylim(-1.05, len(clusters) - 0.15)
    ax.axvline(0, color=GREY_LITE, lw=0.6, ls=":", zorder=0)
    despine(ax)
    # footnote: the grain caveat for T3/T4 (deposited best-model on a coarser averaging unit)
    fig_note = ("† T3 / T4: deposited best-model & floor are macro-averaged on a coarser grain "
                "(T3 dataset n=5; T4 modality-fold n=2) than the per-unit reliability (per KO-gene), "
                "so the per-unit ceiling can sit below the coarse-grain score.")
    ax.text(0.0, -0.34, fig_note, transform=ax.transAxes, fontsize=5.4, color=GREY_MID,
            va="top", ha="left", style="italic", wrap=True)
    panel_title(ax, "", "Observed-effect reliability ceiling vs best model, per task",
                sub="split-half pseudo-replicate reliability of the observed effect "
                    "(Spearman-Brown full-sample); floor = best simple baseline",
                x_letter=-0.01)

    legend_handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor=NAVY, markeredgecolor="white",
               markersize=9, label="Reliability ceiling (full-sample, SB)"),
        Line2D([0], [0], marker="d", color="none", markerfacecolor=SLATE_BAND,
               markeredgecolor="white", markersize=6, label="Split-half (half-sample) r"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CLAY_DARK, markeredgecolor="white",
               markersize=8, label="Best model (deposited headline)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=SIMPLE_GREY,
               markeredgecolor="white", markersize=8, label="Universal floor (best simple baseline)"),
    ]
    leg = ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                    ncol=2, fontsize=6.4, frameon=True, handletextpad=0.4, borderpad=0.6,
                    columnspacing=1.4)
    style_legend(leg)

    fig.tight_layout()
    out_png = PAPER / "figure_reliability_ceiling.png"
    out_pdf = PAPER / "figure_reliability_ceiling.pdf"
    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print("WROTE", out_png)
    print("WROTE", out_pdf)


if __name__ == "__main__":
    main()
