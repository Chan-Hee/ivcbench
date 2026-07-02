#!/usr/bin/env python
"""Supplementary figure: a trivial nearest-gene perturbation-representation prior also fails to
beat the simple floor on C3 unseen-gene prediction — the failure is REPRESENTATIONAL, not just a
deep-model failure.

Navy editorial style (src/ivcbench/report/style.py). Two panels:
  (a) BARS — mean response-direction Pearson-Δ across the 5 C3 datasets at the 50% LO-gene holdout,
      95% CI bootstrapped over datasets, for the simple floor (cell-mean / linear-PCA), the two
      nearest-gene priors (co-expression NN, GO-Jaccard NN), the perturbation-agnostic OT floor
      (CINEMA-OT†), and the deep graph models (GEARS, AttentionPert). A dashed line marks the
      cell-mean floor; every learned/transfer method sits left of it.
  (b) ROBUSTNESS — the same methods across 10/25/50% holdout (mean over datasets). The cell-mean
      floor is the bold reference the nearest-gene priors and deep models never clear.

ALL numbers read from deposited data:
  - results/C3/results_raw.csv             (floor, deep models, CINEMA-OT — the published C3 grid)
  - results/C3/nearest_gene_baseline.csv   (the two nearest-gene priors, scored on the SAME splits/
                                            units/metric by scripts/c3_nearest_gene_baseline.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.report.style import (set_pub_style, despine, panel_title, style_legend,  # noqa: E402
                                   NAVY, NAVY_DARK, SLATE_BAND, CLAY_DARK, SIMPLE_GREY, SIMPLE_DARK,
                                   GREY_MID, INK, NULL_GREY, LEGEND_EC)

PAPER = ROOT / "results" / "_paper"
SPLITS = [("C3_true_lo_gene_10", 10), ("C3_true_lo_gene_25", 25), ("C3_true_lo_gene_50", 50)]


def _u(s: str) -> str:
    return s.replace("-", "−")


def boot_ci(vals, n=10000, seed=0):
    """Bootstrap 95% CI of the mean over datasets (the resampling unit)."""
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = vals[rng.integers(0, len(vals), size=(n, len(vals)))].mean(1)
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load_grid():
    dep = pd.read_csv(ROOT / "results/C3/results_raw.csv")
    dep = dep[dep.ran == True][["baseline", "dataset", "split", "pearson_delta"]].copy()
    nng = pd.read_csv(ROOT / "results/C3/nearest_gene_baseline.csv")
    nng = nng[["baseline", "dataset", "split", "pearson_delta"]].copy()
    return pd.concat([dep, nng], ignore_index=True)


def per_split_mean(grid, baseline, split):
    v = grid[(grid.baseline == baseline) & (grid.split == split)]["pearson_delta"]
    return list(v.values)


def main():
    set_pub_style()
    grid = load_grid()

    # method -> (display label, color, is_floor)
    METHODS = [
        ("cell-mean",            "cell-mean (floor)",        NAVY,        "floor"),
        ("linear-PCA",           "linear-PCA (floor)",       "#6E97B4",   "floor"),
        ("CINEMA-OT",            "CINEMA-OT (OT floor †)", SLATE_BAND, "floor"),
        ("nearest-gene-coexpr",  "nearest-gene: co-expr NN", "#9E5A3C",   "prior"),
        ("nearest-gene-go",      "nearest-gene: GO-Jaccard NN", "#C28C6F", "prior"),
        ("GEARS",                "GEARS (graph)",            "#882255",   "deep"),
        ("AttentionPert",        "AttentionPert (graph)",    "#AA4499",   "deep"),
    ]

    fig = plt.figure(figsize=(11.0, 4.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.34,
                          left=0.255, right=0.985, top=0.86, bottom=0.27)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ================= (a) bars at 50% LO-gene, mean over datasets + 95% CI =================
    rows = []
    for key, lab, col, kind in METHODS:
        m, lo, hi = boot_ci(per_split_mean(grid, key, "C3_true_lo_gene_50"))
        rows.append((key, lab, col, kind, m, lo, hi))
    # floor reference value (cell-mean)
    floor_m = next(r[4] for r in rows if r[0] == "cell-mean")
    # order top->bottom: floor block first, then priors, then deep (visual narrative)
    order = ["cell-mean", "linear-PCA", "CINEMA-OT", "nearest-gene-coexpr", "nearest-gene-go",
             "GEARS", "AttentionPert"]
    rows = sorted(rows, key=lambda r: order.index(r[0]))
    yy = np.arange(len(rows))[::-1]                     # first method at top

    for y, (key, lab, col, kind, m, lo, hi) in zip(yy, rows):
        edge = NAVY_DARK if kind == "floor" else INK
        hatch = None
        alpha = 0.95 if kind == "floor" else (0.9 if kind == "prior" else 0.85)
        axA.barh(y, m, height=0.66, color=col, edgecolor=edge, linewidth=0.7, alpha=alpha,
                 zorder=3, hatch=hatch)
        axA.plot([lo, hi], [y, y], color=INK, lw=1.0, zorder=4)
        axA.plot([lo, lo], [y - 0.13, y + 0.13], color=INK, lw=1.0, zorder=4)
        axA.plot([hi, hi], [y - 0.13, y + 0.13], color=INK, lw=1.0, zorder=4)
        xlab = m + 0.012 if m >= 0 else m - 0.012
        ha = "left" if m >= 0 else "right"
        axA.text(max(hi, m) + 0.015, y, _u(f"{m:.2f}"), va="center", ha="left",
                 fontsize=6.8, color=INK)  # _u here IS correct: this is a formatted number, sign-sensitive

    axA.axvline(floor_m, color=NAVY_DARK, lw=1.1, ls="--", zorder=2)
    axA.text(floor_m + 0.012, len(rows) - 0.62, "cell-mean floor", rotation=90, va="top", ha="left",
             fontsize=6.2, color=NAVY_DARK, style="italic")
    axA.axvline(0, color="#bbb", lw=0.6, zorder=1)
    # method names below are hyphenated COMPOUND WORDS (cell-mean, linear-PCA, co-expr, GO-Jaccard,
    # etc.) — plain ASCII hyphens, never unicode-minus'd (that substitution is reserved for signed
    # numbers/axis ticks; see figure_reliability_ceiling.py / FIGURE_DESIGN_STANDARDS.md).
    axA.set_yticks(yy)
    axA.set_yticklabels([r[1] for r in rows], fontsize=7.2)
    axA.set_ylim(-0.6, len(rows) - 0.4)
    axA.set_xlim(min(-0.05, min(r[5] for r in rows) - 0.03), 0.62)
    axA.set_xlabel("response-direction Pearson-Δ  (50% leave-one-gene-out, mean over 5 datasets)  ↑",
                   fontsize=7.6)
    panel_title(axA, "a", "A nearest-gene prior also fails to beat the floor",
                sub="every gene-side transfer sits below the simple cell-mean floor", x_letter=-0.40)
    despine(axA)

    # ================= (b) robustness across 10/25/50% holdout =================
    CURVES = [
        ("cell-mean",           NAVY,        "-",  4.5, 2.6, "cell-mean (floor)"),
        ("linear-PCA",          "#6E97B4",   "-",  3.2, 1.4, "linear-PCA (floor)"),
        ("nearest-gene-coexpr", "#9E5A3C",   "-",  3.6, 1.6, "nearest-gene: co-expr NN"),
        ("nearest-gene-go",     "#C28C6F",   "-",  3.6, 1.6, "nearest-gene: GO-Jaccard NN"),
        ("GEARS",               "#882255",   "--", 3.4, 1.5, "GEARS"),
        ("AttentionPert",       "#AA4499",   "--", 3.4, 1.5, "AttentionPert"),
    ]
    xs = [p for _, p in SPLITS]
    handles = []
    # floor band fill (under cell-mean)
    cm_y = [np.nanmean(per_split_mean(grid, "cell-mean", sp)) for sp, _ in SPLITS]
    for key, col, ls, ms, lw, lab in CURVES:
        ys = [np.nanmean(per_split_mean(grid, key, sp)) for sp, _ in SPLITS]
        h, = axB.plot(xs, ys, ls=ls, lw=lw, color=col, marker="o", ms=ms,
                      zorder=(5 if key == "cell-mean" else 3),
                      label=lab)
        handles.append(h)
    ytop = max(cm_y) * 1.13
    axB.fill_between(xs, cm_y, ytop, color=NAVY, alpha=0.05, zorder=0)
    axB.axhline(0, color="#bbb", lw=0.6, zorder=0)
    axB.annotate("priors & deep models\nstay below the floor", xy=(50, np.mean([cm_y[2], 0.25])),
                 xytext=(33, ytop * 0.55), fontsize=6.6, color=GREY_MID, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-|>", color="#999", lw=0.8,
                                 connectionstyle="arc3,rad=-0.2"))
    axB.set_xticks([10, 25, 50])
    axB.set_xlim(6, 54)
    axB.set_ylim(min(-0.05, axB.get_ylim()[0]), ytop)
    axB.set_xlabel("held-out target genes (%)", fontsize=7.6)
    axB.set_ylabel("mean Pearson-Δ across datasets  ↑", fontsize=7.6)
    panel_title(axB, "b", "Robustness to holdout fraction",
                sub="floor leads at every holdout level", x_letter=-0.16)
    despine(axB)

    leg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.105),
                     ncol=6, fontsize=6.6, frameon=True, handlelength=1.6, columnspacing=1.2)
    style_legend(leg)

    # two manually-wrapped footnote lines, centered under both panels (below the legend), so they
    # never collide with the legend box or run off the figure edges. Plain hyphens throughout —
    # these are all compound words (co-expression-graph, downstream-only, etc.), not signed numbers.
    foot = [
        "Nearest-gene priors predict a held gene's effect as the observed training effect of its single nearest "
        "training gene, by co-expression-graph NN on control cells, or by GO-term Jaccard NN.",
        "Scored on the identical downstream-only Pearson-Δ, splits and units as the deep models. "
        "† CINEMA-OT is a perturbation-agnostic OT floor (not headline-ranked).",
    ]
    fig.text(0.5, 0.052, foot[0], fontsize=5.9, color=GREY_MID, ha="center", va="bottom")
    fig.text(0.5, 0.018, foot[1], fontsize=5.9, color=GREY_MID, ha="center", va="bottom")

    out_png = PAPER / "figS_c3_nearest_gene.png"
    out_pdf = PAPER / "figS_c3_nearest_gene.pdf"
    fig.savefig(out_png, dpi=350)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[fig] wrote {out_png}")
    print(f"[fig] wrote {out_pdf}")


if __name__ == "__main__":
    main()
