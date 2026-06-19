#!/usr/bin/env python
"""Supplementary figure: unseen-cytokine extrapolation (pseudobulk-DE) — a direct test of the
no-unseen-perturbation-extrapolation law on CYTOKINES.

Reads ONLY results/newdata/cytokine_loco_*.csv (computed by scripts/newdata_cytokine_loco.py from
the Cytokine Dictionary summary table). Navy editorial style (src/ivcbench/report/style.py).

Three panels tell the nuanced finding:
  (a) POOLED bars — mean response-direction Pearson over every held cytokine (pooled across celltypes)
      for the zero baseline, the cytokine-mean floor, the annotation feature-nearest predictor, and
      the DE-profile-nearest predictor. A dashed line marks the cytokine-mean floor. 95% CI over held
      cytokines.
  (b) PER-CELLTYPE gap — DE-profile-nearest minus floor, per celltype (paired); positive in 19/24
      celltypes. Shows the transfer win is broad, concentrated in the well-sampled immune lineages.
  (c) THE TWO CONDITIONING REGIMES — a scatter of feature-nearest (x, annotation-only: the cytokine
      is treated as truly novel) vs DE-profile-nearest (y, the cytokine observed in OTHER celltypes),
      each point a celltype, both relative to the floor. Annotation-only conditioning sits at/below
      the floor (law holds); observed-elsewhere transfer sits above it (law is escaped only when the
      perturbation has been seen in another context).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.report.style import (set_pub_style, despine, panel_title, style_legend,  # noqa: E402
                                   NAVY, NAVY_DARK, SLATE_BAND, CLAY_DARK, CONDITIONED,
                                   CONDITIONED_DARK, SIMPLE_GREY, SIMPLE_DARK, GREY_MID, INK,
                                   NULL_GREY, LEGEND_EC)

NEWDATA = ROOT / "results" / "newdata"
PAPER = ROOT / "results" / "_paper"


def _u(s: str) -> str:
    return s.replace("-", "−")


def boot_ci(vals, n=10000, seed=0):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    set_pub_style()
    res = pd.read_csv(NEWDATA / "cytokine_loco_per_held.csv")
    per_ct = pd.read_csv(NEWDATA / "cytokine_loco_per_celltype.csv")
    with open(NEWDATA / "cytokine_loco_summary.json") as f:
        summ = json.load(f)

    wide = res.pivot_table(index=["celltype", "cytokine"], columns="method", values="pearson")
    floor_col = "cytokine-mean"

    # ---- 2-row layout (was a single very-wide 1x3 row at 4.52:1, which scaled the axis and
    # cell-type labels down to illegibility in Word/PDF). Panel (b) carries 24 cell-type rows so it
    # anchors the FULL-HEIGHT left column; panels (a) and (c) stack in the right column. The long
    # methods footnote now lives in the figure CAPTION (Supplementary Fig. S11 caption), not in the
    # figure, so the plate breathes. Target aspect ~2:1. No data value / colour / label changes.
    fig = plt.figure(figsize=(10.2, 5.2))                       # ~1.96:1
    gs = fig.add_gridspec(2, 2, width_ratios=[1.02, 1.0], height_ratios=[1.0, 1.0],
                          wspace=0.46, hspace=0.62,
                          left=0.07, right=0.985, top=0.90, bottom=0.085)
    axB = fig.add_subplot(gs[:, 0])     # per-celltype gap — tall, spans both rows (left column)
    axA = fig.add_subplot(gs[0, 1])     # pooled bars — top-right
    axC = fig.add_subplot(gs[1, 1])     # two conditioning regimes — bottom-right

    # ================= (a) pooled bars =================
    METHODS = [
        ("zero",               "zero (no response)",          NULL_GREY,    "floor"),
        ("cytokine-mean",      "cytokine-mean (floor)",       NAVY,         "floor"),
        ("feature-nearest",    "feature-nearest (annotation)", CLAY_DARK,   "cond"),
        ("DE-profile-nearest", "DE-profile-nearest (transfer)", CONDITIONED, "cond"),
    ]
    rows = []
    for key, lab, col, kind in METHODS:
        m, lo, hi = boot_ci(res[res.method == key]["pearson"].values)
        rows.append((key, lab, col, kind, m, lo, hi))
    floor_m = next(r[4] for r in rows if r[0] == floor_col)
    yy = np.arange(len(rows))[::-1]
    for y, (key, lab, col, kind, m, lo, hi) in zip(yy, rows):
        edge = NAVY_DARK if key == floor_col else (CONDITIONED_DARK if kind == "cond" else SIMPLE_DARK)
        axA.barh(y, m, height=0.64, color=col, edgecolor=edge, linewidth=0.8,
                 alpha=0.95 if kind == "floor" else 0.9, zorder=3)
        if np.isfinite(lo):
            axA.plot([lo, hi], [y, y], color=INK, lw=1.0, zorder=4)
            axA.plot([lo, lo], [y - 0.12, y + 0.12], color=INK, lw=1.0, zorder=4)
            axA.plot([hi, hi], [y - 0.12, y + 0.12], color=INK, lw=1.0, zorder=4)
        axA.text(max(hi, m) + 0.012, y, _u(f"{m:.2f}"), va="center", ha="left",
                 fontsize=7.0, color=INK)
    axA.axvline(floor_m, color=NAVY_DARK, lw=1.1, ls="--", zorder=2)
    axA.text(floor_m + 0.008, len(rows) - 0.58, "floor", rotation=90, va="top", ha="left",
             fontsize=6.4, color=NAVY_DARK, style="italic")
    axA.axvline(0, color="#bbb", lw=0.6, zorder=1)
    axA.set_yticks(yy)
    axA.set_yticklabels([_u(r[1]) for r in rows], fontsize=7.0)
    axA.set_ylim(-0.55, len(rows) - 0.45)
    axA.set_xlim(-0.02, 0.42)
    axA.set_xlabel(_u("response-direction Pearson  (held-out cytokine, mean over all held)  ↑"),
                   fontsize=7.4)
    panel_title(axA, "a", "Unseen-cytokine extrapolation",
                sub=f"leave-one-cytokine-out, {summ['n_held_cytokine_instances']:,} held instances "
                    f"× {summ['n_celltypes_tested']} celltypes", x_letter=-0.30)
    despine(axA)

    # ================= (b) per-celltype paired gap =================
    pc = per_ct.copy()
    pc["gap"] = pc["DE-profile-nearest"] - pc[floor_col]
    pc = pc.sort_values("gap")
    yb = np.arange(len(pc))
    cols = [CONDITIONED if g > 0 else CLAY_DARK for g in pc["gap"]]
    axB.barh(yb, pc["gap"], height=0.72, color=cols, edgecolor=INK, linewidth=0.5,
             alpha=0.9, zorder=3)
    axB.axvline(0, color=NAVY_DARK, lw=1.0, zorder=2)
    axB.set_yticks(yb)
    axB.set_yticklabels([_u(c.replace("_", " ")) for c in pc["celltype"]], fontsize=6.0)
    axB.set_ylim(-0.7, len(pc) - 0.3)
    n_pos = int((pc["gap"] > 0).sum())
    axB.set_xlabel(_u("DE-profile-nearest − cytokine-mean floor  (Δ Pearson)  →"),
                   fontsize=7.4)
    axB.text(0.98, 0.04, f"transfer beats floor in\n{n_pos}/{len(pc)} celltypes",
             transform=axB.transAxes, ha="right", va="bottom", fontsize=6.6,
             color=CONDITIONED_DARK, style="italic")
    axB.text(0.02, 0.96, "floor wins\n(small-n lineages)", transform=axB.transAxes,
             ha="left", va="top", fontsize=6.2, color=CLAY_DARK, style="italic")
    panel_title(axB, "b", "The transfer win is broad across immune lineages",
                sub="observed-elsewhere cytokine transfer, per celltype", x_letter=-0.20)
    despine(axB)

    # ================= (c) the two conditioning regimes =================
    # x = annotation-only (feature-nearest) gap vs floor; y = observed-elsewhere (DE-profile) gap.
    g_ft = (per_ct["feature-nearest"] - per_ct[floor_col]).values
    g_de = (per_ct["DE-profile-nearest"] - per_ct[floor_col]).values
    sz = 14 + 0.55 * per_ct["n_held"].values
    axC.axhline(0, color=NAVY_DARK, lw=1.0, ls="--", zorder=1)
    axC.axvline(0, color=NAVY_DARK, lw=1.0, ls="--", zorder=1)
    # shaded quadrants
    axC.axhspan(0, 0.35, xmin=0.0, xmax=1.0, color=CONDITIONED, alpha=0.05, zorder=0)
    axC.scatter(g_ft, g_de, s=sz, c=CONDITIONED, edgecolor=CONDITIONED_DARK, linewidth=0.6,
                alpha=0.85, zorder=4)
    axC.scatter([np.mean(g_ft)], [np.mean(g_de)], marker="D", s=46, c=NAVY,
                edgecolor="white", linewidth=0.9, zorder=6)
    axC.annotate("pooled mean", xy=(np.mean(g_ft), np.mean(g_de)),
                 xytext=(np.mean(g_ft) - 0.085, np.mean(g_de) + 0.045), fontsize=6.2,
                 color=NAVY_DARK, ha="center",
                 arrowprops=dict(arrowstyle="-", color=NAVY_DARK, lw=0.6))
    axC.text(0.97, 0.97, "transfer escapes\nthe floor", transform=axC.transAxes, ha="right",
             va="top", fontsize=6.6, color=CONDITIONED_DARK, style="italic")
    axC.text(0.03, 0.04, "annotation-only\nconditioning fails", transform=axC.transAxes,
             ha="left", va="bottom", fontsize=6.6, color=CLAY_DARK, style="italic")
    axC.set_xlabel(_u("feature-nearest − floor  (annotation only)"), fontsize=7.4)
    axC.set_ylabel(_u("DE-profile-nearest − floor  (observed elsewhere)"), fontsize=7.4)
    panel_title(axC, "c", "Two conditioning regimes",
                sub="each point a celltype; marker size ~ held cytokines", x_letter=-0.22)
    despine(axC)

    # NOTE: the long methods footnote that used to sit under the plate is intentionally NOT drawn
    # here — at the new 2-row aspect it would crowd the panels and re-shrink the labels. Its content
    # (leave-one-cytokine-out protocol; feature-nearest = annotation-only truly-novel regime vs
    # DE-profile-nearest = observed-elsewhere transfer; pseudobulk-DE summary-table probe) is carried
    # by the Supplementary Figure S11 caption instead.

    out_png = PAPER / "figS_newdata_cytokine_loco.png"
    out_pdf = PAPER / "figS_newdata_cytokine_loco.pdf"
    fig.savefig(out_png, dpi=350)
    fig.savefig(out_pdf)
    plt.close(fig)
    # also drop a copy under results/newdata for the data dir
    fig_png2 = NEWDATA / "figS_newdata_cytokine_loco.png"
    import shutil
    shutil.copy(out_png, fig_png2)
    print(f"[fig] wrote {out_png}")
    print(f"[fig] wrote {out_pdf}")
    print(f"[fig] copied {fig_png2}")


if __name__ == "__main__":
    main()
