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
import matplotlib.colors as mcolors
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
    # anchors the FULL-HEIGHT column; panels (a) and (c) stack in the other column. The plate is
    # laid out so the on-page reading order runs a (top-left) -> b (right, full-height) -> c
    # (bottom-left): the pooled-bars result (a) starts top-left, the tall per-celltype panel (b)
    # spans the RIGHT column, and the two-regime scatter (c) closes bottom-left. The long methods
    # footnote lives in the figure CAPTION (Supplementary Fig. S8 caption), not in the figure, so
    # the plate breathes. Target aspect ~2:1. No data value / colour / label changes.
    # 6.9 in wide, not 10.2: the supplement prints every figure at 6.30 in, and at 10.2 the
    # 6.0 pt cell-type labels of panel b reached the page at 3.5 pt. Same aspect, same point
    # sizes, authored at the size it is printed at.
    # Only the WIDTH is constrained (the supplement prints at 6.30 in); height is free, and
    # panel b's 24 rows want it. panel_title places its title and subtitle in AXES fractions, so
    # shortening the panels would push both onto the axes -- the height stays.
    fig = plt.figure(figsize=(6.6, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.02], height_ratios=[1.0, 1.0],
                          # 0.72 left a 0.53 in strip of the left column completely empty
                          # between (a) and (c) while (b) ran full height beside it.
                          wspace=0.70, hspace=0.58,
                          left=0.245, right=0.985, top=0.925, bottom=0.075)
    axA = fig.add_subplot(gs[0, 0])     # pooled bars — top-left
    axC = fig.add_subplot(gs[1, 0])     # two conditioning regimes — bottom-left
    axB = fig.add_subplot(gs[:, 1])     # per-celltype gap — tall, spans both rows (right column)

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
        # The edge was chosen by KIND, so feature-nearest -- orange fill, kind "cond" -- wore
        # the conditioned plum, and the bar read as outlined in another series' colour. Every
        # bar now takes a darkened version of its own fill, as the navy and magenta already did.
        edge = tuple(0.62 * c for c in mcolors.to_rgb(col))
        axA.barh(y, m, height=0.64, color=col, edgecolor=edge, linewidth=0.8,
                 alpha=0.95 if kind == "floor" else 0.9, zorder=3)
        if np.isfinite(lo):
            axA.plot([lo, hi], [y, y], color=INK, lw=1.0, zorder=4)
            axA.plot([lo, lo], [y - 0.12, y + 0.12], color=INK, lw=1.0, zorder=4)
            axA.plot([hi, hi], [y - 0.12, y + 0.12], color=INK, lw=1.0, zorder=4)
        # The bar labels sit just past each bar, which put the 0.15 one ON the dashed floor
        # line at 0.195. Only a label that would land in that band is pushed past the line;
        # nudging all of them drove the zero bar's label into the rotated "floor" annotation.
        lab_x = max(hi, m) + 0.012
        if floor_m - 0.035 < lab_x < floor_m + 0.012:
            lab_x = floor_m + 0.016
        # Two decimals printed the floor as 0.19 against the caption's 0.195; three decimals is
        # the project's rounding, and the widest label still ends 0.02 inside the right limit.
        axA.text(lab_x, y, _u(f"{m:.3f}"),
                 va="center", ha="left",
                 fontsize=7.0, color=INK)  # _u here IS correct: a formatted, sign-sensitive number
    axA.axvline(floor_m, color=NAVY_DARK, lw=1.1, ls="--", zorder=2)
    # +0.008 left 1.6 pt between the glyphs and the dashed rule, which reads as one object.
    axA.text(floor_m + 0.020, len(rows) - 0.58, "floor", rotation=90, va="top", ha="left",
             fontsize=6.4, color=NAVY_DARK, style="italic")
    axA.axvline(0, color="#bbb", lw=0.6, zorder=1)
    # method names + axis text below are compound words (cytokine-mean, feature-nearest,
    # DE-profile-nearest, held-out, etc.) — plain ASCII hyphens; unicode-minus is reserved for
    # signed numbers/axis ticks (see FIGURE_DESIGN_STANDARDS.md).
    axA.set_yticks(yy)
    axA.set_yticklabels([r[1] for r in rows], fontsize=7.0)
    axA.set_ylim(-0.55, len(rows) - 0.45)
    axA.set_xlim(-0.02, 0.42)
    axA.set_xlabel("response-direction Pearson  (held-out cytokine, mean over all held)  ↑",
                   fontsize=7.4)
    panel_title(axA, "a", "Unseen-cytokine extrapolation",
                sub=f"leave-one-cytokine-out, {summ['n_held_cytokine_instances']:,} held instances "
                    f"across {summ['n_celltypes_tested']} cell types", x_letter=-0.30)
    despine(axA)

    # ================= (b) per-celltype paired gap =================
    pc = per_ct.copy()
    pc["gap"] = pc["DE-profile-nearest"] - pc[floor_col]
    pc = pc.sort_values("gap")
    yb = np.arange(len(pc))
    # A negative gap means the FLOOR wins, so it takes the floor's own colour. CLAY_DARK is
    # the annotation-only predictor in panel a and the "annotation-only fails" note in panel
    # c, so using it here made orange mean two different things inside one figure.
    cols = [CONDITIONED if g > 0 else NAVY for g in pc["gap"]]
    axB.barh(yb, pc["gap"], height=0.72, color=cols, edgecolor=INK, linewidth=0.5,
             alpha=0.9, zorder=3)
    axB.axvline(0, color=NAVY_DARK, lw=1.0, zorder=2)
    axB.set_yticks(yb)
    axB.set_yticklabels([c.replace("_", " ") for c in pc["celltype"]], fontsize=6.0)
    axB.set_ylim(-0.7, len(pc) - 0.3)
    n_pos = int((pc["gap"] > 0).sum())
    # "−" here is a real subtraction (DE-profile-nearest minus the floor); the two method names
    # either side of it keep plain hyphens as compound words.
    axB.set_xlabel("DE-profile-nearest − cytokine-mean floor  (Δ Pearson)  ↑",
                   fontsize=7.4)
    # Each note goes beside the rows it describes, in the empty half of those rows. The winning
    # rows have bars running RIGHT from zero, so their free space is on the left; the losing rows
    # run left, so theirs is on the right. Putting both notes on one side, as before, meant one of
    # them always sat either on the bars or beside the wrong rows.
    axB.text(0.02, 0.97, f"transfer beats floor in\n{n_pos}/{len(pc)} cell types",
             transform=axB.transAxes, ha="left", va="top", fontsize=6.6,
             color=CONDITIONED_DARK, style="italic")
    # It used to sit at y=0.96, beside CD4 T cell and CD4 Memory T cell -- the two largest
    # transfer wins -- while the rows it describes are the five orange bars at the bottom.
    axB.text(0.98, 0.02, "floor wins\n(small-n lineages)", transform=axB.transAxes,
             ha="right", va="bottom", fontsize=6.2, color=NAVY_DARK, style="italic")
    panel_title(axB, "b", "The transfer win is broad across immune lineages",
                sub="neighbour chosen on the other cell types, per cell type", x_letter=-0.20)
    despine(axB)

    # ================= (c) the two conditioning regimes =================
    # x = annotation-only (feature-nearest) gap vs floor; y = observed-elsewhere (DE-profile) gap.
    g_ft = (per_ct["feature-nearest"] - per_ct[floor_col]).values
    g_de = (per_ct["DE-profile-nearest"] - per_ct[floor_col]).values
    sz = 14 + 0.55 * per_ct["n_held"].values
    axC.axhline(0, color=NAVY_DARK, lw=1.0, ls="--", zorder=1)
    axC.axvline(0, color=NAVY_DARK, lw=1.0, ls="--", zorder=1)
    # shaded quadrants
    # 0.35 was a hardcoded top inside axes that reach 0.38, so a hard-edged white strip sat
    # between the band and the panel top -- on a panel with no top spine, that edge reads as a
    # threshold the data respects. The band is drawn after the data, to the axis limit.
    # Filled and near-opaque, two coincident cell types printed as one marker -- and the two that
    # clear the floor (CD14 Mono and Mono, 0.001 apart in x and 0.008 in y) are the caption's own
    # "2 of 24". A light fill with a solid edge keeps the size encoding and shows both rings.
    # Hollow. Both axes are measured values, so the points cannot be dodged apart, and a light
    # fill was still not enough for the pair the caption counts: CD14 Mono and Mono differ by
    # 0.0009 in x and 0.0075 in y and printed as one marker against a stated "2 of 24". Two rings
    # read as two; the marker-size encoding survives.
    axC.scatter(g_ft, g_de, s=sz, facecolor="none", edgecolor=CONDITIONED_DARK,
                linewidth=1.0, alpha=0.95, zorder=4)
    axC.scatter([np.mean(g_ft)], [np.mean(g_de)], marker="D", s=46, c=NAVY,
                edgecolor="white", linewidth=0.9, zorder=6)
    # Up and left, the leader crossed the MAIT ring on its way to the diamond. Down and right
    # of the mean the panel is empty, so nothing is drawn over.
    axC.annotate("mean over cell types", xy=(np.mean(g_ft), np.mean(g_de)),
                 xytext=(np.mean(g_ft) + 0.050, np.mean(g_de) - 0.062), fontsize=6.2,
                 color=NAVY_DARK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=NAVY_DARK, lw=0.6))
    # The caption counts "2 of 24" to the right of the zero line. Those two are Mono
    # (+0.034, +0.207) and CD14 Mono (+0.035, +0.199): 0.0009 apart in x, 0.0075 in y, and with
    # 87 held cytokines each they draw at the SAME radius, so the pair printed as one thick ring
    # and the count could not be checked. Neither axis can be dodged -- both are measured -- so
    # the right limit is opened and each ring is named, which is what the count needs.
    axC.set_xlim(right=max(0.078, axC.get_xlim()[1]))
    for _nm, _dy in (("Mono", 6), ("CD14 Mono", -8)):
        _i = int(np.argmax(per_ct["celltype"].values == _nm.replace(" ", "_")))
        axC.annotate(_nm, xy=(g_ft[_i], g_de[_i]), xytext=(9, _dy),
                     textcoords="offset points", ha="left", va="center", fontsize=6.0,
                     color=CONDITIONED_DARK,
                     arrowprops=dict(arrowstyle="-", color=CONDITIONED_DARK, lw=0.5, alpha=0.8))
    # Anchored to the panel's right edge, this sat across the dashed zero line at x=0.00 on the
    # narrower plate. It is placed just right of that line instead, in the quadrant it names.
    # Opening the right limit for the Mono labels moved the ring cluster under this text. Every
    # point with y above 0.19 sits right of x = -0.05, so the band's top-LEFT corner is the one
    # corner of the quadrant that is empty.
    axC.text(0.03, 0.97, "transfer escapes\nthe floor", transform=axC.transAxes, ha="left",
             va="top", fontsize=6.6, color=CONDITIONED_DARK, style="italic")
    # The Plasmablast outlier sits at the extreme bottom-left corner of the panel (the widest
    # gap in both x and y), exactly where this label used to anchor — text landed on top of the
    # marker. Moved up and right, clear of the point, with a short leader back to it.
    # The leader ended on the single Plasmablast outlier, but the sentence describes the 22 of
    # 24 cell types left of the zero line. A quadrant label needs no leader.
    axC.text(0.15, 0.14, "annotation-only\nconditioning fails", transform=axC.transAxes,
             ha="left", va="bottom", fontsize=6.6, color=CLAY_DARK, style="italic")
    # same convention: real subtraction "−" between method and floor, plain hyphens within names.
    axC.set_xlabel("feature-nearest − floor  (annotation only)", fontsize=7.4)
    # "(observed elsewhere)" read as transferring the held cytokine's own effect from another cell
    # type; the full phrase "neighbour from other cell types" overran the panel height. The
    # neighbour is matched on the held cytokine's profile in those cell types.
    axC.set_ylabel("DE-profile-nearest − floor  (profile-matched)", fontsize=7.4)
    _y0, _y1 = axC.get_ylim()
    axC.axhspan(0, _y1, xmin=0.0, xmax=1.0, color=CONDITIONED, alpha=0.05, zorder=0)
    axC.set_ylim(_y0, _y1)
    panel_title(axC, "c", "Two conditioning regimes",
                sub="each point a cell type; marker size ~ held cytokines", x_letter=-0.34)
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
