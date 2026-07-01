#!/usr/bin/env python
"""Figure (Nature-grade) — the CELL-CONTEXT axis, immune lens, navy editorial system.

(a) C1 Kang per-lineage forest: scGen (conditioned) vs the best simple baseline, 95% bootstrap /
    n=3-seed CIs. scGen leads on CD14+ monocytes and CD8 T by point estimate; no lineage is a
    CI-separated win, so the panel makes the honest "leads, but within noise" claim.
(b) C5 program-vs-magnitude dissociation: type-I-IFN program recovery (x) vs bulk Pearson-delta (y)
    per model — only the chemistry-side FP-ridge clears BOTH gates; the latent/hybrid methods recover
    the program but not the magnitude; the simple baselines are the mirror image.
(c) C5 IFN recovery vs a compound-shuffle permutation null per lineage (z = 8-10, P < 5e-4).

All numbers are read from results_raw.csv + results/C5/ifn_shuffle_null.csv; nothing is hardcoded.
Layout: top row a | b ; full-width bottom row c.

DESIGN — one restrained navy editorial system shared across the 8-figure plate:
  * NAVY = the conditioned / observed / "winner" series everywhere;
  * ONE grey = the simple-baseline family everywhere; a separate pale grey = the permutation null;
  * methods are separated by MARKER SHAPE, not by a saturated rainbow of hues;
  * group labels are placed DIRECTLY beside their marks (no tall colour-decode legend) and each panel
    carries at most one small, tidy key.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ivcbench.report.style import (set_pub_style, despine, panel_title, style_legend,  # noqa: E402
                                   NAVY, NAVY_DARK, SLATE_BAND, SIMPLE_GREY, NULL_GREY,
                                   INK, GREY_MID, GREY_LITE)

RESULTS = Path(__file__).resolve().parents[1] / "results"
SIMPLE = ["ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"]

# ----------------------------------------------------------------------------------------------
# ONE restrained navy editorial palette (no rainbow). Methods differ by MARKER SHAPE, not hue.
#   NAVY        — the conditioned / observed / winning series (scGen, FP-ridge, observed recovery)
#   SLATE_BAND  — a single muted slate for the *other* conditioned-deep methods (CPA, STATE),
#                 kept clearly distinct from NAVY but in the same cool family (never a saturated hue)
#   SIMPLE_GREY — the simple-baseline family everywhere
#   NULL_GREY   — the permutation-null band / background grey only
C_NAVY = NAVY
C_NAVY_DARK = NAVY_DARK
C_SLATE = SLATE_BAND
C_FLOOR = SIMPLE_GREY
C_NULL = NULL_GREY
C_INK = INK
C_QUAD = "#EAF0F4"   # the lightest tint of the navy ramp — neutral "both-gates" quadrant wash

# model -> (colour, marker, size).  Shape carries identity; navy = the chemistry winner & scGen,
# slate = the other two conditioned-deep methods.  No saturated hue anywhere.
MODEL_STYLE = {
    "FP-ridge": (C_NAVY, "*", 230),
    "scGen":    (C_NAVY, "o", 58),
    "CPA":      (C_SLATE, "s", 46),
    "STATE":    (C_SLATE, "^", 50),
}

# shared body type sizes (effective ~6 pt at the ~0.93 display scale)
FS_AXLABEL = 7.0
FS_TICK = 6.6
FS_LABEL = 6.8     # direct inline data labels
FS_TAG = 6.0       # small tags / captions
FS_MICRO = 5.6     # micro annotation


def _leader(ax, x, y, tx, ty, text, color, ha, va, fs=FS_LABEL, weight="normal"):
    """Label at (tx,ty) data-coords with a thin leader back to the marker at (x,y).

    When the label sits at (almost) the same height as the marker, a plain straight leader is used
    (a right-angled 'angle' connector degenerates to a zero-length segment there); otherwise a soft
    right-angled leader keeps the routing tidy.
    """
    straight = abs(ty - y) < 1e-9
    cs = "arc3,rad=0" if straight else "angle,angleA=0,angleB=90,rad=2"
    ax.annotate(text, xy=(x, y), xytext=(tx, ty), ha=ha, va=va, fontsize=fs, color=color,
                weight=weight, zorder=7,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.5, connectionstyle=cs,
                                shrinkA=0.5, shrinkB=2.5))


def main():
    set_pub_style()
    fig = plt.figure(figsize=(7.1, 6.5))
    # Tighten the vertical band between the a/b row and panel c (the old layout left an oversized
    # void there); reserve a calm strip at the very bottom for ONE shared frameless legend.
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.12], height_ratios=[1.34, 0.96],
                          wspace=0.42, hspace=0.62, left=0.115, right=0.955,
                          top=0.915, bottom=0.150)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    # ======================== (a) C1 per-lineage forest ========================
    d1 = pd.read_csv(RESULTS / "C1" / "results_raw.csv"); d1 = d1[d1.ran == True]  # noqa: E712
    loct = d1[d1.split.str.startswith("C1_loct")]
    # 3-seed override where a multiseed scGen replication exists: plot the 3-seed mean + n=3 t-CI
    # (not the optimistic seed-0 draw), matching the seed-aggregated table.
    _ms = pd.read_csv(RESULTS / "_paper" / "multiseed_scgen_summary.csv")
    _ms = _ms[(_ms.cluster == "C1") & (_ms.n_seed >= 2)]
    MULTISEED = {r.lineage: (float(r.pearson_mean), float(r.pearson_sd)) for r in _ms.itertuples()}
    _T2 = 4.302653  # t(0.975, df=2)
    rows = []
    for s in loct.split.unique():
        sub = loct[loct.split == s]
        sg = sub[sub.baseline == "scGen"]
        bs = sub[sub.baseline.isin(SIMPLE)].sort_values("pearson_delta").iloc[-1:]
        if not (len(sg) and len(bs)):
            continue
        sgr, bsr = sg.iloc[0].copy(), bs.iloc[0]
        _lin = s.replace("C1_loct_", "")
        if _lin in MULTISEED:
            _m, _sd = MULTISEED[_lin]; _se = _sd / (3 ** 0.5)
            sgr["pearson_delta"] = _m
            sgr["pearson_delta_lo"] = _m - _T2 * _se
            sgr["pearson_delta_hi"] = _m + _T2 * _se
        delta = float(sgr.pearson_delta) - float(bsr.pearson_delta)
        win = float(sgr.pearson_delta_lo) > float(bsr.pearson_delta_hi)
        rows.append(dict(lin=_lin, sg=sgr, bs=bsr, delta=delta, win=win))
    rows.sort(key=lambda d: d["delta"])  # smallest delta at bottom, largest (scGen lead) at top
    # pretty lineage labels
    PRETTY = {"Mono_CD14": "CD14 mono", "Mono_FCGR3A": "CD16 mono", "CD4T": "CD4 T",
              "CD8T": "CD8 T", "Mk": "Megakaryo.", "DC": "DC", "NK": "NK", "B": "B"}
    lins = [PRETTY.get(r["lin"], r["lin"]) for r in rows]
    y = np.arange(len(rows))

    floor_x = [float(r["bs"].pearson_delta) for r in rows]
    ref = float(np.median(floor_x))

    # faint vertical reference at the pooled-median baseline (drawn first, behind the data)
    axA.axvline(ref, color=C_FLOOR, ls=(0, (1, 1.6)), lw=0.8, zorder=1)

    for yi, r in zip(y, rows):
        for which, col, off in [("bs", C_FLOOR, -0.165), ("sg", C_NAVY, 0.165)]:
            rr = r[which]
            lo, hi, mid = float(rr.pearson_delta_lo), float(rr.pearson_delta_hi), float(rr.pearson_delta)
            axA.plot([lo, hi], [yi + off] * 2, color=col, lw=1.7, solid_capstyle="round", zorder=2)
            axA.plot(mid, yi + off, "o", color=col, ms=4.6, zorder=3, mec="white", mew=0.8)

    # honest lead/within-noise tag on the single best point-lead row (CD14 mono, top), placed in the
    # genuine whitespace BELOW-LEFT of that row's scGen marker, well inside the axes so it clears the
    # right margin by a wide margin — a direct inline note, no legend glyph, no leader running to edge.
    top = rows[-1]
    if top["delta"] > 0:
        axA.text(float(top["sg"].pearson_delta_lo) - 0.012, len(rows) - 1 - 0.34,
                 "scGen leads (within CI)", fontsize=FS_MICRO, color=C_NAVY_DARK,
                 ha="right", va="center", weight="normal", zorder=6)

    # pooled-median reference label in the dead space just above the x-axis spine, well below the data,
    # right-anchored so its whole right edge clears the dotted median guide (the guide never bisects it).
    axA.text(ref - 0.035, -0.58, "pooled median baseline", fontsize=FS_MICRO, color=C_FLOOR,
             ha="right", va="center", zorder=4)

    axA.set_yticks(y)
    axA.set_yticklabels(lins, fontsize=FS_TICK)
    axA.set_xlabel(r"Pearson-$\Delta$, IFN-$\beta$ response   (higher better)", fontsize=FS_AXLABEL,
                   color=C_INK)
    axA.set_xlim(0.40, 1.00)
    axA.set_ylim(-0.70, len(rows) - 0.18)
    axA.set_xticks([0.4, 0.6, 0.8, 1.0])
    axA.tick_params(colors=C_INK, labelsize=FS_TICK)
    despine(axA)
    # NOTE: the navy "scGen (conditioned)" vs grey "best simple baseline" key is NOT drawn here.
    # It shares its colour semantics with panel c, so both fold into ONE frameless figure-level
    # legend centred along the bottom (built after panel c), matching the reference's single legend.

    # ======================== (b) C5 program-vs-magnitude ========================
    d5 = pd.read_csv(RESULTS / "C5" / "results_raw.csv"); d5 = d5[d5.ran == True]  # noqa: E712
    loc5 = d5[d5.split.str.startswith("C5_loct")]
    g = loc5.groupby("baseline").agg(ifn=("aucell::type_I_IFN", "mean"),
                                     bulk=("pearson_delta", "mean"),
                                     fam=("family", "first")).reset_index()
    floor = g[g.baseline.isin(SIMPLE)].bulk.max()

    XLO, XHI, YLO, YHI = -0.16, 1.05, -0.06, 0.50
    THR = 0.5
    axB.set_xlim(XLO, XHI)
    axB.set_ylim(YLO, YHI)
    # neutral "both gates cleared" quadrant — the lightest navy-ramp tint, NO frame (reads as a tint,
    # not a boxed inset); the dashed threshold lines below already delimit it.
    axB.add_patch(Rectangle((THR, floor), XHI - THR, YHI - floor, facecolor=C_QUAD,
                            edgecolor="none", lw=0.0, zorder=0, clip_on=True))
    axB.axhline(floor, color=C_FLOOR, ls="--", lw=0.9, zorder=1)
    axB.axvline(THR, color=C_FLOOR, ls=(0, (1, 1.6)), lw=0.8, zorder=1)
    # quadrant title: ONE line hugging the top of the box, right-aligned (clears the FP-ridge mark/label
    # to its left); the recovery-threshold caption hangs lower-left by the vertical guide so the two
    # never collide.
    axB.text(0.985, YHI - 0.018, "both gates cleared", fontsize=FS_TAG, color=C_NAVY_DARK,
             style="italic", ha="right", va="top", zorder=3)
    axB.text(1.02, floor - 0.014, f"best simple\nbaseline = {floor:.2f}", color=C_FLOOR, fontsize=FS_TAG,
             ha="right", va="top", linespacing=1.0, zorder=5)
    axB.text(THR + 0.018, floor + 0.058, "program-recovery\ngate", color=GREY_MID, fontsize=FS_MICRO,
             ha="left", va="bottom", linespacing=1.0, zorder=2)

    gi = g.set_index("baseline")
    # the three high simple baselines pile up at ifn=0 (linear-PCA, donor-shift, CINEMA-OT) at
    # near-equal bulk (~0.25-0.27). Display-jitter them into a tight vertical column so each grey dot
    # is resolvable; the real datum is unchanged (ifn=0, bulk as in the CSV) — only the on-screen x/y
    # is nudged, exactly like display jitter.
    floor_stack = [b for b in ("linear-PCA", "donor-shift", "CINEMA-OT") if b in gi.index]
    floor_stack = sorted(floor_stack, key=lambda b: -float(gi.loc[b, "bulk"]))
    jit = {b: dx for b, dx in zip(floor_stack, (-0.010, -0.045, -0.080))}
    yjit = {"STATE": -0.013, "CPA": +0.013}

    for _, r in g.iterrows():
        star = r.baseline == "FP-ridge"
        col, mk, sz = MODEL_STYLE.get(r.baseline, (C_FLOOR, "o", 46))
        xpos = r.ifn + jit.get(r.baseline, 0.0)
        ypos = r.bulk + yjit.get(r.baseline, 0.0)
        axB.scatter(xpos, ypos, s=sz, marker=mk, color=col,
                    edgecolor=C_INK, lw=1.0 if star else 0.6, zorder=5)

    # the x=0 baseline column: ONE italic group caption above the cluster, then the individual names
    # placed beside their marks with SHORT leaders. The label column is kept tight (small vertical
    # spread) so the five-deep leader fan is shortened to a calm, low-density set, matching the
    # reference's restrained annotation.
    by = [float(gi.loc[b, "bulk"]) for b in floor_stack]
    bx = [float(gi.loc[b, "ifn"]) + jit[b] for b in floor_stack]
    axB.text(max(bx) + 0.012, max(by) + 0.050, "linear baselines", fontsize=FS_TAG, color=C_FLOOR,
             ha="left", va="bottom", style="italic", zorder=3)
    lab_x = float(np.mean(bx)) + 0.030
    # tighter spread around the actual cluster height keeps the leaders short and parallel
    name_y = {b: yv for b, yv in zip(floor_stack, (0.218, 0.176, 0.134))}
    for b in floor_stack:
        mx, my = float(gi.loc[b, "ifn"]) + jit[b], float(gi.loc[b, "bulk"])
        _leader(axB, mx, my, lab_x, name_y[b], b, C_FLOOR, ha="left", va="center", fs=FS_TAG)

    # cell-mean / ctrl-pred sit lower; label each directly beside its own point (short straight note)
    for b, dy, dx in [("cell-mean", 0.000, 0.028), ("ctrl-pred", -0.006, 0.034)]:
        if b in gi.index:
            axB.annotate(b, (gi.loc[b, "ifn"], gi.loc[b, "bulk"]),
                         (gi.loc[b, "ifn"] + dx, gi.loc[b, "bulk"] + dy),
                         fontsize=FS_TAG, ha="left", va="center", color=C_FLOOR)

    # the four conditioned methods: direct labels adjacent to their marks with SHORT, STRAIGHT,
    # NON-CROSSING leaders. FP-ridge sits inside the quadrant (label to its lower-left); scGen is
    # labelled at its own mark height. CPA and STATE sit almost on top of each other at the bottom-right
    # (CPA ~0.79/0.05, STATE ~0.75/0.05), so their labels are split vertically — CPA ABOVE the cluster
    # and STATE BELOW it — each with its OWN short leader back to its OWN marker so both read clearly.
    _leader(axB, gi.loc["FP-ridge", "ifn"], gi.loc["FP-ridge", "bulk"], THR - 0.024, 0.405,
            "FP-ridge", C_NAVY_DARK, ha="right", va="bottom", fs=FS_LABEL + 0.6, weight="bold")
    _leader(axB, gi.loc["scGen", "ifn"], gi.loc["scGen", "bulk"], 0.610, gi.loc["scGen", "bulk"],
            "scGen", C_NAVY_DARK, ha="right", va="center", fs=FS_LABEL, weight="bold")
    _leader(axB, gi.loc["CPA", "ifn"], gi.loc["CPA", "bulk"] + yjit["CPA"], 0.905, 0.150,
            "CPA", C_SLATE, ha="left", va="bottom", fs=FS_LABEL, weight="bold")
    _leader(axB, gi.loc["STATE", "ifn"], gi.loc["STATE", "bulk"] + yjit["STATE"], 0.905, -0.038,
            "STATE", C_SLATE, ha="left", va="top", fs=FS_LABEL, weight="bold")

    axB.set_xlabel(r"type-I IFN program recovery   (AUCell-$\Delta$ corr.)", fontsize=FS_AXLABEL, color=C_INK)
    axB.set_ylabel(r"bulk Pearson-$\Delta$   (higher better)", fontsize=FS_AXLABEL, color=C_INK)
    axB.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axB.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    axB.tick_params(colors=C_INK, labelsize=FS_TICK)
    despine(axB)

    # ======================== (c) C5 IFN recovery vs shuffle null ========================
    # A wide null-vs-observed panel has a large empty middle (null near 0, observed ~0.7-0.85). A soft
    # x-axis break compresses that void to a thin gutter; every plotted value is the real datum (only its
    # on-screen x is remapped, like display jitter), and the tick labels show the true underlying values.
    nu = pd.read_csv(RESULTS / "C5" / "ifn_shuffle_null.csv")
    nu = nu.sort_values("null_z").reset_index(drop=True)
    yy = np.arange(len(nu))
    n_perm = int(nu.n_perm.iloc[0])
    p_floor = 1.0 / n_perm
    p_exp = int(np.floor(np.log10(p_floor)))
    p_mant = p_floor / (10.0 ** p_exp)
    p_floor_str = rf"P < {p_mant:g}$\times$10$^{{{p_exp}}}$"

    LO, HI = -0.22, 1.02
    GAP0, GAP1, GAPW = 0.20, 0.60, 0.07
    shift = (GAP1 - GAP0) - GAPW

    def xt(v):
        v = np.asarray(v, dtype=float)
        out = np.where(v <= GAP0, v,
                       np.where(v >= GAP1, v - shift,
                                GAP0 + (v - GAP0) / (GAP1 - GAP0) * GAPW))
        return out if out.shape else float(out)

    axC.set_xlim(xt(LO), xt(HI))
    axC.set_ylim(-0.62, len(nu) - 0.34)

    PRETTY_C = {"T_cells": "T cells", "Mono": "Mono", "NK": "NK", "B": "B"}

    axC.axvline(xt(0.0), color=GREY_MID, lw=0.6, zorder=1)
    for yi, r in zip(yy, nu.itertuples()):
        lo5, hi95 = xt(r.null_5pct), xt(r.null_95pct)
        axC.barh(yi, hi95 - lo5, left=lo5, height=0.44, color=C_NULL, zorder=1)
        axC.plot([xt(r.null_mean)], [yi], "|", color=GREY_MID, ms=9, mew=1.1, zorder=2)
        xobs = xt(r.obs_IFN_recovery)
        # effect-size arrow: tail on the null 95th-pct edge, head at the observed navy diamond
        axC.annotate("", xy=(xobs - 0.014, yi), xytext=(hi95, yi),
                     arrowprops=dict(arrowstyle="-|>", color=C_NAVY, lw=1.1, alpha=0.9,
                                     shrinkA=0, shrinkB=0, mutation_scale=8), zorder=3)
        axC.plot([hi95, hi95], [yi - 0.10, yi + 0.10], color=GREY_MID, lw=1.0, zorder=3)
        axC.plot([xobs], [yi], "D", color=C_NAVY, ms=6.5, mec=C_INK, mew=0.7, zorder=4)
        axC.text(xobs + 0.032, yi + 0.04, f"z = {r.null_z:.1f}", fontsize=FS_LABEL, va="bottom",
                 ha="left", color=C_INK, weight="bold")
        axC.text(xobs + 0.032, yi - 0.05, p_floor_str, fontsize=FS_MICRO, va="top",
                 ha="left", color=GREY_MID)

    # axis-break glyph at the gutter — a pair of clearer, larger double-slash strokes drawn on the
    # x-axis spine so the discontinuity reads cleanly at print size (composition only; the break
    # location and all plotted values are unchanged).
    bxk = GAP0 + GAPW / 2.0
    y0 = axC.get_ylim()[0]
    dh = 0.15
    for off in (-0.016, 0.016):
        axC.plot([bxk + off - 0.013, bxk + off + 0.013], [y0 - dh, y0 + dh],
                 color=INK, lw=1.1, clip_on=False, zorder=6, solid_capstyle="round")

    real_ticks = [-0.1, 0.0, 0.1, 0.2, 0.7, 0.8, 0.9]
    axC.set_xticks([xt(t) for t in real_ticks])
    axC.set_xticklabels([(f"{t:g}".replace("-", "−") if t != 0 else "0") for t in real_ticks],
                        fontsize=FS_TICK)
    axC.set_yticks(yy)
    axC.set_yticklabels([PRETTY_C.get(l, l) for l in nu.lineage], fontsize=FS_TICK)
    axC.set_xlabel("IFN-program recovery   (higher better)", fontsize=FS_AXLABEL, color=C_INK)
    axC.tick_params(colors=C_INK, labelsize=FS_TICK)
    despine(axC)
    # NOTE: panel c's key is NOT drawn boxed at the right edge any more — it folds into the single
    # frameless figure-level legend centred along the bottom (built below).

    # ======================== panel letters + concise titles ========================
    FS_HEAD = 8.2
    FS_LETTER = 9.8

    def head(ax, letter, title):
        panel_title(ax, letter, title, sub=None, fs_title=FS_HEAD, fs_letter=FS_LETTER)

    def headC(ax, letter, title):
        panel_title(ax, letter, title, sub=None, x_letter=-0.058, y=1.120,
                    fs_title=FS_HEAD, fs_letter=FS_LETTER)

    head(axA, "a", "Cytokine transfer, per immune lineage")
    head(axB, "b", "Program recovery vs magnitude")
    headC(axC, "c", "IFN recovery is real signal, not chance")

    # ======================== ONE shared frameless bottom legend ========================
    # The navy / grey colour semantics are identical across panels a and c (navy = conditioned /
    # observed winner; grey = simple-baseline / null). A single calm, frameless row centred along the
    # bottom of the plate decodes the whole figure — matching the reference's one bottom legend and
    # removing all three boxed per-panel keys. Panel b is labelled directly inline, so it needs no key.
    leg_handles = [
        Line2D([0], [0], marker="o", color=C_NAVY, lw=1.7, ms=4.6, mec="white", mew=0.8,
               label="conditioned / observed (scGen, FP-ridge)"),
        Line2D([0], [0], marker="o", color=C_FLOOR, lw=1.7, ms=4.6, mec="white", mew=0.8,
               label="simple baseline"),
        Patch(fc=C_NULL, ec="none", label="compound-shuffle null (5–95%)"),
        Line2D([0], [0], marker="|", color=GREY_MID, lw=0, ms=9, mew=1.1, label="null mean"),
    ]
    leg = fig.legend(handles=leg_handles, loc="lower center", ncol=4, frameon=False,
                     fontsize=FS_TAG, handletextpad=0.5, columnspacing=1.6,
                     bbox_to_anchor=(0.5, 0.012))
    leg.set_zorder(8)

    out = RESULTS / "_paper" / "figure_cellcontext.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
