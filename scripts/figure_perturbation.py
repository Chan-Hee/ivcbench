#!/usr/bin/env python
r"""Figure 5 (Nature-grade) — the PERTURBATION and MODALITY axes.

(a) C3 unseen-gene dissociation: best simple floor vs best conditioned across the 15 dataset x
    holdout cells (dumbbells; simple wins 15/15). (b) Ahlmann-Eltze in-frame linear-deep margin:
    ~0 on unseen compounds (reproducing the prior) but ~-0.24 on the immune unseen-gene axis.
    (c) C4 modality: per-CITE-marker recovery (predicted vs observed knockout delta); PD-1 / HLA
    class I recovered, PD-L1 not. (d) Immune-program recovery on the unseen-gene axis is
    degenerate-zero (88.7% exactly 0). Numbers from results_raw.csv + cite_marker_recovery.csv.

JEWEL DATA-MARK / NAVY-INK SYSTEM (one palette language across the plate):
    Structural NAVY ink is RESERVED for titles, rules, group-accent bars and brackets — it is never
    a data mark. The data marks come from the saturated jewel register so a reader can never confuse
    a structural rule with a model result.
    conditioned magenta #C13E8E = conditioned / deep model family  (one magenta everywhere for the
                                   deep-vs-simple contrast; CONDITIONED token)
    slate grey   #9AA3AD = simple / linear floor family  (one floor grey everywhere; SIMPLE_GREY)
    null grey    #C7CDD4 = null / background / "exactly 0"  (null / background only; NULL_GREY)
    coral-rust   #C24E32 = the warm counter-pole (the un-recovered PD-L1 marker only; CLAY_DARK)
    CITE markers (panel c) — restrained jewel trio: teal = recovered (jewel HYBRID), coral = the
    single PD-L1 miss (warm reads as a miss), null-grey = the anonymous background markers.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ivcbench.report.style import (set_pub_style, despine,  # noqa: E402
                                   NAVY, NAVY_DARK, NAVY_RAMP, CLAY_DARK,
                                   CONDITIONED, CONDITIONED_DARK, HYBRID, HYBRID_DARK,
                                   SIMPLE_GREY, SIMPLE_DARK, NULL_GREY, INK,
                                   GREY_MID, GREY_LITE, LEGEND_EC)

RESULTS = Path(__file__).resolve().parents[1] / "results"
SIMPLE = ["ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"]
DEEP = {"latent", "graph", "foundation", "hybrid"}

# ---- ONE jewel data-mark palette + navy structural ink (one concept = one colour) ---------------
C_DEEP = CONDITIONED       # conditioned / deep family DATA MARK  (saturated magenta #C13E8E)
C_DEEP_DARK = CONDITIONED_DARK  # legible-on-white magenta for deep-family TEXT (callouts / brackets)
C_SIMPLE = SIMPLE_GREY     # simple / linear floor family  (one cool slate-grey #9AA3AD everywhere)
C_SIMPLE_DARK = SIMPLE_DARK  # darker grey for simple-family TEXT (legible on white)
C_NULL = NULL_GREY         # null / background / "exactly 0"  (neutral grey #C7CDD4)
C_DUMBBELL = "#CDD6DD"     # dumbbell connector (cool neutral, lighter than the markers)
C_BAND = "#F4F6F8"         # faint cool grouping band (one band tint everywhere)
C_STRUCT = NAVY            # structural navy ink — group-accent bars / brackets / rules ONLY
C_STRUCT_DARK = NAVY_DARK  # darker navy for structural TEXT (bracket labels, legible on white)
# CITE-marker hues (panel c) — restrained jewel trio: recovered = teal; the single miss = coral.
C_HIT = HYBRID             # recovered markers (predicted ~ observed) — jewel teal reads as success
C_HIT_DARK = HYBRID_DARK   # legible-on-white teal for the recovered-marker callout text
C_MISS = CLAY_DARK         # PD-L1, the un-recovered marker (the warm coral counter-pole)
# per-dataset replicate-point fill/edge for the deep-vs-linear margin (panel b)
C_PT_FILL = "#E6B9D4"      # light magenta fill so the distribution strip reads as the conditioned family
C_PT_EDGE = CONDITIONED_DARK  # deep magenta edge — high contrast on white


# ---- typographic scale (effective pt at the ~6.6 in displayed width) --------------------------
FS_PANEL_LETTER = 9.4    # panel letter a/b/c/d (bold)
FS_PANEL_TITLE  = 7.8    # bold panel header (single line)
FS_AXIS_LABEL   = 6.9    # x/y axis labels
FS_INPANEL      = 6.1    # in-panel value / category labels
FS_TICK         = 5.7    # axis tick labels
FS_ANNOT        = 5.5    # italic notes / small callouts / sub-labels
FS_LEGEND       = 6.1    # legend entries
TITLE_Y         = 1.075  # baseline of the (single-line) panel title


def _head(ax, letter, title):
    """Left-aligned bold panel letter + single-line title on one tidy baseline.

    Single line (no wrap) keeps the header band shallow and the reading hierarchy calm; the bold
    letter aligns to the same baseline as the title, Nature-style.
    """
    ax.text(-0.155, TITLE_Y, letter, transform=ax.transAxes, fontsize=FS_PANEL_LETTER,
            fontweight="bold", va="bottom", ha="right", color=INK)
    ax.text(0.0, TITLE_Y, title, transform=ax.transAxes, fontsize=FS_PANEL_TITLE,
            fontweight="bold", va="bottom", ha="left", color=NAVY_DARK)


def _fmt_signed(v, dec=2):
    """Signed fixed-point with a TRUE unicode minus and NO negative-zero artefact.

    A value that rounds to zero is printed as a clean unsigned '0.00' (never '-0.00' or '+0.00').
    """
    s = f"{v:+.{dec}f}"
    zero = f"{0.0:.{dec}f}"            # e.g. '0.00'
    if s[1:] == zero:                 # rounds to zero -> drop the sign entirely
        return zero
    return s.replace("-", "−")   # ASCII hyphen -> unicode minus


def main():
    set_pub_style()
    # Generous design size; bbox_inches='tight' crops to the real content. The 2x2 plate is given
    # wide margins and open inter-panel gaps so nothing crowds or touches an edge.
    fig = plt.figure(figsize=(7.2, 7.4))
    # 2x2 grid. Panel c is forced square (set_aspect equal). Open wspace/hspace so every panel
    # breathes; a wide left margin holds the panel-a dataset names INSIDE the figure.
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.42,
                          width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0],
                          left=0.175, right=0.975, top=0.915, bottom=0.085)
    axA, axB, axC, axD = (fig.add_subplot(gs[i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)])

    d3 = pd.read_csv(RESULTS / "C3" / "results_raw.csv"); d3 = d3[d3.ran == True]  # noqa: E712
    M = "pearson_delta_ontarget"

    # =========================================================================================
    # (a) 15-cell dumbbells, grouped by dataset, labelled DIRECTLY with a coloured accent bar
    # =========================================================================================
    DS_NICE = {"chen": "Chen", "schmidt": "Schmidt", "shifrut": "Shifrut",
               "mccutcheon_CRISPRa": "McCutcheon (CRISPRa)",
               "mccutcheon_CRISPRi": "McCutcheon (CRISPRi)"}
    by_ds = {}
    for ds in sorted(d3.dataset.unique()):
        rows = []
        for hold in ["10", "25", "50"]:
            sub = d3[(d3.dataset == ds) & (d3.split == f"C3_true_lo_gene_{hold}")]
            if not len(sub):
                continue
            si = sub[sub.baseline.isin(SIMPLE)][M].max()
            co = sub[sub.family.isin(DEEP)][M].max()
            if si == si and co == co:
                rows.append((hold, si, co))
        if rows:
            by_ds[ds] = rows
    # order datasets by their mean simple-floor (descending so strongest floor sits on top)
    ds_order = sorted(by_ds, key=lambda d: -np.mean([r[1] for r in by_ds[d]]))

    yticks, ylabels, yi = [], [], 0
    group_spans = []
    first_row = None  # (y, co, si) of the topmost dumbbell, for the inline decode labels
    for gi, ds in enumerate(ds_order):
        rows = sorted(by_ds[ds], key=lambda r: -int(r[0]))  # 10 / 25 / 50 top-to-bottom
        y0 = yi
        for hold, si, co in rows:
            axA.plot([co, si], [yi, yi], color=C_DUMBBELL, lw=2.6, zorder=1,
                     solid_capstyle="round")
            axA.plot(co, yi, "o", color=C_DEEP, ms=5.0, zorder=3,
                     markeredgecolor="white", markeredgewidth=0.6)
            axA.plot(si, yi, "o", color=C_SIMPLE, ms=5.0, zorder=3,
                     markeredgecolor="white", markeredgewidth=0.6)
            if first_row is None:
                first_row = (yi, co, si)
            yticks.append(yi); ylabels.append(hold); yi += 1
        group_spans.append((y0, yi - 1, DS_NICE.get(ds, ds), gi % 2))
        yi += 0.7  # gap between dataset groups

    xlo, xhi = 0.0, 0.66
    # faint cool grouping band on every other dataset (a uniform visual-grouping device)
    for y0, y1, _, par in group_spans:
        if par:
            axA.axhspan(y0 - 0.5, y1 + 0.5, color=C_BAND, zorder=0)
    # DIRECT group labels: a short structural-navy accent bar at the left margin + the dataset name
    # beside it — the "row-group labelled directly" grammar, no rotated bracket / rule. Navy here is
    # STRUCTURAL ink (a grouping rule), not a data mark, so it stays navy by design.
    # The accent bar sits clearly LEFT of the two-digit %-held tick column so it never overlaps a
    # tick label ("10"/"25"/"50"); the name is right-aligned just left of the bar.
    X_BAR = -0.090      # accent-bar x (axes fraction), left of the tick-label column
    X_NAME = -0.115     # dataset name x, right-aligned just left of the bar
    for y0, y1, name, _ in group_spans:
        ymid = (y0 + y1) / 2.0
        axA.plot([X_BAR, X_BAR], [y0 - 0.30, y1 + 0.30], color=C_STRUCT, lw=2.8,
                 transform=axA.get_yaxis_transform(), clip_on=False, solid_capstyle="butt")
        axA.text(X_NAME, ymid, name, transform=axA.get_yaxis_transform(),
                 ha="right", va="center", fontsize=FS_TICK, color=INK, clip_on=False)

    axA.set_yticks(yticks); axA.set_yticklabels(ylabels, fontsize=FS_TICK, color=GREY_MID)
    # bottom margin opens a clear strip BELOW the lowest (Chen) dataset rows; deep enough that the
    # inline decode labels sit entirely below the bottom dumbbell (never overlapping a data mark).
    axA.set_ylim(-2.7, yi - 0.7)
    axA.set_xlim(xlo, xhi)
    axA.set_xticks([0.0, 0.2, 0.4, 0.6])
    # pad the %-held tick labels off the spine so the two digits sit in clear space (not clipped)
    axA.tick_params(axis="y", length=0, pad=3.0)
    # one micro column-head over the numeric %-held ticks
    axA.text(-0.012, 1.012, "% genes held", transform=axA.transAxes,
             ha="right", va="bottom", fontsize=FS_ANNOT, color=GREY_MID, style="italic")
    # MINIMAL inline decode (no separate boxed legend): label the two endpoints of the bottom
    # representative dumbbell directly — navy = conditioned/deep, grey = simple baseline. The
    # connecting bar is self-evident, so the old "per-cell gap" key is dropped entirely.
    fy, fco, fsi = first_row
    axA.annotate("conditioned / deep", xy=(fco, fy), xytext=(fco, fy - 1.55),
                 ha="center", va="top", fontsize=FS_LEGEND, color=C_DEEP_DARK,
                 arrowprops=dict(arrowstyle="-", color=C_DEEP, lw=0.6, shrinkA=2, shrinkB=3))
    axA.annotate("simple baseline", xy=(fsi, fy), xytext=(fsi, fy - 1.55),
                 ha="center", va="top", fontsize=FS_LEGEND, color=C_SIMPLE_DARK,
                 arrowprops=dict(arrowstyle="-", color=C_SIMPLE, lw=0.6, shrinkA=2, shrinkB=3))
    axA.set_xlabel("per-cell Pearson-$\\Delta$ vs control  (higher is better $\\rightarrow$)",
                   fontsize=FS_AXIS_LABEL)
    _head(axA, "a", "Unseen-gene axis: the floor is never beaten")
    despine(axA)

    # =========================================================================================
    # (b) Ahlmann-Eltze in-frame linear-deep margin
    # =========================================================================================
    margins, margin_pts = [], []
    for hold in ["10", "25", "50"]:
        gaps = []
        for ds in d3.dataset.unique():
            sub = d3[(d3.dataset == ds) & (d3.split == f"C3_true_lo_gene_{hold}")]
            if len(sub):
                gaps.append(sub[sub.family.isin(DEEP)][M].max() - sub[sub.baseline.isin(SIMPLE)][M].max())
        margins.append(("gene\n" + hold + "%", float(np.nanmean(gaps))))
        margin_pts.append(gaps)
    d5 = pd.read_csv(RESULTS / "C5" / "results_raw.csv"); d5 = d5[d5.ran == True]  # noqa: E712
    gc = d5[d5.split == "C5_global_compound_holdout"]
    cmp_margin = float(gc[gc.family.isin(DEEP | {"chemistry"})].pearson_delta.max()
                       - gc[gc.baseline.isin(SIMPLE)].pearson_delta.max())
    labels = [m[0] for m in margins] + ["OP3\ncompound"]
    vals = [m[1] for m in margins] + [cmp_margin]
    # navy = deep loses (immune unseen-gene); grey = deep ties the linear floor (compound prior)
    cols = [C_DEEP] * 3 + [C_SIMPLE]
    x = np.arange(len(vals))

    axB.set_ylim(-0.40, 0.085)
    halo = [pe.withStroke(linewidth=2.0, foreground="white")]
    # the delta = 0 reference line (labelled inline once on the OP3 bar, below, not twice)
    axB.axhline(0, color=GREY_MID, lw=0.9, ls=(0, (5, 3)), zorder=2)

    bars = axB.bar(x, vals, width=0.52, color=cols, edgecolor="white", linewidth=0.9, zorder=3)
    # the near-zero OP3 bar: a crisp outline so it reads as a real near-0 bar (not absent)
    bars[3].set_edgecolor(C_SIMPLE_DARK); bars[3].set_linewidth(1.1)
    # per-dataset replicate points: navy edge + slight jitter, overlaid as a distribution strip
    rng = np.random.default_rng(0)
    for i in range(3):
        pts = margin_pts[i]
        jit = 0.17 + rng.random(len(pts)) * 0.14
        axB.scatter(np.full(len(pts), i) + jit, pts, s=15, facecolor=C_PT_FILL,
                    edgecolor=C_PT_EDGE, linewidth=0.7, zorder=5)
    # in-bar numeric mean delta: plain white text dropped into the clear navy bar interior (NO box
    # / outline), well above each bar's highest per-dataset dot so it never collides with the marks
    for xi, v in zip(x[:3], vals[:3]):
        axB.text(xi, -0.060, _fmt_signed(v), ha="center", va="center", fontsize=FS_INPANEL,
                 weight="bold", color="white", zorder=6)
    # group bracket over the 3 gene bars, in the clear upper band — structural navy ink (a rule, not
    # a data mark)
    yb_top = 0.042
    axB.plot([-0.27, 2.27], [yb_top, yb_top], color=C_STRUCT_DARK, lw=1.1, zorder=4)
    for xx in (-0.27, 2.27):
        axB.plot([xx, xx], [yb_top, yb_top - 0.007], color=C_STRUCT_DARK, lw=1.1, zorder=4)
    axB.text(1.0, yb_top + 0.006, r"$\mathrm{deep}<\mathrm{linear}$",
             ha="center", va="bottom", fontsize=FS_ANNOT, color=C_STRUCT_DARK, weight="bold")
    # compound annotation — ONE consolidated region: the −0.01 value on the OP3 bar plus a single
    # caption-style line folding "no difference" + "matches prior" together. No competing blocks.
    axB.annotate(_fmt_signed(cmp_margin), xy=(3, cmp_margin), xytext=(3, cmp_margin - 0.045),
                 fontsize=FS_INPANEL, ha="center", va="top", color=C_SIMPLE_DARK, weight="bold",
                 arrowprops=dict(arrowstyle="-", color=C_SIMPLE_DARK, lw=0.7),
                 path_effects=halo)
    axB.text(3, cmp_margin - 0.090,
             "no difference\n(matches prior)",
             fontsize=FS_ANNOT, ha="center", va="top", color=C_SIMPLE_DARK, style="italic",
             linespacing=1.15)

    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=FS_TICK, linespacing=1.0)
    axB.set_xlim(-0.6, 3.55)
    axB.set_yticks([-0.4, -0.3, -0.2, -0.1, 0.0])
    axB.set_yticklabels([_fmt_signed(t, 1) for t in [-0.4, -0.3, -0.2, -0.1, 0.0]])
    axB.set_ylabel(r"best deep $-$ best linear  (Pearson-$\Delta$)", fontsize=FS_AXIS_LABEL)
    axB.tick_params(axis="y", labelsize=FS_TICK)
    _head(axB, "b", "Ahlmann-Eltze 2025, re-derived in-frame")
    despine(axB)

    # =========================================================================================
    # (c) C4 per-CITE-marker pred vs obs — restrained navy/clay, direct inline labels
    # =========================================================================================
    cm = pd.read_csv(RESULTS / "C4" / "cite_marker_recovery.csv")
    cm = cm[cm.held_frac_pct == 25]
    rval, pval = stats.pearsonr(cm.predDelta, cm.obsDelta_mean)
    nC = len(cm)
    # recovered markers read navy; the single PD-L1 miss reads clay; everything else is background.
    hl = {"PD-1 (CD279)": C_HIT, "HLA-E (class I)": C_HIT,
          "HLA-A (class I)": C_HIT, "PD-L1 (CD274)": C_MISS}
    lim = 0.50
    axC.axhline(0, color="#E6E6E6", lw=0.7, zorder=0)
    axC.axvline(0, color="#E6E6E6", lw=0.7, zorder=0)
    axC.plot([-lim, lim], [-lim, lim], color=GREY_LITE, ls=(0, (4, 3)), lw=0.9, zorder=1)
    # y=x label moved inboard along the diagonal (~15% toward centre) with a small perpendicular
    # offset, so clear whitespace separates it from both the top and right spines.
    yx = lim - 0.085
    axC.text(yx + 0.018, yx - 0.018, "$y = x$", fontsize=FS_ANNOT, color=GREY_MID,
             ha="center", va="center", rotation=45, rotation_mode="anchor",
             path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    # background markers (the 16 anonymous CITE markers): light grey, slight transparency
    n_bg = 0
    for _, r in cm.iterrows():
        if r.alias not in hl:
            axC.scatter(r.predDelta, r.obsDelta_mean, s=22, color=C_NULL, alpha=0.55,
                        edgecolor="white", linewidth=0.3, zorder=2)
            n_bg += 1
    for _, r in cm.iterrows():
        if r.alias in hl:
            axC.scatter(r.predDelta, r.obsDelta_mean, s=62, color=hl[r.alias],
                        edgecolor=INK, linewidth=0.9, zorder=5)
    # direct inline labels with short straight grey leaders (no curves: this is a pred-vs-obs
    # scatter, not a fit). Labels routed into clear space on the SAME side of y=x as their marker.
    mk = {r.alias: (r.predDelta, r.obsDelta_mean) for _, r in cm.iterrows()}
    LEAD = dict(arrowstyle="-", color="#9a9a9a", lw=0.5, shrinkA=1, shrinkB=3)
    callouts = {
        "PD-1 (CD279)":   ("PD-1\nrecovered",     (-0.30, -0.20), C_HIT_DARK),
        "HLA-A (class I)": ("HLA-A\nrecovered",    (0.21, -0.22), C_HIT_DARK),
        "PD-L1 (CD274)":  ("PD-L1\nnot recovered", (-0.22, 0.28), C_MISS),
        "HLA-E (class I)": ("HLA-E\nrecovered",    (0.21, 0.35), C_HIT_DARK),
    }
    for alias, (lab, txy, col) in callouts.items():
        mx, my = mk[alias]
        axC.annotate(lab, xy=(mx, my), xytext=txy,
                     fontsize=FS_ANNOT, color=col, weight="bold", ha="center", va="center",
                     linespacing=1.05, arrowprops=LEAD,
                     path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    # tidy key for the anonymous background markers (accounts for all n=20), bottom-left
    axC.scatter([-0.455], [-0.455], s=22, color=C_NULL, alpha=0.55, edgecolor="white",
                linewidth=0.3, zorder=2, clip_on=False)
    axC.text(-0.41, -0.455, f"+{n_bg} others", fontsize=FS_ANNOT, color=GREY_MID,
             va="center", ha="left")
    # stats box (r, P, n) in the clear lower-right corner
    pstr = "$P$ < 0.001" if pval < 1e-3 else f"$P$ = {pval:.3f}"
    axC.text(0.955, 0.04, f"$r$ = {rval:.2f}\n{pstr}\n$n$ = {nC} markers",
             transform=axC.transAxes, fontsize=FS_ANNOT, va="bottom", ha="right", linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=LEGEND_EC, lw=0.6, alpha=1.0))
    axC.set_xlabel(r"predicted knockout $\Delta$", fontsize=FS_AXIS_LABEL)
    axC.set_ylabel(r"observed knockout $\Delta$", fontsize=FS_AXIS_LABEL)
    axC.set_xlim(-lim, lim); axC.set_ylim(-lim, lim)
    axC.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4]); axC.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    axC.set_xticklabels([_fmt_signed(t, 1) for t in [-0.4, -0.2, 0.0, 0.2, 0.4]])
    axC.set_yticklabels([_fmt_signed(t, 1) for t in [-0.4, -0.2, 0.0, 0.2, 0.4]])
    axC.tick_params(axis="both", labelsize=FS_TICK)
    axC.set_aspect("equal", adjustable="box")
    _head(axC, "c", "Surface-protein panel (Frangieh CITE-seq)")
    despine(axC)

    # =========================================================================================
    # (d) C3 degenerate-zero immune-program recovery — failure->success ramp, DIRECT labels
    # =========================================================================================
    progcols = [c for c in d3.columns if c.startswith("aucell::")]
    pv = d3[d3.family.isin(DEEP)][progcols].values.flatten(); pv = pv[~np.isnan(pv)]
    n = pv.size
    az = np.abs(pv)
    # sequential failure -> success ramp, all within the editorial palette: null grey (degenerate
    # 0) -> mid navy-ramp -> deep navy (real signal). One coherent ramp, no off-palette amber/green.
    C_PARTIAL = NAVY_RAMP[2]   # mid navy (weak-but-present)
    C_REAL = NAVY              # deep navy (real signal)
    cats = [("exactly $0$", int((pv == 0).sum()), C_NULL),
            (r"$0 < |\mathrm{corr}| \leq 0.1$", int(((az > 0) & (az <= 0.1)).sum()), C_PARTIAL),
            (r"$|\mathrm{corr}| > 0.1$", int((az > 0.1).sum()), C_REAL)]
    pcts = [100 * c[1] / n for c in cats]
    yb = np.array([2.0, 1.0, 0.0])  # top (exactly 0) -> bottom (|corr|>0.1)
    h = 0.54
    axD.barh(yb, pcts, color=[c[2] for c in cats], height=h, edgecolor="white",
             linewidth=0.8, zorder=3)
    # x-axis runs the full 0-100 % with a GENEROUS right gutter so the count labels and the inboard
    # semantic tags sit in clear panel space, never flush to the right spine (the brief's
    # pull-right-columns-inward rule). All tags live INSIDE the axes.
    X_HI = 118.0          # extends past 100 % to leave a wide right whitespace gutter
    X_TAG = 100.0         # inboard anchor for the "null" / "real signal" semantic tags (well left of edge)
    for yi, c, p in zip(yb, cats, pcts):
        lab = f"{p:.1f}%  ({c[1]}/{n})"
        if c[0].startswith("exactly"):
            # the dominant bar: white count label dropped INSIDE the bar interior
            axD.text(p - 1.8, yi, lab, va="center", ha="right", fontsize=FS_INPANEL, weight="bold",
                     color="white", zorder=6)
        else:
            axD.text(p + 2.2, yi, lab, va="center", fontsize=FS_INPANEL, color=GREY_MID)
    axD.set_yticks(yb)
    axD.set_yticklabels([c[0] for c in cats], fontsize=FS_TICK)
    axD.set_ylim(-0.6, 2.6)
    axD.set_xlim(0, X_HI)
    axD.set_xticks([0, 25, 50, 75, 100])
    axD.set_xlabel(f"% of conditioned program-cells  ($n$ = {n})", fontsize=FS_AXIS_LABEL)
    axD.tick_params(axis="x", labelsize=FS_TICK)
    # DIRECT semantic end-tags, pulled INWARD to a fixed inboard column (x=100) with clear whitespace
    # to the right spine: "null" beside the grey top bar, "real signal" beside the bottom bar, each
    # coloured to its category. These read down the right edge of the bars, never off-axis.
    axD.text(X_TAG, 2.0, "null", fontsize=FS_ANNOT, color=GREY_MID, ha="left", va="center",
             weight="bold")
    axD.text(X_TAG, 0.0, "real\nsignal", fontsize=FS_ANNOT, color=NAVY_DARK, ha="left",
             va="center", weight="bold", linespacing=1.0)
    # compact in-panel note pulled inward to fill the mid-right space and balance against panel a
    axD.text(60, 1.0, "no held-target program\nis rebuilt above noise",
             fontsize=FS_ANNOT, color=GREY_MID, ha="left", va="center", style="italic",
             linespacing=1.1)
    axD.tick_params(axis="y", length=0, pad=2.5)
    _head(axD, "d", "Immune-program recovery: degenerate zeros")
    despine(axD)

    out = RESULTS / "_paper" / "figure_perturbation.png"
    # generous outer padding frames the whole plate with a comfortable white border (Nature margins)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.18, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
