#!/usr/bin/env python
r"""Figure (Nature-Methods-grade): method x split RANKING HEATMAP — the scPerturBench
Fig 2b/4a analog for the immune-aware benchmark. Rows = methods grouped by model family,
columns = evaluation splits grouped by generalisation axis. The near block-diagonal data
structure (broadly-applicable simple/latent methods vs perturbation-only methods) is split
into TWO sub-panels so the axis dissociation reads at a glance:

  panel a  — cell-context / modality / donor axes  (methods scored on cell-state transfer);
  panel b  — perturbation axis  (methods scored on unseen-perturbation extrapolation).

EDITORIAL SYSTEM (matches the author's own Section2 reference plate):
  * ONE restrained navy diverging scale drives every cell FILL = raw Pearson-delta, centred at
    zero. A muted terracotta encodes negative (worse than control); a warm near-white sits at
    zero; the author's navy ramp encodes positive (better than control). The scale is the shared
    style.floor_diverging_cmap(), so this figure sits in the same navy plate as its siblings.
  * the printed NUMBER is the SAME raw Pearson-delta (luminance-contrast ink, no halo).
  * row-groups (model families) are labelled DIRECTLY in a dedicated left gutter lane with a single
    muted slate accent bar + an inline family name — the accent bar lives in its OWN clean gutter,
    never touching or underlapping a cell — NO separate tall colour-decode legend (the reference's
    grammar). The only legend is one tidy bottom strip.
  * caveat micro-glyphs (two clearly DISTINCT marks, no decode legend needed):
      - a single thin corner NOTCH (lower-left) = ADAPTED (re-fit / non-native to that split);
      - a sparse diagonal HATCH over the cell = SHARED simple baseline (a shared floor, not a
        native run).
  * flat light-grey cell = method not applicable / not run on that split.

  Panels a (cell-context / modality / donor) and b (perturbation) are STACKED a-over-b at a shared
  cell size and a shared left gutter, so cell SIZE is constant and the two blocks read as deliberate
  siblings on one grid (no side-by-side dwarfing of the 5-column perturbation block by the
  16-column cell-context block). The single bottom legend sits in the clear space beside panel b.

Every plotted value is computed from results/{C1,C3,C4,C5}/results_raw.csv; no hardcoded numbers.
The set of plotted methods/columns/values is IDENTICAL to the prior figure — only composition,
layout, colour, spacing and typography were redesigned.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ivcbench.report.style import (  # noqa: E402
    set_pub_style, floor_diverging_cmap, NAVY_DARK, SLATE_BAND, INK, GREY_MID, GREY_LITE,
    LEGEND_EC, CLAY_DARK,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"

# family display order (rows): simple floors first, then conditioned families.
FAM_ROWS = [
    ("simple", ["ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"]),
    ("latent", ["scGen", "CPA", "chemCPA"]),
    ("graph", ["GEARS", "AttentionPert"]),
    ("foundation", ["scGPT"]),
    ("hybrid", ["STATE"]),
    ("chemistry", ["FP-ridge"]),
    ("shift", ["linear-shift-KOemb"]),
    ("opt-transport", ["CINEMA-OT", "CellOT"]),
]
# human-readable family names for the inline gutter group labels (the reference labels its
# row-groups directly; no colour-decode legend). Kept short so they sit cleanly in the gutter.
FAM_LABEL = {
    "simple": "simple", "latent": "latent", "graph": "graph", "foundation": "foundation",
    "hybrid": "hybrid", "chemistry": "chemistry", "shift": "KO-shift", "opt-transport": "OT",
}
# short, gutter-safe display names for the longest method tokens.
METHOD_SHORT = {
    "linear-shift-KOemb": "lin-shift-KO",
    "AttentionPert": "AttnPert",
}
# clean column display labels: the cluster/dataset name is shown ONCE in the band above the
# ticks, so the tick itself carries only the short lineage/dataset name.
COL_DISPLAY = {
    "C1·B": "B", "C1·CD4T": "CD4T", "C1·CD8T": "CD8T", "C1·DC": "DC",
    "C1·Mk": "Mk", "C1·Mono_CD14": "Mono14", "C1·Mono_FCGR3A": "Mono16",
    "C1·NK": "NK", "C1·LODO": "Kang", "C2·Soskic": "Soskic",
    "C5·B": "B", "C5·Mono": "Mono", "C5·NK": "NK", "C5·T_cells": "T",
    "C3·chen": "Chen", "C3·McCi": "McCutcheon-i", "C3·McCa": "McCutcheon-a", "C3·Schm": "Schmidt",
    "C3·Shif": "Shifrut", "C5·cpd": "compound",
    "C4·RNA": "RNA", "C4·prot": "protein",
}
# per-column cluster id (drives the dataset band above the ticks).
COL_CLUSTER = {
    "C1·B": "C1", "C1·CD4T": "C1", "C1·CD8T": "C1", "C1·DC": "C1", "C1·Mk": "C1",
    "C1·Mono_CD14": "C1", "C1·Mono_FCGR3A": "C1", "C1·NK": "C1", "C1·LODO": "C1",
    "C2·Soskic": "C2",
    "C5·B": "C5", "C5·Mono": "C5", "C5·NK": "C5", "C5·T_cells": "C5", "C5·cpd": "C5",
    "C3·chen": "C3", "C3·McCi": "C3", "C3·McCa": "C3", "C3·Schm": "C3", "C3·Shif": "C3",
    "C4·RNA": "C4", "C4·prot": "C4",
}
# dataset tag shown INLINE in each dataset band. Dataset NAMES only (the internal Cx codes are
# dropped from the figure — a reader has no reason to memorise them; the full roster is in the
# caption). One calm slate for every band: datasets are a grouping, not a semantic axis.
CLUSTER_TAG = {"C1": "Kang", "C2": "Soskic", "C3": "CRISPR", "C4": "Frangieh", "C5": "OP3"}
# generalisation-axis -> the panel it belongs to (block-diagonal split).
PANEL_A_AXES = ["cell-context", "modality", "donor"]
PANEL_B_AXES = ["perturbation"]
AXIS_TITLE = {"cell-context": "cell-context", "perturbation": "perturbation",
              "modality": "modality", "donor": "donor"}
AXIS_SHORT = {"modality": "modality", "donor": "donor"}

# ---- the single restrained navy editorial diverging scale (shared across the 8-figure plate) ----
CMAP = floor_diverging_cmap()       # clay (worse) -> warm near-white (0) -> author navy (better)
NA_FC = "#f0f0f0"                   # one flat light grey: not-applicable / not run
GRID = "#ffffff"                    # cell gridlines (white seams, like the reference)
ACCENT = SLATE_BAND                 # ONE calm slate for the family row-group accent bars
BAND_GREY = SLATE_BAND              # ONE calm slate for the dataset bands (no rainbow)
SEP = "#d8dde1"                     # faint within-panel axis separators
ADAPT_C = "#5a5a5a"                 # neutral grey caveat-triangle stroke (recedes behind the fill)
LUM_T = 0.55                        # luminance threshold: white<->ink text on cell fill
NEG_INK = CLAY_DARK                 # clay ink for negative numbers on a light fill


def _ran(c):
    d = pd.read_csv(RESULTS / c / "results_raw.csv")
    return d[d["ran"] == True]  # noqa: E712


def _norm_action(a):
    if a == "run_floor":
        return "floor"
    if a == "run_adapted":
        return "adapted"
    return "native"


def _action_of(sub):
    acts = set(sub.get("action", []))
    if "run_floor" in acts:
        return "floor"
    if "run_adapted" in acts:
        return "adapted"
    return "native"


def long_table():
    rows = []

    def add(method, family, axis, col, score, action):
        if score == score:  # not NaN
            rows.append(dict(method=method, family=family, axis=axis, col=col,
                             score=float(score), action=action))

    d1 = _ran("C1")
    for s in sorted(x for x in d1.split.unique() if x.startswith("C1_loct")):
        lin = s.replace("C1_loct_", "")
        for _, r in d1[d1.split == s].iterrows():
            add(r.baseline, r.family, "cell-context", f"C1·{lin}", r.pearson_delta, _norm_action(r.action))
    lodo = d1[d1.split.str.startswith("C1_lodo")]
    for b in lodo.baseline.unique():
        sub = lodo[lodo.baseline == b]
        add(b, sub.family.iloc[0], "donor", "C1·LODO", sub.pearson_delta.mean(), _action_of(sub))

    p2 = RESULTS / "soskic_donor_axis.csv"
    if p2.exists():
        d2 = pd.read_csv(p2)
        for b in d2.model.unique():
            sub = d2[d2.model == b]
            fam = sub.family.iloc[0] if "family" in sub.columns else ("latent" if b == "scGen" else "simple")
            add(b, fam, "donor", "C2·Soskic", sub.pearson_delta.mean(), "native")

    d3 = _ran("C3")
    d3 = d3[d3.split == "C3_true_lo_gene_10"]   # headline 10% LO-gene split (not pooled 10/25/50%)
    for ds in sorted(d3.dataset.unique()):
        for b in d3[d3.dataset == ds].baseline.unique():
            sub = d3[(d3.dataset == ds) & (d3.baseline == b)]
            add(b, sub.family.iloc[0], "perturbation", f"C3·{ds.replace('shifrut','Shif').replace('schmidt','Schm').replace('mccutcheon_CRISPRi','McCi').replace('mccutcheon_CRISPRa','McCa')[:6]}",
                sub.pearson_delta_ontarget.mean(), _action_of(sub))

    d4 = _ran("C4")
    for ds in sorted(d4.dataset.unique()):
        mod = "RNA" if "rna" in str(ds).lower() else "prot"   # case-insensitive: frangieh_rna vs frangieh_RNA
        for b in d4[d4.dataset == ds].baseline.unique():
            sub = d4[(d4.dataset == ds) & (d4.baseline == b)]
            add(b, sub.family.iloc[0], "modality", f"C4·{mod}", sub.pearson_delta_ontarget.mean(),
                _action_of(sub))

    d5 = _ran("C5")
    for s in sorted(x for x in d5.split.unique() if x.startswith("C5_loct")):
        lin = s.replace("C5_loct_", "")
        for _, r in d5[d5.split == s].iterrows():
            add(r.baseline, r.family, "cell-context", f"C5·{lin}", r.pearson_delta, _norm_action(r.action))
    gc = d5[d5.split == "C5_global_compound_holdout"]
    for _, r in gc.iterrows():
        add(r.baseline, r.family, "perturbation", "C5·cpd", r.pearson_delta, _norm_action(r.action))

    AM = RESULTS.parent / "outputs" / "additional_models"
    ck = AM / "cellot_kang_by_lineage.csv"
    if ck.exists():
        for _, r in pd.read_csv(ck).query("metric == 'pearson_delta'").iterrows():
            add("CellOT", "opt-transport", "cell-context", f"C1·{r.lineage}", r.cellot_score, "native")
    cs = AM / "cellot_summary.csv"
    if cs.exists():
        s = pd.read_csv(cs)
        so = s[(s.dataset == "soskic2022") & (s.metric == "pearson_delta")]
        if len(so):
            add("CellOT", "opt-transport", "donor", "C2·Soskic", float(so.model_score.iloc[0]), "adapted")
    cc = AM / "chemcpa_op3_unseen_compound_summary.csv"
    if cc.exists():
        add("chemCPA", "latent", "perturbation", "C5·cpd", float(pd.read_csv(cc).chemCPA_score.iloc[0]), "native")
    return pd.DataFrame(rows)


def _fmt(v):
    """Uniform 2-dp leading-decimal label for every in-cell number. Every value is a Pearson-delta
    in (-1, 1), so the leading zero is dropped (0.45 -> '.45'). CRITICAL — one consistent precision
    for EVERY cell: a value that rounds to zero prints '.00' (NOT a bare '0' and NOT a negative-zero
    token '−.00'), exactly matching the precision used for positives and negatives, so the grid
    reads uniformly. Underlying values are never changed — formatting only."""
    r = round(v, 2)
    if r == 0:                             # rounds to zero -> '.00', never bare '0' nor '−.00'
        return ".00"
    s = f"{r:.2f}"                          # e.g. '0.45', '-0.24'
    if s.startswith("-0."):
        s = "−." + s[3:]                   # unicode minus, leading-decimal
    elif s.startswith("-"):
        s = "−" + s[1:]
    elif s.startswith("0."):
        s = "." + s[2:]
    return s


def _lum(rgb):
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ---- gutter geometry shared by every panel (data units; identical so panels align exactly) ----
# A clean DEDICATED gutter, in the REFERENCE lane order, left-to-right:
#       family label (rotated) | accent bar (own lane) | method names | cells
# Each element has its own non-overlapping lane. The widest method name is 'CINEMA-OT' (~2.42 cells),
# so the names lane is ~2.5 units wide; the accent bar sits just left of that, and the rotated family
# label just left of the bar. The bar never touches a cell, and no method name ever overlaps the bar
# or the rotated label (the prior collision bug).
NAME_X    = -0.65                    # right edge of right-anchored method names (0.15 left of cells)
NAME_LMAX = 2.55                     # widest name reaches this far left of NAME_X
BAR_X1    = NAME_X - NAME_LMAX - 0.18   # accent-bar lane, just left of the longest name
BAR_X0    = BAR_X1 - 0.12
FAMLAB_X  = BAR_X0 - 0.30            # rotated multi-row family label, just left of the accent bar
FAMLAB_RX = BAR_X0 - 0.16           # right edge for horizontal single-row family labels
GUT_LEFT  = FAMLAB_RX - 2.15        # left edge of the data-axes (fits the widest family word)
YHEAD_BAND = 1.35                    # headroom for the axis header + dataset band
YFOOT = -0.55                        # axis bottom hugs the last cell row (ticks hang below)


def _build_block(L, axes):
    sub = L[L.axis.isin(axes)]
    methods = [m for _, ms in FAM_ROWS for m in ms if m in set(sub.method)]
    cols, axis_per_col = [], []
    for ax in axes:
        for c in dict.fromkeys(sub[sub.axis == ax].col):
            cols.append(c)
            axis_per_col.append(ax)
    nM, nC = len(methods), len(cols)
    S = np.full((nM, nC), np.nan)
    ACT = np.full((nM, nC), "", dtype=object)
    for i, m in enumerate(methods):
        for j, c in enumerate(cols):
            s = sub[(sub.method == m) & (sub.col == c)]
            if len(s):
                S[i, j] = s.score.mean()
                acts = set(s.action)
                ACT[i, j] = "floor" if "floor" in acts else ("adapted" if "adapted" in acts else "native")
    return methods, cols, axis_per_col, S, ACT


def _draw_block(ax, methods, cols, axis_per_col, S, ACT, norm, family_of, x_right=None):
    """Render one heatmap block. Cell-local coords: col j -> x=j, row i -> y=nM-1-i.

    x_right pins the right edge of the x-range to a SHARED value (panel a's column count) so a
    narrower block (panel b) keeps the SAME cell size and left-aligns under the wider one rather than
    stretching its cells to fill the axes width."""
    nM, nC = len(methods), len(cols)
    for i in range(nM):
        for j in range(nC):
            x, y = j, nM - 1 - i
            if np.isnan(S[i, j]):
                ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fc=NA_FC,
                                           ec=GRID, lw=1.1, zorder=1))
                continue
            val = S[i, j]
            fill = CMAP(norm(val))
            dark = _lum(fill) < LUM_T
            act = ACT[i, j]
            # SHARED simple baseline -> a sparse diagonal hatch over the whole cell (a texture, not a
            # corner glyph): unmistakably different from the adapted notch and readable without a
            # decode legend. The hatch ink tracks the fill luminance so it stays visible on dark or
            # light cells.
            hatch = "////" if act == "floor" else None
            hatch_ec = "white" if dark else GREY_MID
            r = plt.Rectangle((x - .5, y - .5), 1, 1, fc=fill, ec=GRID, lw=1.1, zorder=1,
                              hatch=hatch)
            r.set_edgecolor(GRID)            # cell seam stays white; hatch colour set separately
            ax.add_patch(r)
            if hatch:                        # colour the hatch lines (matplotlib uses edgecolor)
                r.set_edgecolor(hatch_ec)
                r.set_linewidth(0.0)         # no heavy border; only the hatch texture shows
                ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fc="none",
                                           ec=GRID, lw=1.1, zorder=1.5))  # restore white seam on top
            txt_c = ("white" if dark else NEG_INK) if val < 0 else ("white" if dark else INK)
            ax.text(x, y, _fmt(val), ha="center", va="center",
                    fontsize=7.4, color=txt_c, zorder=4)
            # ADAPTED (re-fit) -> ONE clearly distinct mark: a single thin corner NOTCH in the
            # upper-right (two short strokes cutting the corner). Restrained, but large enough to
            # read at print size without a decode legend.
            if act == "adapted":
                # a single thin corner notch, drawn just INSIDE the upper-right corner so the
                # stroke is always cell-bounded — never a stray mark hanging off the matrix edge
                # into the right/top margin (the prior clip_on=False let edge-column notches escape).
                nk = 0.28
                ax.plot([x + .46 - nk, x + .46], [y + .46, y + .46 - nk],
                        color=ADAPT_C, lw=1.0, zorder=3, solid_capstyle="round",
                        clip_on=True)

    # ---- column tick labels: short lineage/dataset only ----
    ax.set_xticks(range(nC))
    ax.set_xticklabels([COL_DISPLAY.get(c, c) for c in cols],
                       rotation=40, ha="right", va="top", rotation_mode="anchor",
                       fontsize=7.6, color=INK)
    ax.tick_params(axis="x", pad=3, length=0)
    ax.set_yticks([])

    # ---- left gutter: [family accent bar | family label] | method names | cells ----
    for i, m in enumerate(methods):
        y = nM - 1 - i
        ax.text(NAME_X, y, METHOD_SHORT.get(m, m), ha="right", va="center",
                fontsize=7.6, color=INK, clip_on=False)
    # Contiguous family blocks -> ONE muted accent bar per group, living in its OWN dedicated gutter
    # lane (BAR_X0..BAR_X1, far to the left of the cells) with the family name labelled DIRECTLY
    # beside it. The bar never touches or underlaps a cell (the prior collision bug) and there is no
    # separate colour-decode legend — the reference grammar. Every group, including singletons, gets
    # a name beside its bar; the bar has a small vertical inset so adjacent groups read as distinct.
    yacc = nM - 1
    for fam, ms in FAM_ROWS:
        present = [m for m in ms if m in methods]
        if not present:
            continue
        n = len(present)
        y_hi = yacc + .5
        y_lo = yacc - n + .5
        ax.add_patch(plt.Rectangle((BAR_X0, y_lo + .14), BAR_X1 - BAR_X0,
                                   (y_hi - y_lo) - .28, fc=ACCENT, ec="none",
                                   clip_on=False, zorder=5))
        # rotate the label for multi-row groups (it spans the bar height); set it horizontal,
        # right-anchored just left of the bar, for single rows so a 1-cell-tall word never collides.
        ymid = (y_hi + y_lo) / 2
        if n >= 2:
            ax.text(FAMLAB_X, ymid, FAM_LABEL.get(fam, fam), ha="center", va="center",
                    rotation=90, fontsize=6.8, color=GREY_MID, clip_on=False, zorder=5)
        else:
            ax.text(FAMLAB_RX, ymid, FAM_LABEL.get(fam, fam), ha="right", va="center",
                    fontsize=6.6, color=GREY_MID, clip_on=False, zorder=5)
        yacc -= n

    # ---- axis-group headers with span rules ----
    rule_y = nM - .5 + 0.50
    head_y = rule_y + 0.12
    order = list(dict.fromkeys(axis_per_col))
    # narrow side-by-side axis groups (e.g. 'modality' over Frangieh and 'donor' over the donor cols,
    # 2 columns each) sit close enough that two centred bold words touch ("modalitydonor"). When two
    # narrow (<4-col) groups are adjacent, lean each label OUTWARD off the shared seam — the left one
    # right-anchored to its own right edge minus a hair, the right one left-anchored to its left edge —
    # so each word hugs its own block and a clear gap opens between them.
    is_narrow = [len([k for k, a in enumerate(axis_per_col) if a == axn]) < 4 for axn in order]
    acc = 0
    for gi, axn in enumerate(order):
        idx = [k for k, a in enumerate(axis_per_col) if a == axn]
        n = len(idx)
        x0, x1 = idx[0] - .5 + 0.06, idx[-1] + .5 - 0.06
        lw = 1.6 if n > 1 else 1.0
        ax.plot([x0, x1], [rule_y, rule_y], color=NAVY_DARK, lw=lw, clip_on=False, zorder=6)
        label = AXIS_SHORT.get(axn, AXIS_TITLE.get(axn, axn)) if n < 4 else AXIS_TITLE.get(axn, axn)
        head_fs = 9.0 if n >= 4 else 7.2
        # default: centred over the group's span.
        lab_x, lab_ha = (x0 + x1) / 2, "center"
        if n < 4:
            crowd_right = gi + 1 < len(order) and is_narrow[gi + 1]   # a narrow group follows
            crowd_left = gi - 1 >= 0 and is_narrow[gi - 1]            # a narrow group precedes
            if crowd_right and not crowd_left:                        # lean toward own left edge
                lab_x, lab_ha = x1 - 0.02, "right"
            elif crowd_left and not crowd_right:                      # lean toward own right edge
                lab_x, lab_ha = x0 + 0.02, "left"
        ax.text(lab_x, head_y, label, ha=lab_ha, va="bottom",
                fontsize=head_fs, weight="bold", color=NAVY_DARK, clip_on=False, zorder=6)
        acc += n
        if acc < nC:
            ax.plot([acc - .5, acc - .5], [-.5, nM - .5], color=SEP, lw=0.8, zorder=2)

    # ---- ONE calm slate dataset band directly above the cells, with inline dataset name ----
    band_lo, band_hi = nM - .5 + 0.06, nM - .5 + 0.40
    cl_order = [COL_CLUSTER.get(c, "") for c in cols]
    k = 0
    while k < nC:
        cl = cl_order[k]
        j = k
        while j < nC and cl_order[j] == cl:
            j += 1
        span_cols = j - k
        x0, x1 = k - .5 + 0.05, (j - 1) + .5 - 0.05
        ax.add_patch(plt.Rectangle((x0, band_lo), x1 - x0, band_hi - band_lo,
                                   fc=BAND_GREY, ec="none", zorder=5, clip_on=False))
        # multi-column bands carry the inline dataset name; single-column bands stay an unlabelled
        # slate tick — the column tick below ALREADY spells out the dataset (Kang / Soskic /
        # compound), so a tag here would only clip and repeat. This keeps every band tag fully
        # inside its own band (no overflow) while the column map stays self-documenting.
        if span_cols >= 2:
            ax.text((x0 + x1) / 2, (band_lo + band_hi) / 2, CLUSTER_TAG.get(cl, cl),
                    ha="center", va="center", fontsize=7.0, color="white",
                    fontweight="bold", zorder=6, clip_on=False)
        k = j

    ax.set_xlim(GUT_LEFT, (x_right if x_right is not None else nC) - .5)
    ax.set_ylim(YFOOT, nM - .5 + YHEAD_BAND)
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    ax.set_facecolor("none")
    for sp in ax.spines.values():
        sp.set_visible(False)
    return rule_y


def main():
    set_pub_style()
    L = long_table()
    vmin = float(np.nanmin(L.score.values))
    vmax = float(np.nanmax(L.score.values))
    vmin = min(vmin, -1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    family_of = {m: f for f, ms in FAM_ROWS for m in ms}
    mA, cA, axA, SA, ACTA = _build_block(L, PANEL_A_AXES)
    mB, cB, axB, SB, ACTB = _build_block(L, PANEL_B_AXES)

    nMA, nCA = len(mA), len(cA)
    nMB, nCB = len(mB), len(cB)
    # STACKED layout: panels a (top) and b (bottom) share ONE cell size and ONE left gutter, so cell
    # SIZE is constant and the 5-column panel b is never dwarfed by the 16-column panel a (the side-
    # by-side imbalance the prior layout had). The shared x-span is the WIDER block (panel a), so both
    # axes occupy the same horizontal footprint; panel b simply uses fewer of those columns.
    spanX = (max(nCA, nCB) - .5) - GUT_LEFT        # shared horizontal span (panel a is wider)
    spanYA = (nMA - .5 + YHEAD_BAND) - YFOOT
    spanYB = (nMB - .5 + YHEAD_BAND) - YFOOT
    cell = 0.245                                   # inch per data unit (square cells)

    # generous fixed margins (inches) — whitespace is a first-class element here.
    m_left = 0.52                                  # left y-title column
    m_right = 0.30                                 # right whitespace
    gap_in = 0.64                                  # airy gap BETWEEN the two stacked panels (a/b)
    m_top = 0.66                                   # title + panel-a header/band
    # bottom band budget (inches): panel b's rotated tick labels hang BELOW its last cell row; the
    # longest ('McCutcheon', rotated 40 deg) reaches ~0.52in down. Then a clear airy gap, then the
    # single horizontal legend strip + its caption, then edge whitespace.
    tick_hang_in = 0.52                            # how far b's longest rotated tick reaches below row 0
    legend_band_in = 0.60                          # the one horizontal legend strip + its caption
    m_bot = tick_hang_in + 0.34 + legend_band_in   # ticks + airy gap + legend + edge whitespace

    body_h = cell * (spanYA + spanYB) + gap_in     # both panels + the inter-panel gap
    fig_w = m_left + cell * spanX + m_right
    fig_h = m_top + body_h + m_bot
    fig = plt.figure(figsize=(fig_w, fig_h))

    cfy = cell / fig_h
    cfx = cell / fig_w
    hA, hB = cfy * spanYA, cfy * spanYB
    wfull = cfx * spanX                            # shared width fraction for both panels

    top_frac = 1.0 - m_top / fig_h
    x0 = m_left / fig_w                            # shared left edge for both panels
    yA = top_frac - hA                             # panel a hangs from the top
    yB = yA - (gap_in / fig_h) - hB                # panel b sits below a, across the airy gap

    axA_ = fig.add_axes([x0, yA, wfull, hA])
    axB_ = fig.add_axes([x0, yB, wfull, hB])
    axA_.set_anchor("NW"); axB_.set_anchor("NW")
    axA_.set_clip_on(False); axB_.set_clip_on(False)

    nC_shared = max(nCA, nCB)
    _draw_block(axA_, mA, cA, axA, SA, ACTA, norm, family_of, x_right=nC_shared)
    _draw_block(axB_, mB, cB, axB, SB, ACTB, norm, family_of, x_right=nC_shared)

    bbA = axA_.get_position()
    bbB = axB_.get_position()
    bot = bbB.y0                                    # bottom of the lower panel's axes
    # data-cell left/right edges (fig fraction) of panel a, for centring titles over the matrix only.
    cell_l = bbA.x0 + (0 - .5 - GUT_LEFT) / spanX * bbA.width
    cell_r = bbA.x1
    cx = (cell_l + cell_r) / 2

    # ---- panel letters: bold, at the true top-left corner of each panel's axes ----
    fig.text(bbA.x0 - 0.004, top_frac + 0.014, "a", fontsize=12.0, fontweight="bold",
             va="bottom", ha="left", color=INK)
    fig.text(bbB.x0 - 0.004, bbB.y1 + 0.012, "b", fontsize=12.0, fontweight="bold",
             va="bottom", ha="left", color=INK)

    # ---- figure title, centred over the data matrix ----
    fig.text(cx, 1.0 - 0.012, r"Method $\times$ split performance landscape",
             ha="center", va="top", fontsize=11.0, fontweight="bold", color=NAVY_DARK)

    # ---- shared method/split axis titles (regular weight, grey) ----
    fig.text(0.008, (bbB.y0 + bbA.y1) / 2, "method (grouped by model family)",
             rotation=90, ha="left", va="center", fontsize=8.4, color=GREY_MID)

    # =================== ONE tidy bottom legend strip ===================
    # A SINGLE calm horizontal legend in a reserved band at the figure foot, left-aligned with the
    # data matrix: the diverging colour scale on the LEFT (terracotta note folded UNDER it as a
    # one-line caption), then the three cell-status keys inline to its RIGHT — all on ONE baseline.
    # No tall colour-decode card; family names are already labelled directly in the gutter, so this is
    # the figure's only legend furniture. The band is anchored to the figure foot so it can never
    # collide with panel b's rotated tick labels.
    band_mid = (0.16 + legend_band_in * 0.5) / fig_h   # vertical centre of the reserved legend band
    cb_x0 = cell_l
    cb_w_frac = 0.215
    cb_h_frac = 0.013
    cb_y0 = band_mid - cb_h_frac / 2
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([vmin, vmax])
    cax = fig.add_axes([cb_x0, cb_y0, cb_w_frac, cb_h_frac])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([vmin, 0.0, vmax])
    cb.set_ticklabels([_fmt(vmin), _fmt(0.0), _fmt(vmax)])
    cb.ax.tick_params(labelsize=7.2, length=0, pad=2)
    cb.outline.set_visible(False)
    _tls = cb.ax.get_xticklabels()
    if len(_tls) >= 3:
        _tls[0].set_ha("left"); _tls[-1].set_ha("right")
    fig.text(cb_x0, cb_y0 + cb_h_frac + 0.010, r"cell fill = Pearson-$\Delta$",
             fontsize=8.0, ha="left", va="bottom", color=INK, fontweight="bold")
    fig.text(cb_x0, cb_y0 - 0.028, "coral = worse than control",
             fontsize=7.0, ha="left", va="top", color=CLAY_DARK, style="italic")

    # ---- the three status keys, inline to the RIGHT of the colorbar, all on ONE baseline ----
    sww = 0.013                                    # swatch width (fig fraction)
    swh = sww * fig_w / fig_h                       # square in display units
    key_yc = band_mid                              # one shared vertical centre for every swatch
    lab_dx = sww + 0.009

    def _key(xt, draw, text):
        a = fig.add_axes([xt, key_yc - swh / 2, sww, swh])
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
        draw(a)
        fig.text(xt + lab_dx, key_yc, text, fontsize=7.2, ha="left", va="center", color=GREY_MID)

    def _d_adapt(a):
        a.add_patch(plt.Rectangle((0, 0), 1, 1, fc=CMAP(norm(0.45)), ec=GRID, lw=0.8))
        a.plot([1 - 0.40, 1], [1, 1 - 0.40], color="white", lw=1.2, solid_capstyle="round")

    def _d_floor(a):
        a.add_patch(plt.Rectangle((0, 0), 1, 1, fc=CMAP(norm(0.45)), ec="white", lw=0.8,
                                  hatch="////"))
        a.add_patch(plt.Rectangle((0, 0), 1, 1, fc="none", ec=GRID, lw=0.8))

    def _d_na(a):
        a.add_patch(plt.Rectangle((0, 0), 1, 1, fc=NA_FC, ec=LEGEND_EC, lw=0.8))

    # left x of each key, marching rightward with EVEN gaps. Advances = (swatch + label width) +
    # one constant inter-key gap, so the three keys never crowd or overlap (measured label widths in
    # fig-fraction at fs 7.2: adapted .117, shared-baseline .182, not-applicable .110).
    GAP = 0.028                                    # one constant gap between keys
    kx = cb_x0 + cb_w_frac + 0.052
    _key(kx, _d_adapt, "adapted (re-fit)")
    kx += lab_dx + 0.117 + GAP
    _key(kx, _d_floor, "shared simple baseline")
    kx += lab_dx + 0.182 + GAP
    _key(kx, _d_na, "not applicable")

    out = RESULTS / "_paper" / "figure_ranking.png"
    # generous, even outer border on every side (the right-crowding complaint); with the adapted
    # notch now cell-bounded there is no escaping ink, so a larger pad gives clean whitespace at the
    # right table border instead of the rightmost column sitting flush to the frame.
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.22, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.22, facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
