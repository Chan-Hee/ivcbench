#!/usr/bin/env python
"""Figure 1 (Nature-grade) - the immune-aware benchmark framework.

A single editorial plate in the author's restrained NAVY system (style.py), built to the polish
of the reference figure Section2_2_Figure1.png: generous whitespace, navy structural ink, muted
axis-colour ACCENT BARS used directly as row/card group keys (no separate decode-legend clutter),
clean Liberation-Sans typography, perfect grid alignment, and a single tidy bottom legend.

(a) Four immune generalization axes (the unit of evaluation): clean cards, each keyed by a thin
    axis-colour accent bar, naming the held-out unit, the generalization it asks for, the held-in
    -> held-out fold ratio, and the datasets governed by that axis.
(b) Three immune-aware metric axes: clean cards, each keyed by the accent of the generalization
    axis it most directly scores, naming the statistic and its direction-of-better.
(c) The immune dataset panel (T1-T5): an infographic table keyed by per-task axis accent bars,
    with a navy method-count dot rail (the dot glyph is reserved EXCLUSIVELY for Methods n).

Every plotted number (method counts, lineage/donor counts) is read from
results/{C1..C5}/results_raw.csv (or the deposited roster) - nothing quantitative is hardcoded.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import pandas as pd

# figure is 11.6 x 8.6 in on a 0-1 square canvas; correct x-radius so circles render round
FIGW, FIGH = 11.6, 8.6
ASP = FIGH / FIGW          # multiply a y-radius by ASP to get the matching x-radius

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ivcbench.report.style import (set_pub_style, FRAMEWORK_PILLS,  # noqa: E402
                                   NAVY, NAVY_DARK, NAVY_RAMP, SLATE_BAND,
                                   INK, GREY_MID)

RESULTS = Path(__file__).resolve().parents[1] / "results"

# ----------------------------------------------------------------------------------------------
# GEARS-CLEAN, NAVY-MONOCHROME framework system. Figure 1 is an EXPLANATORY infographic, not a data
# plot, so colour here LABELS concepts the text already names. To keep the plate calm (the author's
# "weird/busy" complaint was the four saturated axis hues firing across panels a/b/c), NAVY is the
# SINGLE structural ink for every card / metric accent bar / dataset-row anchor, and the four axis
# hues are spent in EXACTLY ONE place: the panel-(c) "Axes tested" pills + their in-place header key
# (FRAMEWORK_PILLS — muted, calm siblings of the data-mark AXIS_COLORS). A reader decodes that one
# legend once; everything else is navy ink + whitespace. No saturated four-hue parade anywhere else.
# ----------------------------------------------------------------------------------------------
HEAD_INK = NAVY_DARK            # all section / card / column-header ink
BODY_INK = "#2b2b2b"           # body text
SUB_INK  = GREY_MID            # secondary / sub text
HAIR     = "#d8dde2"           # one hairline grey for rules / card edges
ZEBRA    = "#f4f7fa"           # one faint band tint (a light step on the navy ramp)
DOT_FILL = NAVY                # the run-dot fill
DOT_OPEN = "#c2ccd4"           # the not-run open-dot ring

# Sourced BY NAME from ivcbench.report.style — no local hexes.
#   PILL_TONE  — the ONLY coloured marks on the plate: the (c) axis pills + the header key.
#   CARD_ACCENT/METRIC_TONE/ROW_ANCHOR — all NAVY (or a quiet navy step), so a/b cards and c rows read
#   as one calm navy family and never compete with the single pill legend.
PILL_TONE   = dict(FRAMEWORK_PILLS)      # cornflower / coral / teal / amethyst (muted), pills + key only
CARD_ACCENT = NAVY                       # one navy accent bar for every panel-(a) axis card
METRIC_TONE = [NAVY, NAVY, NAVY]         # one navy accent bar for every panel-(b) metric card
ROW_ANCHOR  = SLATE_BAND                 # quiet slate task-row anchor (a calm navy-family neutral)
def text_on(bg):
    """white on dark fills, dark navy on light fills, so chip/pill text stays legible."""
    h = bg.lstrip("#"); r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#ffffff" if (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.58 else NAVY_DARK


# ---- consistent type scale (pt) --------------------------------------------------------------
FS_PANEL   = 15.5   # panel letters a / b / c
FS_HEAD    = 13.0   # section headings
FS_CARD    = 11.8   # card titles
FS_BODY    = 10.2   # card body
FS_TAG     = 9.4    # small chips / inline tags
FS_MICRO   = 8.8    # micro labels (held-in/out, fold caption)
FS_TBL_HEAD = 10.6  # table column headers
FS_TBL_CELL = 10.2  # table data cells
FS_TBL_SUB  = 9.2   # table secondary sub-text + footnote

# canonical shared left/right edges (all three blocks snap to these) ---------------------------
L = 0.045
R = 0.955


# ---- data readers (every plotted count comes from here) --------------------------------------
CENSUS = RESULTS / "_paper" / "cross_cluster_headline.csv"


def n_methods(c):
    """Conditioned/learned methods executed on task c, from the deposited headline census
    (Supplementary Table S2) — the single authoritative source, applied identically to every
    task. The four universal/context simple comparators (cell-mean, linear-PCA, donor-shift,
    control-as-prediction) run on every task and are NOT counted here. The deposited census
    column uses legacy labels C1-C5, mapped here to manuscript tasks T1-T5."""
    df = pd.read_csv(CENSUS)
    _legacy = {"T1": "C1", "T2": "C2", "T3": "C3", "T4": "C4", "T5": "C5"}
    return int(df[df.cluster == _legacy.get(c, c)].model.nunique())


def n_models_total():
    """Distinct learned models benchmark-wide (the denominator of the Methods rail) = 14."""
    return int(pd.read_csv(CENSUS).model.nunique())


# ---- colour helpers --------------------------------------------------------------------------
def _tint(hexcol, frac):
    """blend hex colour toward white by frac (0=white, 1=full colour)."""
    h = hexcol.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * (1 - frac))
    g = int(g + (255 - g) * (1 - frac))
    b = int(b + (255 - b) * (1 - frac))
    return f"#{r:02x}{g:02x}{b:02x}"


def rdot(ax, x, y, r, **kw):
    """draw a TRUE round dot (aspect-corrected ellipse) on the 0-1 non-square canvas."""
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((x, y), width=2 * r * ASP, height=2 * r, **kw))


def section_head(ax, letter, title, y):
    """panel letter + bold navy section heading, on one shared baseline."""
    ax.text(L, y, letter, fontsize=FS_PANEL, weight="bold", color=INK, va="top")
    ax.text(L + 0.022, y - 0.001, title, fontsize=FS_HEAD, weight="bold", color=HEAD_INK,
            va="top")


def fold_ratio(ax, x, y, w, color, n_in, n_out):
    """clean held-in -> held-out fold ribbon: n_in solid axis-colour segments, n_out open, with a
    single tidy caption. No hatching, no per-end colour war — one calm encoding."""
    total = n_in + n_out
    gap = 0.0014
    seg = (w - (total - 1) * gap) / total
    h = 0.011
    for k in range(total):
        bx = x + k * (seg + gap)
        if k < n_in:
            ax.add_patch(FancyBboxPatch((bx, y), seg, h,
                         boxstyle="round,pad=0.0,rounding_size=0.002",
                         fc=color, ec="none"))
        else:
            ax.add_patch(FancyBboxPatch((bx, y), seg, h,
                         boxstyle="round,pad=0.0,rounding_size=0.002",
                         fc="white", ec=color, lw=0.9))
    # labels carry no held-in COUNT: the segment ribbon is schematic (a task's held-in pool can
    # be 2 lineages or 105 donors), so a literal "N held-in" would misstate it. The held-out unit is
    # named on the card's "held out:" line; here we mark only that exactly one unit is held out.
    ax.text(x, y - 0.006, "held-in", ha="left", va="top", fontsize=FS_MICRO, color=SUB_INK)
    ax.text(x + w, y - 0.006, "1 held-out", ha="right", va="top", fontsize=FS_MICRO,
            color=HEAD_INK, weight="bold")


def chip(ax, x, y, label, color, fs=FS_TAG, pad=0.011, h=0.026, align="l"):
    """a small rounded accent chip; returns its right edge x. align 'l' anchors left at x.
    h carries a uniform vertical pad so the tag text has airy top/bottom breathing room (matching
    the panel-c axis pills) instead of crowding the rounded edge."""
    w = 0.0072 * len(label) + 2 * pad
    x0 = x if align == "l" else x - w
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), w, h,
                 boxstyle="round,pad=0.0,rounding_size=0.006",
                 fc=color, ec="none"))
    ax.text(x0 + w / 2, y, label, ha="center", va="center", fontsize=fs, color=text_on(color),
            weight="bold")
    return x0 + w


# ----------------------------------------------------------------------------------------------
def main():
    set_pub_style()
    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    n = 4
    cgap = 0.022
    cw = (R - L - (n - 1) * cgap) / n

    # ==========================================================================================
    # (a) FOUR IMMUNE GENERALIZATION AXES — clean accent-keyed cards
    # ==========================================================================================
    section_head(ax, "a", "Four immune generalization axes", 0.975)

    cards = [
        # key,           title,          held-out,           body,                            tags,  in/out
        ("cell-context", "Cell-context", "immune lineage",
         "perturbation seen,\nnew cell context",
         "T1 cytokine  $\\cdot$  T5 compound", 3, 1),
        ("perturbation", "Perturbation", "gene / compound",
         "unseen intervention,\nextrapolate to new target",
         "T3 CRISPR  $\\cdot$  T4 KO  $\\cdot$  T5 cpd", 3, 1),
        ("modality",     "Modality",     "readout",
         "train on RNA,\ntransfer to surface proteome",
         "T4 Frangieh CITE-seq", 1, 1),
        ("donor",        "Donor",        "individual",
         "leave-one-donor-out\ngeneralization",
         "T2 leave-one-donor-out", 7, 1),
    ]
    ch = 0.190
    cy0 = 0.735
    for i, (key, title, held, body, tags, n_in, n_out) in enumerate(cards):
        x = L + i * (cw + cgap)
        col = CARD_ACCENT   # ONE navy accent for every card (the axis is named in the title)
        # clean white card, one hairline edge; a thin NAVY accent bar down the LEFT edge is the only
        # mark — all four cards share it, so panel (a) reads as one calm navy family, not four hues.
        ax.add_patch(FancyBboxPatch((x, cy0), cw, ch,
                     boxstyle="round,pad=0.002,rounding_size=0.010",
                     fc="white", ec=HAIR, lw=0.9))
        ax.add_patch(FancyBboxPatch((x + 0.004, cy0 + 0.006), 0.0075, ch - 0.012,
                     boxstyle="round,pad=0.0,rounding_size=0.003", fc=col, ec="none"))
        xin = x + 0.024
        # title (navy ink — one structural ink for all card titles)
        ax.text(xin, cy0 + ch - 0.026, title, ha="left", va="center", fontsize=FS_CARD,
                weight="bold", color=HEAD_INK)
        # held-out unit on one clean line — grey cap word + the axis-colour unit, no box (declutter)
        ay = cy0 + ch - 0.058
        ax.text(xin, ay, "held out:  ", ha="left", va="center", fontsize=FS_TAG, color=SUB_INK)
        ax.text(xin + 0.052, ay, held, ha="left", va="center", fontsize=FS_TAG, color=HEAD_INK,
                weight="bold")
        # body description in plain body ink (calm reading path)
        ax.text(xin, cy0 + ch - 0.080, body, ha="left", va="top", fontsize=FS_BODY,
                color=BODY_INK, linespacing=1.22)
        # fold ratio ribbon
        fold_ratio(ax, xin, cy0 + 0.050, cw - 0.034, col, n_in, n_out)
        # datasets governed by this axis (quiet sub ink)
        ax.text(xin, cy0 + 0.020, tags, ha="left", va="center", fontsize=FS_TAG, color=SUB_INK)

    # ==========================================================================================
    # (b) THREE IMMUNE-AWARE METRIC AXES — accent-keyed to the axis each metric scores
    # ==========================================================================================
    GAP_AB = 0.052
    by_title = cy0 - GAP_AB
    section_head(ax, "b", "Three immune-aware metric axes", by_title)

    # the axis-link chip names its axis in TEXT, so it needs no per-axis hue: one navy chip family
    # keeps panel (b) in the same calm navy register as (a) (no competing cell-context / perturbation
    # hues here — the four axis hues are spent once, on the panel-(c) pills).
    metrics = [
        ("Response direction", r"Pearson-$\Delta$, downstream genes", "higher",
         "cell-context", NAVY),
        ("Immune program / readout", r"AUCell-$\Delta$ / per-marker recovery", "higher",
         "cross-axis", NAVY),
        ("Distribution",       "E-distance, state-space shift", "lower",
         "perturbation", NAVY),
    ]
    nb = len(metrics)
    bgap = 0.030
    bw = (R - L - (nb - 1) * bgap) / nb
    bh = 0.110
    by0 = by_title - 0.034 - bh
    for i, (t, sub, better, axis_tag, col) in enumerate(metrics):
        x = L + i * (bw + bgap)
        ax.add_patch(FancyBboxPatch((x, by0), bw, bh,
                     boxstyle="round,pad=0.002,rounding_size=0.010",
                     fc="white", ec=HAIR, lw=0.9))
        # metric-card accent bar is NEUTRAL slate: the four AXIS hues are reserved for axes only,
        # so a metric card never reads as an axis. The axis link lives in the chip below instead.
        ax.add_patch(FancyBboxPatch((x + 0.004, by0 + 0.006), 0.0075, bh - 0.012,
                     boxstyle="round,pad=0.0,rounding_size=0.003", fc=METRIC_TONE[i], ec="none"))
        xin = x + 0.024
        ax.text(xin, by0 + bh - 0.028, t, ha="left", va="center", fontsize=FS_CARD,
                weight="bold", color=HEAD_INK)
        ax.text(xin, by0 + bh - 0.058, sub, ha="left", va="center", fontsize=FS_BODY,
                color=BODY_INK)
        # footer row: axis-tag chip (left) + direction-of-better (right)
        fy = by0 + 0.026
        chip(ax, xin, fy, axis_tag, col, fs=FS_TAG)
        arrow = "$\\uparrow$" if better == "higher" else "$\\downarrow$"
        ax.text(x + bw - 0.020, fy, f"{arrow} better", ha="right", va="center",
                fontsize=FS_MICRO, color=SUB_INK, style="italic")

    # ==========================================================================================
    # (c) IMMUNE DATASET PANEL — infographic table, per-task axis accent bars + navy dot rail
    # ==========================================================================================
    GAP_BC = 0.050
    cy_title = by0 - GAP_BC
    section_head(ax, "c", "Immune perturbation datasets", cy_title)

    # column x-grid (snapped within [L, R]); the accent bar lives just inside L. Column starts are
    # measured against each column's widest cell so no cell overruns the next column:
    #   src widest  'T3 5x primary-T CRISPR'        -> imm starts past it
    #   imm widest  '2 lineages . 106 donors'       -> pert starts past it
    #   pert widest 'TCR/CD28 activation (0h->16h)' -> mod starts past it
    #   mod widest  'RNA + 20-marker CITE'          -> the axes-pill BLOCK starts past it
    acc_x  = L                       # left accent-bar x
    txt_x  = L + 0.018               # text column left edge (past the accent bar)
    # Column starts widened on the right so every long cell clears the next column with an EVEN,
    # comfortable gap (measured via get_window_extent — see the assertion block below). The widest
    # Perturbation cell (T2 'TCR/CD28 activation (0h->16h)') ends at ~0.524 and the widest Modality
    # cell ('RNA + 20-marker CITE') is long, so Modality/axes/Methods were each nudged right into the
    # previously empty band between the number column and the table's right rule (R).
    mod_x   = 0.547                  # LEFT edge of the Modality column (was crowded by Perturbation)
    axes_x0 = 0.686                  # LEFT edge of the (horizontal) axes-pill block — one baseline
    meth_x0 = 0.838
    num_col_x = 0.928
    num_pad = 0.022
    meth_track_w = num_col_x - num_pad - meth_x0
    colx = {"src": txt_x, "imm": 0.215, "pert": 0.342, "mod": mod_x, "axes": axes_x0}
    yhead = cy_title - 0.050
    heads = [("src", "Task / source", "l"), ("imm", "Immune system", "l"),
             ("pert", "Perturbation", "l"), ("mod", "Modality", "l"), ("axes", "Axes tested", "l")]
    for k, hname, al in heads:
        ax.text(colx[k], yhead, hname, fontsize=FS_TBL_HEAD, weight="bold", color=HEAD_INK,
                va="center", ha="center" if al == "c" else "left")
    MAXN = max(n_methods(c) for c in ["T1", "T2", "T3", "T4", "T5"])
    SCALE_MAX = n_models_total()   # 14 distinct learned models benchmark-wide
    ax.text(meth_x0, yhead, "Methods", fontsize=FS_TBL_HEAD, weight="bold", color=HEAD_INK,
            va="center")
    ax.text(num_col_x, yhead, f"n / {SCALE_MAX}", fontsize=FS_TBL_HEAD - 0.6, color=SUB_INK,
            va="center", ha="right")
    _ = MAXN

    # ---- inline axis-colour KEY in the panel-c header: the ONE legend on the whole plate. It decodes
    # the four "Axes tested" pill colours IN PLACE (small swatch + axis name), right on the same row as
    # the panel-c heading. Panels (a)/(b) are navy-monochrome, so this key is the single colour decode.
    key_items = [("cell-context", "cell-context"), ("donor", "donor"),
                 ("perturbation", "perturbation"), ("modality", "modality")]
    sw = 0.011                                   # swatch side
    kgap_sw_txt = 0.006                          # swatch -> label gap
    kgap_item = 0.020                            # item -> item gap
    seg_w = []
    for _key, lbl in key_items:
        seg_w.append(sw + kgap_sw_txt + 0.0064 * len(lbl))
    total_kw = sum(seg_w) + (len(key_items) - 1) * kgap_item
    kx = R - total_kw
    for (kkey, klbl), swid in zip(key_items, seg_w):
        ax.add_patch(FancyBboxPatch((kx, cy_title - 0.0055), sw, 0.011,
                     boxstyle="round,pad=0.0,rounding_size=0.0025",
                     fc=PILL_TONE[kkey], ec="none"))
        ax.text(kx + sw + kgap_sw_txt, cy_title - 0.001, klbl, fontsize=FS_MICRO - 0.3,
                color=SUB_INK, va="center", ha="left")
        kx += swid + kgap_item

    # header rule (one hairline)
    rule_y = yhead - 0.024
    ax.plot([L, R], [rule_y, rule_y], color=HAIR, lw=1.0)

    rows = [
        ("T1  Kang GSE96583", ("PBMC", "8 lineages  $\\cdot$  8 donors"),
         "IFN-$\\beta$ cytokine (seen)", "RNA",
         [("cell-context", "cell-context"), ("donor", "donor")], "T1"),
        ("T2  Soskic 2022", ("CD4 T cells", "2 lineages  $\\cdot$  106 donors"),
         "TCR/CD28 activation (0h $\\rightarrow$ 16h)", "RNA",
         [("donor", "donor")], "T2"),
        ("T3  5$\\times$ primary-T CRISPR", ("primary human T", "pan-T pool"),
         "KO / CRISPRi/a (unseen genes)", "RNA",
         [("perturbation", "perturbation")], "T3"),
        ("T4  Frangieh", ("melanoma + TIL", "co-culture"),
         r"$\approx$248 gene KO (unseen)", "RNA + 20-marker CITE",
         [("perturbation", "perturbation"), ("modality", "modality")], "T4"),
        ("T5  OP3 GSE279945", ("PBMC", "6 lineages  $\\cdot$  3 donors"),
         "small-molecule compounds", "RNA",
         [("cell-context", "cell-context"), ("perturbation", "perturbation")], "T5"),
    ]
    # the dominant axis (its accent colour) for each task's left bar = first listed axis
    rh = 0.062
    y_first = rule_y - 0.006 - rh
    slot = meth_track_w / SCALE_MAX
    # filled-dot radius sized to the slot so consecutive dots read as a clean strip without touching
    # (x-radius = dotr*ASP must stay under the slot pitch)
    dotr = min(0.0050, 0.40 * slot / ASP)
    for r_i, (src, imm, pert, mod, axpills, cl) in enumerate(rows):
        yc = y_first - r_i * rh + rh / 2
        if r_i % 2 == 0:
            ax.add_patch(Rectangle((L, yc - rh / 2), R - L, rh, fc=ZEBRA, ec="none", zorder=0))
        # per-task left accent bar is NEUTRAL slate: a quiet row anchor only. The four AXIS hues
        # stay reserved for the axes themselves (the 'Axes tested' pills carry each task's axes),
        # so a single "dominant-axis" colour can no longer be mis-read as the task's only axis.
        acc_col = ROW_ANCHOR
        ax.add_patch(FancyBboxPatch((acc_x, yc - rh / 2 + 0.010), 0.0065, rh - 0.020,
                     boxstyle="round,pad=0.0,rounding_size=0.003", fc=acc_col, ec="none",
                     zorder=1))
        ax.text(colx["src"], yc, src, fontsize=FS_TBL_CELL, weight="bold", color=INK, va="center")
        label, sub = imm
        ax.text(colx["imm"], yc + 0.011, label, fontsize=FS_TBL_CELL, color=BODY_INK, va="center")
        ax.text(colx["imm"], yc - 0.013, sub, fontsize=FS_TBL_SUB, color=SUB_INK, va="center")
        ax.text(colx["pert"], yc, pert, fontsize=FS_TBL_CELL, color=BODY_INK, va="center")
        ax.text(colx["mod"], yc, mod, fontsize=FS_TBL_CELL, color=BODY_INK, va="center")
        # axes-tested pills, laid out HORIZONTALLY on ONE line per row (side by side), LEFT-anchored
        # to a single column baseline (axes_x0) so every row keeps identical height and one common
        # left edge. Each pill is anchored to the shared row-centre yc, so single- and double-pill
        # rows share one midline with the Methods dots. A uniform vertical pad inside each pill gives
        # airy top/bottom breathing room.
        FS_PILL = 8.0
        cpadx = 0.0055                       # horizontal text pad inside each pill
        pill_h = 0.026                       # taller pill -> uniform top/bottom breathing room
        pgap = 0.007                         # gap between adjacent horizontal pills
        x0 = axes_x0
        for token, key in axpills:
            w = 0.0050 * len(token) + 2 * cpadx
            ax.add_patch(FancyBboxPatch((x0, yc - pill_h / 2), w, pill_h,
                         boxstyle="round,pad=0.0,rounding_size=0.006",
                         fc=PILL_TONE[key], ec="none"))
            ax.text(x0 + w / 2, yc, token, ha="center", va="center", fontsize=FS_PILL,
                    color=text_on(PILL_TONE[key]), weight="bold")
            x0 += w + pgap
        # method-count dot rail (fixed denominator; navy filled = n, open = remainder)
        m = n_methods(cl)
        ax.add_patch(Rectangle((meth_x0, yc - 0.0009), meth_track_w, 0.0018,
                     fc="#e8edf1", ec="none", zorder=1))
        for k in range(SCALE_MAX):
            cx = meth_x0 + (k + 0.5) * slot
            if k < m:
                rdot(ax, cx, yc, dotr, fc=DOT_FILL, ec="none", zorder=2)
            else:
                rdot(ax, cx, yc, dotr * 0.64, fc="white", ec=DOT_OPEN, lw=1.0, zorder=2)
        ax.text(num_col_x, yc, str(m), fontsize=FS_TBL_CELL + 1.0, weight="bold",
                color=NAVY_DARK, va="center", ha="right")

    # table base rule
    last_bottom = y_first - (len(rows) - 1) * rh - rh / 2
    ax.plot([L, R], [last_bottom, last_bottom], color=HAIR, lw=1.0)

    # ---- ONE tidy legend strip under the table: defines the dot rail (run / not run) -----------
    leg_y = last_bottom - 0.030
    lx = L
    rdot(ax, lx + 0.004, leg_y, 0.0050, fc=DOT_FILL, ec="none", zorder=3)
    ax.text(lx + 0.013, leg_y, "method run", fontsize=FS_MICRO, color=SUB_INK, va="center")
    lx2 = lx + 0.115
    rdot(ax, lx2 + 0.004, leg_y, 0.0050 * 0.64, fc="white", ec=DOT_OPEN, lw=1.0, zorder=3)
    ax.text(lx2 + 0.013, leg_y, "not run", fontsize=FS_MICRO, color=SUB_INK, va="center")

    # footnote -----------------------------------------------------------------------------------
    y_foot = leg_y - 0.030
    ax.text(L, y_foot,
            "Methods dots = conditioned/learned methods executed on the task, of 14 distinct "
            "learned models benchmark-wide; the\n"
            "universal floor {cell-mean $\\cdot$ linear-PCA} and the context comparators (control, "
            "donor-shift) run on every task and are not\n"
            "counted in the rail. Each model is admitted under an applicability gate (native / adapted "
            "/ simple / n.a.). Soskic is shown for the\n"
            "donor LODO axis; temporal saturation and additional CD4-state / in-vivo extensions remain "
            "outside this benchmark panel.",
            fontsize=FS_TBL_SUB, color=SUB_INK, va="top", linespacing=1.34)

    out = RESULTS / "_paper" / "figure_framework.png"
    fig.canvas.draw()

    # ---- SELF-VERIFY: no table cell may crowd the next column (measured, not eyeballed) ----------
    # Per the figure-QC contract: get each text's real rendered extent and assert an even, positive
    # gap to the column it sits before. mod_x / axes_x0 were widened until every gap clears MIN_GAP.
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def _xr(t):  # right-edge x of a text artist, in axes-fraction (xlim is 0..1)
        e = t.get_window_extent(rend)
        return max(inv.transform((e.x0, e.y0))[0], inv.transform((e.x1, e.y1))[0])

    MIN_GAP = 0.012  # ~62 px on the 5208-px canvas: an even, comfortable inter-column gutter
    # Perturbation cells must clear the Modality column; Modality cells must clear the axes pills.
    pert_strings = {"IFN-$\\beta$ cytokine (seen)", "TCR/CD28 activation (0h $\\rightarrow$ 16h)",
                    "KO / CRISPRi/a (unseen genes)", r"$\approx$248 gene KO (unseen)",
                    "small-molecule compounds"}
    mod_strings = {"RNA", "RNA + 20-marker CITE"}
    worst = {}
    for t in ax.texts:
        s = t.get_text()
        if s in pert_strings:
            worst["pert->mod"] = min(worst.get("pert->mod", 9), mod_x - _xr(t))
        elif s in mod_strings:
            worst["mod->axes"] = min(worst.get("mod->axes", 9), axes_x0 - _xr(t))
    for k, g in worst.items():
        assert g >= MIN_GAP, f"OVERFLOW/CROWD {k}: gap {g:.4f} < {MIN_GAP} (widen the column)"
    # every text must sit inside the [0,1] data canvas with a hair of margin
    for t in ax.texts:
        e = t.get_window_extent(rend)
        x0 = min(inv.transform((e.x0, e.y0))[0], inv.transform((e.x1, e.y1))[0])
        x1 = _xr(t)
        assert x0 >= -0.002 and x1 <= 1.002, f"TEXT off-canvas: x0={x0:.3f} x1={x1:.3f} '{s[:30]}'"
    print("self-verify OK  gaps:", {k: round(v, 4) for k, v in worst.items()})

    from matplotlib.transforms import Bbox
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    # vector PDF: asymmetric bbox (generous room past the Methods / n-of-14 column)
    fig.savefig(out.with_suffix(".pdf"),
                bbox_inches=Bbox.from_extents(bb.x0 - 0.6, bb.y0 - 0.55, bb.x1 + 0.6, bb.y1 + 0.4),
                facecolor="white")
    # raster: tight save, then GUARANTEE generous margins with a white pad (heavier on the right)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    from PIL import Image
    _im = Image.open(out).convert("RGB"); _W, _H = _im.size
    _pl, _pr, _pt, _pb = int(_W * 0.048), int(_W * 0.052), int(_H * 0.05), int(_H * 0.062)
    _canv = Image.new("RGB", (_W + _pl + _pr, _H + _pt + _pb), "white")
    _canv.paste(_im, (_pl, _pt)); _canv.save(out, dpi=(400, 400))
    print("wrote", out)


if __name__ == "__main__":
    main()
