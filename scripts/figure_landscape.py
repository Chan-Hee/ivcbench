#!/usr/bin/env python
r"""Figure 3 (v2 MAIN LANDSCAPE) — the 35-cell (family x model) x task UNIVERSAL-FLOOR landscape.

This is the §2-§3 headline plate. Rows = the (family, model) entrants grouped by model family;
columns = the benchmark tasks grouped by GENERALIZATION AXIS into two dense sub-panels so the
axis dissociation reads at a glance:

  panel a — CONTEXT / DONOR axis  (a seen perturbation moved into an unseen cell type or donor):
            C1 cell-context (LOCT), C2 donor (LODO), C5 cell-context (LOCT).
  panel b — UNSEEN-PERTURBATION axis  (a held-out gene / KO / compound):
            C3 unseen-perturbation (LO-gene), C4 unseen-KO (modality), C5 unseen-compound.

The single quantity that drives every cell is the HEADLINE contrast of the benchmark — the
response-direction Pearson-Δ of the entrant MINUS the universal simple floor (mean of the two
floor members {cell-mean shift, linear-PCA shift}), i.e. `delta_vs_floor_mean` in
cross_cluster_headline.csv. The diverging colour scale is centred at ZERO, so:

  * the ZERO crossing IS the universal floor line — rust below (entrant under the floor), blue
    above (entrant over the floor-mean). One scale, directly comparable across every cell.
  * printed NUMBER = the same Δ-vs-floor (signed, unicode minus).
  * a GOLD ring highlights the only cells that beat BOTH floor members (the §2 sparsity headline:
    2 of 35 — CellOT @ C2-donor, FP-ridge @ C5 cell-context). scPRAM @ C2-donor is now a SCORED
    cell (no longer "not run" grey): in THIS plate's frame the colour = Δ-vs-floor-MEAN = +0.011
    (a pale tile, marginally over the floor-mean line) and it carries NO gold ring because it FAILS
    the cell-mean floor member (Δ-vs-cell-mean = −0.101) — that under-cell-mean read is the §2/§4
    narrative number; the gold-ring absence is exactly how this plate encodes "under floor".
  * flat light-grey cell = that (family, model) was not run / not applicable on that task.

The floor line is also drawn explicitly: an in-cell "= floor" reference and a colorbar 0-tick.
Every plotted value is read straight from results/_paper/cross_cluster_headline.csv — no hardcoded
numbers, no fabrication. Mirrors scripts/figure_ranking.py / figure_framework.py grammar.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ivcbench.report.style import (set_pub_style, FAM_COLORS, AXIS_COLORS, LEGEND_EC,  # noqa: E402
                                   NAVY, NAVY_DARK, SLATE_BAND, CLAY_DARK, GREY_MID,
                                   floor_diverging_cmap)

ROOT = Path(__file__).resolve().parents[1]
HEADLINE = ROOT / "results" / "_paper" / "cross_cluster_headline.csv"

# ---- family display order (rows): conditioned families grouped; the universal floor is NOT a row
# (it is the zero line). The two CSV family tokens 'OT'/'Deterministic shift' map onto the shared
# style.FAM_COLORS spine namespace ('opt-transport'/'shift') so the spine hue == the rest of the plate.
FAM_ROWS = [
    ("Latent", ["scGen", "CPA"]),
    ("Graph", ["GEARS", "AttentionPert"]),
    ("Foundation", ["scGPT", "scFoundation"]),
    ("Hybrid", ["STATE", "PertAdapt"]),
    ("Chemistry", ["FP-ridge", "chemCPA"]),
    ("Deterministic shift", ["linear-shift-KOemb"]),
    ("OT", ["CINEMA-OT", "CellOT", "scPRAM"]),
]
# CSV family token -> the shared style.FAM_COLORS spine key (one hue per family across the plate).
FAM_SPINE_KEY = {
    "Latent": "latent", "Graph": "graph", "Foundation": "foundation", "Hybrid": "hybrid",
    "Chemistry": "chemistry", "Deterministic shift": "shift", "OT": "opt-transport",
}
# family group name shown DIRECTLY in the gutter (the direct row-group label that replaces the
# old separate colour-decode legend). Two short lines where the name is long, so it never overruns.
FAM_LABEL = {
    "Latent": "Latent", "Graph": "Graph", "Foundation": "Foundation", "Hybrid": "Hybrid",
    "Chemistry": "Chemistry", "Deterministic shift": "Determ.\nshift", "OT": "OT",
}
# gutter-safe short names for the longest model tokens.
MODEL_SHORT = {
    "linear-shift-KOemb": "lin-shift-KO",
    "AttentionPert": "AttnPert",
    "scFoundation": "scFound.",
    "CINEMA-OT": "CINEMA-OT",
}

# ---- task columns: ordered within each axis panel. Each task = a (cluster, split) cell of the CSV.
# CONTEXT / DONOR axis (panel a) then UNSEEN-PERTURBATION axis (panel b).
PANEL_A_TASKS = [
    ("C1", "cell-context (LOCT)"),
    ("C2", "donor (LODO)"),
    ("C5", "cell-context (LOCT)"),
]
PANEL_B_TASKS = [
    ("C3", "unseen-perturbation (LO-gene 10%)"),
    ("C4", "unseen-KO (modality, RNA)"),
    ("C5", "unseen-compound"),
]
# axis-group label per task panel (the generalization axis the columns share).
PANEL_AXIS = {"a": "context / donor", "b": "unseen perturbation"}
# muted AXIS_COLORS hue for each panel's header rule (cell-context for a, perturbation for b),
# matching the shared semantic colour system (axis hues are muted; data marks are saturated).
PANEL_AXIS_HUE = {"a": AXIS_COLORS["donor"], "b": AXIS_COLORS["perturbation"]}

# short, self-documenting column tags. Two lines: dataset/cluster on the coloured band above,
# split name on the tick below.
# the coloured band over each single-column task carries only the cluster CODE (always fits its
# band); the dataset NAME moves onto the x-tick as the first line of a two-line tick (dataset \n
# split), so nothing overruns the narrow band. The full roster still lives in the caption.
TASK_CODE = {  # manuscript task code (T1-T5) shown in the band. The tuple KEYS remain the CSV
               # cluster tokens (C1-C5) that index the data; only the DISPLAYED label is the
               # task code, matching the manuscript, Table 1/3 and Figure 1c (C1->T1 ... C5->T5).
    ("C1", "cell-context (LOCT)"): "T1",
    ("C2", "donor (LODO)"): "T2",
    ("C5", "cell-context (LOCT)"): "T5",
    ("C3", "unseen-perturbation (LO-gene 10%)"): "T3",
    ("C4", "unseen-KO (modality, RNA)"): "T4",
    ("C5", "unseen-compound"): "T5",
}
TASK_TICK = {  # two-line x-tick: dataset name (top) then the leak-safe split (bottom)
    ("C1", "cell-context (LOCT)"): "Kang\nLOCT",
    ("C2", "donor (LODO)"): "Soskic\nLODO",
    ("C5", "cell-context (LOCT)"): "OP3\nLOCT",
    ("C3", "unseen-perturbation (LO-gene 10%)"): "CRISPR\nLO-gene",
    ("C4", "unseen-KO (modality, RNA)"): "Frang.\nmod-KO",
    ("C5", "unseen-compound"): "OP3\nLO-cpd",
}
# one refined slate band over each task column (cluster is a dataset grouping, not an axis -> no hue;
# a single editorial slate reads cleaner than five near-identical greys).
CLUSTER_COLOR = {c: SLATE_BAND for c in ("C1", "C2", "C3", "C4", "C5")}

# DIVERGING value colormap centred at zero = the universal floor, in the author's editorial palette:
# muted terracotta below (under floor), warm near-white at the floor line, the author's navy above
# (over floor). Lower chroma than the old rust/blue so the plate reads as a designed editorial plate.
CMAP = floor_diverging_cmap()
NA_FC = "#eef0f2"      # flat cool light-grey: (family, model) not run / not applicable on that task
LUM_T = 0.52           # luminance switch white<->near-black for the in-cell number
INK = "#1a1a1a"        # near-black ink
NEG_INK = CLAY_DARK    # muted-terracotta ink for negatives on a light fill (sign flagged + legible)
WIN_RING = "#D9A300"   # gold ring: beats BOTH universal-floor members (the §2 sparsity headline)


def load_headline():
    """Read the deposited 35-cell headline table; attach axis-panel + a stable task key per row."""
    df = pd.read_csv(HEADLINE)
    panel_a = {(c, s) for c, s in PANEL_A_TASKS}
    df["panel"] = [("a" if (c, s) in panel_a else "b") for c, s in zip(df.cluster, df.split)]
    df["task"] = list(zip(df.cluster, df.split))
    return df


def _fmt(v):
    """Compact signed 2-dp label; drop the leading zero, true unicode minus, NO negative-zero.

    Any value that rounds to .00 prints as a bare "0" (no sign, no decimals) so the grid never
    shows a negative-zero token like "−.00" — the cell colour already encodes which side of the
    floor it sits on. This is purely a display rule; the plotted/coloured value is untouched.
    """
    if abs(v) < 0.005:                       # rounds to 0.00 at 2 dp -> show a clean unsigned 0
        return "0"
    s = f"{v:.2f}"
    s = s.replace("-0.", "-.").replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s
    return s.replace("-", "−")


def _lum(rgb):
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _shared_roster(df):
    """The single SHARED row roster used by BOTH panels: every model that appears anywhere on the
    plate, listed ONCE in family-grouped FAM_ROWS order. A shared axis removes the duplicated row
    labels / accent bars and the unequal-height void — both panels render full-height against this
    one roster, with blank (not-run) cells where a model was not run on a panel's tasks.

    Returns (models, fam_of) where fam_of[m] is the model's display family.
    """
    present = set(df.model)
    models, fam_of = [], {}
    for fam, ms in FAM_ROWS:
        for m in ms:
            if m in present:
                models.append(m)
                fam_of[m] = fam
    return models, fam_of


def _build_block(df, tasks, models):
    """Return (cols, S, BEATS) for one axis panel ON THE SHARED ROSTER `models`.

    S      = delta_vs_floor_mean per (model, task) cell (the headline Δ-vs-universal-floor).
    BEATS  = boolean: that cell beats BOTH floor members (gold-ring highlight).
    cols   = the task list (cluster, split) present in this panel, in PANEL order.
    A model that was not run on any of this panel's tasks keeps an all-NaN row (flat grey).
    """
    cols = [t for t in tasks if t in set(df.task)]
    nM, nC = len(models), len(cols)
    S = np.full((nM, nC), np.nan)
    BEATS = np.zeros((nM, nC), dtype=bool)
    for i, m in enumerate(models):
        for j, c in enumerate(cols):
            r = df[(df.model == m) & (df.task == c)]
            if len(r):
                S[i, j] = float(r.delta_vs_floor_mean.iloc[0])
                BEATS[i, j] = bool(r.beats_both_floor_members.iloc[0])
    return cols, S, BEATS


# shared vertical geometry (data units). Both panel axes and the central gutter axis share the SAME
# row pitch and the SAME y-limits, so the two heatmaps and the shared roster line up to the pixel.
YHEAD_BAND = 1.62                    # headroom above the cells for the axis header + cluster band
                                     # (extra room so the top-left panel letter clears the header)
YFOOT = -1.05                        # axis bottom: fully contains the 2-line task ticks under the cells


def _draw_block(ax, cols, S, BEATS, norm, panel, nM):
    """Render ONE panel's heatmap (cells + cluster band + axis header + ticks) — NO row labels and
    NO family accent bars (those live ONCE in the shared central gutter). Cell-local coords:
    col j -> x=j, shared row i -> y=nM-1-i (i indexes the SHARED roster)."""
    nC = len(cols)
    for i in range(nM):
        for j in range(nC):
            x, y = j, nM - 1 - i
            if np.isnan(S[i, j]):                                  # not applicable -> flat grey
                ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fc=NA_FC,
                                           ec="white", lw=1.0, zorder=1))
                continue
            val = S[i, j]
            fill = CMAP(norm(val))
            dark = _lum(fill) < LUM_T
            ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fc=fill,
                                       ec="white", lw=1.0, zorder=1))
            # a value that rounds to "0" is printed as a neutral, sign-free token, so it takes the
            # neutral ink (NOT the under-floor clay) even when its raw sign is negative — the printed
            # glyph and its colour agree. Negatives that actually print a sign keep the clay ink.
            near_zero = abs(val) < 0.005
            txt_c = "white" if dark else (NEG_INK if (val < 0 and not near_zero) else INK)
            num_fs = 8.6 if (val < 0 and not near_zero) else 9.2
            ax.text(x, y, _fmt(val), ha="center", va="center",
                    fontsize=num_fs, color=txt_c, zorder=4, fontweight="medium")
            if BEATS[i, j]:                                        # gold ring: beats BOTH floor members
                ax.add_patch(plt.Rectangle((x - .42, y - .42), 0.84, 0.84, fc="none",
                                           ec=WIN_RING, lw=2.0, zorder=6))

    # ---- column ticks: two lines (dataset name + leak-safe split), drawn as TEXT just below the
    # cells (NOT axis ticks, which anchor to the far axis floor and would drift into the legend) ----
    tick_y = -0.5 - 0.16                              # a hair under the bottom cell row
    for j, c in enumerate(cols):
        ax.text(j, tick_y, TASK_TICK.get(c, c[1]), ha="center", va="top",
                fontsize=7.0, color="#222", linespacing=1.18, clip_on=False, zorder=6)
    ax.set_xticks([]); ax.set_yticks([])

    # ---- axis-group header + a span rule sized to the cell columns (the generalization axis) ----
    rule_y = nM - .5 + 0.78
    head_y = rule_y + 0.14
    acol = PANEL_AXIS_HUE[panel]
    col_x0, col_x1 = -.5 + 0.06, (nC - 1) + .5 - 0.06   # the cell-column span
    cx = (col_x0 + col_x1) / 2
    ax.text(cx, head_y, PANEL_AXIS[panel], ha="center", va="bottom",
            fontsize=9.4, weight="bold", color=acol, clip_on=False, zorder=6)
    ax.plot([col_x0, col_x1], [rule_y, rule_y], color=acol, lw=1.7, clip_on=False, zorder=6)

    # ---- coloured cluster band directly above the cells: bare cluster-code tags ----
    band_lo, band_hi = nM - .5 + 0.10, nM - .5 + 0.52
    for j, c in enumerate(cols):
        cl = c[0]
        x0b, x1b = j - .5 + 0.04, j + .5 - 0.04
        ax.add_patch(plt.Rectangle((x0b, band_lo), x1b - x0b, band_hi - band_lo,
                                   fc=CLUSTER_COLOR.get(cl, "#888"), ec="none", zorder=5,
                                   clip_on=False))
        ax.text((x0b + x1b) / 2, (band_lo + band_hi) / 2, TASK_CODE.get(c, cl),
                ha="center", va="center", fontsize=8.4, color="white",
                fontweight="bold", zorder=6, clip_on=False)
        if j < nC - 1:                                  # thin column separators between tasks
            ax.plot([j + .5, j + .5], [-.5, nM - .5], color="#cdcdcd", lw=0.8, zorder=4)

    ax.set_xlim(-.5, nC - .5)
    ax.set_ylim(YFOOT, nM - .5 + YHEAD_BAND)
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    ax.set_facecolor("none")
    for sp in ax.spines.values():
        sp.set_visible(False)


# central-gutter geometry (its own data-unit x-axis, shared y-axis with the panels).
# Read left->right: [panel a cells] | accent bar | model name | family group label | [panel b cells].
# The accent bar sits in the GUTTER, OUTSIDE every data cell (the reference grammar), so no value
# can ever touch a coloured bar. The roster is listed ONCE here for BOTH panels.
GUT_W = 3.05                         # gutter width in data units (== one "column-equivalent" span)
BAR_X0, BAR_X1 = 0.10, 0.24          # family accent bar, hard against the gutter's left edge
NAME_X = 0.40                        # left-anchored model name, clear of the accent bar
FAMLAB_X = GUT_W - 0.10              # right-anchored family group label, against the gutter's right edge


def _draw_gutter(ax, models, fam_of, nM):
    """Render the SINGLE shared row axis between the two panels: one accent bar + one model name +
    one family group label per family block. Coords: x in [0, GUT_W]; row i -> y = nM-1-i."""
    # model names, listed once.
    for i, m in enumerate(models):
        y = nM - 1 - i
        ax.text(NAME_X, y, MODEL_SHORT.get(m, m), ha="left", va="center",
                fontsize=8.0, color=INK, clip_on=False)
    # contiguous family blocks -> one accent bar + one direct group label each.
    yacc = nM - 1
    seen, order = set(), []
    for m in models:
        f = fam_of[m]
        if f not in seen:
            order.append(f); seen.add(f)
    for fam in order:
        present = [m for m in models if fam_of[m] == fam]
        y_hi = yacc + .5
        y_lo = yacc - len(present) + .5
        ymid = (y_hi + y_lo) / 2
        hue = FAM_COLORS[FAM_SPINE_KEY[fam]]
        ax.add_patch(plt.Rectangle((BAR_X0, y_lo + .12), BAR_X1 - BAR_X0,
                                   (y_hi - y_lo) - .24, fc=hue,
                                   ec="none", clip_on=False, zorder=5))
        ax.text(FAMLAB_X, ymid, FAM_LABEL[fam], ha="right", va="center",
                fontsize=8.2, color=hue, fontweight="bold", clip_on=False,
                linespacing=0.95, zorder=5)
        yacc -= len(present)
    ax.set_xlim(0, GUT_W)
    ax.set_ylim(YFOOT, nM - .5 + YHEAD_BAND)
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("none")
    for sp in ax.spines.values():
        sp.set_visible(False)


def main():
    set_pub_style()
    df = load_headline()
    # single diverging value scale shared across BOTH panels & every cell, centred at ZERO = the
    # universal floor. vmin/vmax are the true data extremes so nothing is clipped.
    vals = df.delta_vs_floor_mean.values.astype(float)
    vmin = min(float(np.nanmin(vals)), -1e-6)
    vmax = max(float(np.nanmax(vals)), 1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    # ONE shared roster -> both panels render full-height against it (no unequal-height void).
    models, fam_of = _shared_roster(df)
    cA, SA, BA = _build_block(df, PANEL_A_TASKS, models)
    cB, SB, BB = _build_block(df, PANEL_B_TASKS, models)
    nM = len(models)
    nCA, nCB = len(cA), len(cB)

    # ---- figure geometry: [panel a] | [shared gutter] | [panel b], all on one row grid ----
    spanY = (nM - .5 + YHEAD_BAND) - YFOOT          # identical for all three axes
    spanXA = nCA                                    # data-unit width of panel a's cells (col span)
    spanXB = nCB
    cell = 0.400                                    # inch per data unit (square cells)

    # Generous, balanced outer margins (~3-4% per side once converted to fractions below) and a
    # single bottom legend strip docked directly under the combined heatmap block.
    m_left = 0.62
    m_right = 0.62
    pad_in = 0.30                                   # airy pad between a panel's cells and the gutter
    m_top = 1.18                                    # title band + clean gap above the panel letters
    tick_gap_in = 0.20                              # clean air between the 2-line task ticks + legend
    legend_band_in = 0.80                           # single bottom legend strip
    m_bot = 0.34                                    # outer bottom whitespace

    panelA_w = cell * spanXA
    panelB_w = cell * spanXB
    gutter_w = cell * GUT_W
    body_h = cell * spanY                           # incl. the 2-line tick band (YFOOT extends below)
    fig_w = m_left + panelA_w + pad_in + gutter_w + pad_in + panelB_w + m_right
    fig_h = m_top + body_h + tick_gap_in + legend_band_in + m_bot
    fig = plt.figure(figsize=(fig_w, fig_h))

    cfx, cfy = cell / fig_w, cell / fig_h
    hbody = cfy * spanY
    wA, wG, wB = cfx * spanXA, cfx * GUT_W, cfx * spanXB

    top_frac = 1.0 - m_top / fig_h
    ybody = top_frac - hbody                         # all three axes share this top + height
    xA0 = m_left / fig_w
    xG0 = xA0 + wA + pad_in / fig_w
    xB0 = xG0 + wG + pad_in / fig_w

    axA = fig.add_axes([xA0, ybody, wA, hbody])
    axG = fig.add_axes([xG0, ybody, wG, hbody])
    axB = fig.add_axes([xB0, ybody, wB, hbody])
    for a in (axA, axG, axB):
        a.set_anchor("NW"); a.set_clip_on(False)

    _draw_block(axA, cA, SA, BA, norm, "a", nM)
    _draw_gutter(axG, models, fam_of, nM)
    _draw_block(axB, cB, SB, BB, norm, "b", nM)

    bbA, bbB = axA.get_position(), axB.get_position()
    top = top_frac

    # figure-y of a data-y within the body axes (all three share the same y-transform).
    def _fy(data_y):
        return ybody + (data_y - YFOOT) / spanY * hbody

    # ---- NO internal a/b panel letters: when this landscape is embedded as Figure 2a the outer
    # composite already carries the master panel letter "a", so internal a/b letters here would
    # duplicate it (nested a/b inside 2a). The two sub-blocks are instead identified solely by their
    # descriptive axis headers ("context / donor" and "unseen perturbation"), which sit centred above
    # each block — no separate letters needed. (letter_fy retained for any downstream geometry use.)
    letter_fy = _fy(nM - .5 + YHEAD_BAND)                   # top of the body axis (just under title)

    # ---- figure title centred over the COMBINED content block, in its own top band ----
    cx = (bbA.x0 + bbB.x1) / 2
    fig.text(cx, 1.0 - 0.34 / fig_h,
             r"Universal-floor landscape: model family $\times$ generalisation task",
             ha="center", va="top", fontsize=11.0, fontweight="bold", color=NAVY_DARK)

    # =================== one tidy legend strip docked under the heatmap block ===================
    # A single horizontal band directly beneath the combined plate (the reference grammar): the
    # Δ-vs-floor colorbar on the left, then two compact keys to its right. No bordered card, no
    # separate family colour-decode block (families are labelled directly in the gutter).
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([vmin, vmax])

    IN = lambda v: v / fig_h                                # inch -> figure-fraction (vertical)
    band_top = (m_bot + legend_band_in) / fig_h            # top of the legend band, hugging the cells
    cb_x = bbA.x0                                           # left-aligned to panel a
    cb_w_in = 1.85
    cb_w = cb_w_in / fig_w
    cb_h = IN(0.072)
    title_y = band_top - IN(0.02)                          # bar title baseline
    cb_y = title_y - IN(0.21)                              # bar sits below its title
    cax = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([vmin, 0.0, vmax])
    cb.set_ticklabels([_fmt(vmin), "0", _fmt(vmax)])
    cb.ax.tick_params(labelsize=8.0, length=0, pad=2.5)
    cb.outline.set_visible(False)
    _tls = cb.ax.get_xticklabels()
    if len(_tls) >= 3:
        _tls[0].set_ha("left"); _tls[-1].set_ha("right")
    fig.text(cb_x, title_y, r"cell = Pearson-$\Delta$ − floor",
             fontsize=8.6, ha="left", va="bottom", color=NAVY_DARK, fontweight="bold")
    fig.text(cb_x, cb_y - IN(0.30), "0 = universal floor   ·   clay below   ·   navy above",
             fontsize=7.8, ha="left", va="top", color=GREY_MID)

    # ---- the two compact swatch keys, stacked, to the right of the colorbar ----
    key_x0 = cb_x + cb_w + 0.60 / fig_w
    sw_in = 0.17
    sw_w = sw_in / fig_w
    sw_h = sw_in / fig_h
    lab_dx = (sw_in + 0.08) / fig_w
    key_hi = title_y - IN(0.06)
    key_lo = key_hi - IN(0.34)

    def _swatch_key(xc, yc, draw):
        a = fig.add_axes([xc, yc - sw_h / 2, sw_w, sw_h])
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
        draw(a)

    _swatch_key(key_x0, key_hi, lambda a: (
        a.add_patch(plt.Rectangle((0, 0), 1, 1, fc=CMAP(norm(vmax * 0.7)), ec="white", lw=0.8)),
        a.add_patch(plt.Rectangle((0.12, 0.12), 0.76, 0.76, fc="none", ec=WIN_RING, lw=2.0))))
    fig.text(key_x0 + lab_dx, key_hi, "beats both floor members",
             fontsize=8.2, ha="left", va="center", color=INK)
    _swatch_key(key_x0, key_lo, lambda a: a.add_patch(
        plt.Rectangle((0, 0), 1, 1, fc=NA_FC, ec=LEGEND_EC, lw=0.7)))
    fig.text(key_x0 + lab_dx, key_lo, "not run / not applicable",
             fontsize=8.2, ha="left", va="center", color=INK)

    out = ROOT / "results" / "_paper" / "figure_landscape.png"
    # NO bbox_inches="tight": the generous outer margins (m_left/m_right/m_top/m_bot, all >=0.34in
    # ~3-4% per side) are reserved INSIDE the figure, so saving the full canvas preserves them
    # rather than letting a tight crop pull the ink back to the frame.
    fig.savefig(out, dpi=400, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    # report the headline read so a render also QCs the count.
    nbeat = int(df.beats_both_floor_members.sum())
    print(f"wrote {out}  ({len(df)} cells; {nbeat} beat both floor members)")
    for _, r in df[df.beats_both_floor_members].iterrows():
        print(f"   WIN: {r.model} @ {r.cluster} {r.split}  (Δ-floor +{r.delta_vs_floor_mean:.3f})")


if __name__ == "__main__":
    main()
