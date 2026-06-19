#!/usr/bin/env python
r"""Figure 8 (v2 §4 + end-recommendation, the LAST/terminal figure): WITHIN-FAMILY CONSISTENCY + DESCRIPTIVE FIT-MATRIX.

Two panels, both built ONLY from deposited artifacts under results/_paper/ (no fabrication):

  panel a — WITHIN-FAMILY VERDICT-AGREEMENT MATRIX (v2 §4). Rows = method family, columns = task.
            Each cell encodes whether the >=2 evaluated members of that family reach the SAME
            beat-floor verdict (all-beat / none-beat = "agree") and the per-unit rank concordance
            (Spearman rho). Source: within_family_consistency.csv. Of the 13 populated cells, 12
            agree; the ONE within-family verdict SPLIT (clay) is the C2 donor OT pair — CellOT beats
            both floor members, scPRAM fails — i.e. the headline donor win splits its own family.
            A called-out DONOR EXCEPTION inset shows the same fact quantitatively: CellOT wins,
            scPRAM loses on the FINAL deposited paired donor comparison
            (scpram_vs_cellot_donor_paired.csv; scored at full n=106, scPRAM wins 1/106).

  panel b — DESCRIPTIVE / EXPLORATORY FIT-MATRIX (end recommendation, PREREGISTRATION rule 5).
            family x task: does the benchmark show the family beats the universal floor with
            cluster-bootstrap CI_low > 0? Observed gap+CI vs the a-priori expectation
            (conditioning expected to help context-transfer, not unseen-perturbation). Source:
            descriptive_fit_matrix.csv (scripts/assemble_fit_matrix.py -> fit_recommendation rule 5).
            Cells without a deposited/recomputable cluster bootstrap are drawn as PENDING.

Nature-redesign (this revision): adopts the author's own editorial grammar from Section2_2_Figure1 —
restrained navy system, generous whitespace, task columns labelled directly with a coloured
context/perturbation accent rule, a single tidy bottom legend (no tall colour-decode column), clean
sans typography, perfect grid alignment. Every plotted value, count and verdict is IDENTICAL to the
prior version; only composition / layout / colour / spacing / labels / typography changed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ivcbench.report.style import (set_pub_style, INK, GREY_MID, GREY_LITE,  # noqa: E402
                                   NAVY, NAVY_DARK, AXIS_COLORS, CONDITIONED_DARK,
                                   SIMPLE_DARK, SIMPLE_GREY, NULL_GREY, NAVY_RAMP,
                                   CLAY_DARK, LEGEND_EC)

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results" / "_paper"

# --- task column order (context-transfer axes first, then unseen-perturbation), matches §3/§4 ----
# (cluster, split, two-line column label). The dataset tag is carried on the bottom legend, not
# repeated under every column, so the column heads stay short and uncluttered.
TASK_ORDER = [
    # (CSV cluster key, split, two-line displayed column label). The KEY stays the CSV cluster
    # token (C1-C5) that indexes the data; the displayed label uses the manuscript task code
    # (T1-T5) so the figure matches the manuscript, Table 1/2 and Figure 1c (C1->T1 ... C5->T5).
    ("C1", "cell-context (LOCT)", "T1\ncell-context"),
    ("C2", "donor (LODO)", "T2\ndonor"),
    ("C5", "cell-context (LOCT)", "T5\ncell-context"),
    ("C4", "unseen-KO (modality, RNA)", "T4\nmodality"),
    ("C3", "unseen-perturbation (LO-gene 10%)", "T3\nunseen-pert"),
    ("C5", "unseen-compound", "T5\nunseen-cpd"),
]
N_CTX = 3                    # first three columns are context-transfer; the rest unseen-perturbation
FAM_ROWS = ["Latent", "Graph", "Foundation", "Hybrid", "Chemistry", "OT"]

# --- jewel-register matrix-cell fills, all from named style tokens (one navy "beats"; one light-navy
#     "consistent"; one clean coral-rust accent for the lone split; one neutral grey for PENDING;
#     one faintest grey for empty) — no muddy sienna, no washed sand pastels ----------------------
BEATS_FC = NAVY            # family BEATS the floor (panel b only — CI_low>0)
BEATS_TX = "white"
AGREE_FC = NAVY_RAMP[1]    # agree — consistent verdict (light end of the one navy ramp); common case
SPLIT_FC = CLAY_DARK       # within-family SPLIT — the clean coral-rust counter-pole (the C2 OT donor cell)
SPLIT_TX = "white"
PEND_FC = NULL_GREY        # PENDING (single-seed / n<3 units) — the neutral background grey, clearly non-navy
EMPTY_FC = "#EEF1F4"       # <2 members / family not on this task — faintest neutral grey
INK_CELL = INK             # in-cell dark text on light fills


def umin(s: str) -> str:
    return s.replace("-", "−")


def fmt(v, nd=3, signed=True):
    """Format a value with a TRUE unicode minus and NO negative-zero artefact (−0.000 -> 0.000)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if abs(v) < 0.5 * 10 ** (-nd):      # collapse anything that would round to zero -> clean 0
        v = 0.0
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return s.replace("-", "−")


def rho_fmt(rho):
    """Spearman rho as ρ=+x.xx with a true minus and no negative-zero."""
    if rho is None or (isinstance(rho, float) and not np.isfinite(rho)):
        return None
    if abs(rho) < 0.005:
        rho = 0.0
    return ("ρ=" + f"{rho:+.2f}").replace("-", "−")


def load_consistency():
    return pd.read_csv(PAPER / "within_family_consistency.csv")


def load_fit():
    return pd.read_csv(PAPER / "descriptive_fit_matrix.csv")


def load_donor_paired():
    d = pd.read_csv(PAPER / "scpram_vs_cellot_donor_paired.csv").iloc[0]
    return dict(n=int(d["n_shared_donors"]), wins=int(d["scpram_wins"]),
                gap=float(d["mean_gap"]), scpram=float(d["scpram_mean"]),
                cellot=float(d["cellot_mean"]), p=float(d["wilcoxon_p"]), note=str(d["note"]))


# --- shared grid scaffolding (column heads + context/perturbation accent rule + row labels) ------
CTX_COL = AXIS_COLORS["cell-context"]    # muted slate-blue — context-transfer columns
PERT_COL = AXIS_COLORS["perturbation"]   # muted rust       — unseen-perturbation columns


def draw_grid_frame(ax, rows, nC, *, title_gap=False):
    """Column heads, the context|perturbation accent rule, row labels and divider — shared by a/b.

    The accent rule is a thin coloured bar UNDER the column heads (the author's grammar: groups are
    labelled directly by a coloured rule, not decoded in a separate legend)."""
    nR = len(rows)
    ax.set_xlim(-0.5, nC - 0.5)
    ax.set_ylim(-0.5, nR - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # context | perturbation divider (a calm dotted hairline between col 2 and 3)
    ax.axvline(N_CTX - 0.5, color="#D5DBE0", lw=0.9, ls=(0, (1.5, 2.5)), zorder=1)
    # row labels (family) — left, clean
    for i, fam in enumerate(rows):
        y = nR - 1 - i
        ax.text(-0.66, y, fam, ha="right", va="center", fontsize=7.4, color=INK)


def draw_column_heads(ax, nC, nR, y_top):
    """Two-line column heads + a coloured accent rule directly beneath the group labels.

    Vertical stack (bottom -> top), all in data units above the matrix top edge y_top:
        head (cluster + sub-task)  ->  accent rule  ->  group label.
    Kept compact so the head block never reaches into the panel title above it."""
    head_y = y_top + 0.34          # baseline of the per-column cluster head
    rule_y = y_top + 0.92          # the coloured accent rule
    grp_y = y_top + 1.02           # group label, just above the rule
    # the coloured accent rule, drawn as two short bars (context = slate-blue, perturbation = rust)
    ax.plot([-0.42, N_CTX - 0.5 - 0.08], [rule_y, rule_y], color=CTX_COL, lw=2.4,
            solid_capstyle="butt", clip_on=False, zorder=3)
    ax.plot([N_CTX - 0.5 + 0.08, nC - 0.5 + 0.42], [rule_y, rule_y], color=PERT_COL, lw=2.4,
            solid_capstyle="butt", clip_on=False, zorder=3)
    # group labels sit just above the rule, centred over their span
    ax.text((-0.42 + N_CTX - 0.5) / 2, grp_y, "CONTEXT-TRANSFER", ha="center", va="bottom",
            fontsize=6.5, color=CTX_COL, weight="bold", clip_on=False)
    ax.text((N_CTX - 0.5 + nC - 0.5 + 0.42) / 2, grp_y, "UNSEEN-PERTURBATION", ha="center",
            va="bottom", fontsize=6.5, color=PERT_COL, weight="bold", clip_on=False)
    # per-column two-line heads (cluster bold on top, sub-task grey below)
    for j, (_, _, lab) in enumerate(TASK_ORDER):
        cl, sub = lab.split("\n")
        col = CTX_COL if j < N_CTX else PERT_COL
        ax.text(j, head_y + 0.16, cl, ha="center", va="bottom", fontsize=7.2, color=INK,
                weight="bold", clip_on=False)
        ax.text(j, head_y - 0.10, sub, ha="center", va="bottom", fontsize=6.0, color=col,
                clip_on=False)


def cell(ax, x, y, fc, ec="white"):
    ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, fc=fc, ec=ec, lw=1.6, zorder=2))


# ============================== panel a: verdict-agreement matrix ==============================
def panel_agreement(ax, con):
    nR, nC = len(FAM_ROWS), len(TASK_ORDER)
    con = con.copy()
    con["key"] = list(zip(con.cluster, con.split, con.family))
    lut = {k: r for k, r in con.set_index("key").iterrows()}

    draw_grid_frame(ax, FAM_ROWS, nC)
    draw_column_heads(ax, nC, nR, nR - 0.5)

    for j, (cl, sp, _) in enumerate(TASK_ORDER):
        for i, fam in enumerate(FAM_ROWS):
            x, y = j, nR - 1 - i
            row = lut.get((cl, sp, fam))
            if row is None:
                cell(ax, x, y, EMPTY_FC)
                continue
            flagged = isinstance(row["flag"], str) and row["flag"].strip() != ""
            n_beat = int(row["n_beat_both_floor"])
            n_mod = int(row["n_models"])
            split = str(row["verdict_agreement"]).strip().lower() == "split"
            agree = row["verdict_agreement"] == "agree"
            rho = row["spearman_rho_pair"]
            # SPLIT priority (C2 OT) > flagged PENDING (C4) > agree-all / agree-none
            if split:
                fc, tcol = SPLIT_FC, SPLIT_TX
            elif flagged:
                fc, tcol = PEND_FC, INK_CELL
            elif agree and n_beat == n_mod and n_mod > 0:
                fc, tcol = BEATS_FC, BEATS_TX     # all members beat the floor (none in panel a data)
            else:
                fc, tcol = AGREE_FC, INK_CELL     # none beat (consistent-failure case)
            cell(ax, x, y, fc)

            if split:
                ax.text(x, y + 0.205, "SPLIT", ha="center", va="center", fontsize=6.6,
                        color=tcol, weight="bold")
                ax.text(x, y - 0.18, "CellOT beats\nscPRAM fails", ha="center", va="center",
                        fontsize=5.0, color="white", style="italic", linespacing=1.25)
            elif flagged:
                ax.text(x, y + 0.235, "none beat", ha="center", va="center", fontsize=6.1,
                        color=GREY_MID, weight="bold")
                sub = "single-seed" if fam == "OT" else "n<3 units"
                ax.text(x, y - 0.165, "PENDING", ha="center", va="center", fontsize=5.4,
                        color=CONDITIONED_DARK, weight="bold", style="italic")
                ax.text(x, y - 0.345, sub, ha="center", va="center", fontsize=5.0,
                        color=GREY_MID, style="italic")
            else:
                ax.text(x, y + 0.18, "none beat", ha="center", va="center", fontsize=6.4,
                        color=tcol, weight="bold")
                rs = rho_fmt(rho)
                if rs is None:
                    ax.text(x, y - 0.20, "ρ n.a.", ha="center", va="center", fontsize=5.6,
                            color=GREY_LITE)
                else:
                    ax.text(x, y - 0.20, rs, ha="center", va="center", fontsize=6.0,
                            color=GREY_MID)


# ============================== donor-exception inset (within panel a) ==========================
def _p_mathtext(p):
    """Mathtext $m{\\times}10^{e}$ for a small p-value (true minus in the exponent; do NOT umin())."""
    if p is None or not np.isfinite(p) or p <= 0:
        return "—"
    exp = int(np.floor(np.log10(p)))
    mant = p / (10 ** exp)
    return rf"$p\,{{=}}\,{mant:.1f}{{\times}}10^{{{exp}}}$"


def donor_inset(ax, dp):
    """The one place the headline win does NOT hold within-family: CellOT wins, scPRAM loses on the
    FINAL deposited paired shared-donor comparison (n=106). Every number read from the CSV."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.035, 0.06), 0.93, 0.88,
                                boxstyle="round,pad=0.0,rounding_size=0.045",
                                fc="white", ec=LEGEND_EC, lw=0.9, transform=ax.transAxes,
                                clip_on=False))
    # a slim clay accent rule on the left edge ties the inset to the SPLIT cell it explains
    ax.add_patch(Rectangle((0.035, 0.06), 0.016, 0.88, fc=SPLIT_FC, ec="none",
                           transform=ax.transAxes, clip_on=False))
    ax.text(0.085, 0.86, "OT-family donor exception", transform=ax.transAxes, fontsize=6.8,
            weight="bold", color=NAVY_DARK, va="center")
    ax.text(0.085, 0.745, "the lone donor win is\nCellOT-specific, not family-level",
            transform=ax.transAxes, fontsize=5.3, color=GREY_MID, style="italic", va="top",
            linespacing=1.3)
    # two stacked bars: CellOT (navy, above floor) vs scPRAM (grey, below). The bar field and its
    # value labels are sized so the LONGEST label (CellOT "0.369") clears the inset's right inner
    # edge (axes-x 0.965) with margin — the bar max-width and start are pulled in accordingly.
    yC, yS = 0.555, 0.40
    x0 = 0.40
    sc = 0.26 / max(dp["cellot"], dp["scpram"], 1e-6)
    ax.add_patch(Rectangle((x0, yC - 0.040), dp["cellot"] * sc, 0.080, fc=NAVY,
                           ec="none", transform=ax.transAxes, clip_on=False))
    ax.add_patch(Rectangle((x0, yS - 0.040), dp["scpram"] * sc, 0.080, fc=SIMPLE_GREY,
                           ec="none", transform=ax.transAxes, clip_on=False))
    ax.text(x0 - 0.025, yC, "CellOT", transform=ax.transAxes, ha="right", va="center",
            fontsize=6.2, color=NAVY_DARK, weight="bold")
    ax.text(x0 - 0.025, yS, "scPRAM", transform=ax.transAxes, ha="right", va="center",
            fontsize=6.2, color=SIMPLE_DARK, weight="bold")
    ax.text(x0 + dp["cellot"] * sc + 0.02, yC, fmt(dp["cellot"], 3, signed=False),
            transform=ax.transAxes, ha="left", va="center", fontsize=6.0, color=INK)
    ax.text(x0 + dp["scpram"] * sc + 0.02, yS, fmt(dp["scpram"], 3, signed=False),
            transform=ax.transAxes, ha="left", va="center", fontsize=6.0, color=INK)
    # bottom stat block — three tidy lines, evenly stacked, each clearing the right AND bottom inner
    # edges (the Wilcoxon line carries a mathtext superscript exponent, so its anchor is raised so the
    # descender of 10^{−19} stays inside the box bottom at axes-y 0.060).
    ax.text(0.085, 0.300, umin(f"paired n = {dp['n']} shared donors"),
            transform=ax.transAxes, fontsize=5.6, color=INK, va="center")
    ax.text(0.085, 0.210, umin(f"scPRAM wins {dp['wins']}/{dp['n']}   gap {fmt(dp['gap'])}"),
            transform=ax.transAxes, fontsize=5.6, color=INK, va="center")
    ax.text(0.085, 0.122, "Wilcoxon  " + _p_mathtext(dp["p"]),
            transform=ax.transAxes, fontsize=5.5, color=GREY_MID, va="center", style="italic")


# ============================== panel b: descriptive fit-matrix ================================
def panel_fit(ax, fit):
    fit = fit.copy()
    famcap = {"latent": "Latent", "graph": "Graph", "foundation": "Foundation",
              "hybrid": "Hybrid", "chemistry": "Chemistry", "ot": "OT", "shift": "Shift"}
    fit["famC"] = fit["family"].map(lambda x: famcap.get(str(x), str(x).title()))
    ROWS = ["Latent", "Graph", "Foundation", "Hybrid", "Chemistry", "Shift"]
    lut = {(r.cluster, r.task_split, r.famC): r for r in fit.itertuples()}

    nR, nC = len(ROWS), len(TASK_ORDER)
    draw_grid_frame(ax, ROWS, nC)
    draw_column_heads(ax, nC, nR, nR - 0.5)

    for j, (cl, sp, _) in enumerate(TASK_ORDER):
        for i, fam in enumerate(ROWS):
            x, y = j, nR - 1 - i
            r = lut.get((cl, sp, fam))
            if r is None:
                cell(ax, x, y, EMPTY_FC)
                continue
            ci_low = r.ci_low
            gap = r.best_model_gap
            pending = pd.isna(ci_low)
            works = bool(r.works)
            if pending:
                fc, tcol = PEND_FC, INK_CELL
            elif works:
                fc, tcol = BEATS_FC, BEATS_TX     # family beats floor (CI_low>0)
            else:
                fc, tcol = AGREE_FC, INK_CELL     # recommend simple floor
            cell(ax, x, y, fc)
            ax.text(x, y + 0.205, fmt(gap, 3), ha="center", va="center", fontsize=6.4,
                    color=tcol, weight="bold")
            if pending:
                ax.text(x, y - 0.165, "PENDING", ha="center", va="center", fontsize=5.4,
                        color=CONDITIONED_DARK, weight="bold", style="italic")
                ax.text(x, y - 0.345, "n<3 units", ha="center", va="center", fontsize=5.0,
                        color=GREY_MID, style="italic")
            else:
                cic = "white" if works else GREY_MID
                ax.text(x, y - 0.20, umin(f"CI_lo {ci_low:+.3f}"), ha="center", va="center",
                        fontsize=5.6, color=cic)


# ============================== single tidy bottom legend =======================================
def bottom_legend(fig, y):
    """One tidy horizontal legend strip for the whole plate (navy = beats, slate = consistent, clay =
    split, sand = pending, grey = empty) — the single legend, matching the reference's economy.

    Keys are spaced by MEASURED label width so the gap between each label's end and the next swatch
    is CONSTANT (the reference's even rhythm), instead of a fixed pitch that bunches the long keys
    against the short ones. The whole strip is then centred in the figure. Swatches are TRUE squares
    (the x/y fraction is corrected for the very wide, short legend-axes aspect)."""
    ax_h = 0.050
    ax = fig.add_axes([0.0, y, 1.0, ax_h]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    items = [(BEATS_FC, "white", "beats floor (CI_low > 0)"),
             (AGREE_FC, "white", "consistent: none beat"),
             (SPLIT_FC, "white", "within-family split"),
             (PEND_FC, LEGEND_EC, "pending (n<3 / single-seed)"),
             (EMPTY_FC, LEGEND_EC, "not evaluated")]
    fig_w_in, fig_h_in = fig.get_size_inches()
    aspect = (1.0 * fig_w_in) / (ax_h * fig_h_in)
    sw_x = 0.0125                      # swatch width in axes-x fraction
    sw_y = sw_x * aspect               # equal visual height
    sw_gap = 0.0085                    # swatch -> its own label gap
    key_gap = 0.034                    # CONSTANT gap from one label's end to the next swatch
    yb = 0.50
    fs = 6.4

    # measure each label's rendered width (in axes-x fraction) so spacing is by content, not a fixed
    # pitch — this is what makes every swatch-label pair equally spaced across the row.
    fig.canvas.draw()                  # ensure a renderer exists for text extents
    rend = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    lab_w = []
    for _, _, lab in items:
        t = ax.text(0, 0, lab, transform=ax.transAxes, fontsize=fs, alpha=0)
        bb = t.get_window_extent(renderer=rend)
        (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        lab_w.append(x1 - x0)
        t.remove()

    unit_w = [sw_x + sw_gap + w for w in lab_w]          # full width of each swatch+label pair
    total = sum(unit_w) + key_gap * (len(items) - 1)     # + constant gaps between pairs
    x = (1.0 - total) / 2.0                              # centre the whole strip in the figure
    for (fc, ec, lab), w, uw in zip(items, lab_w, unit_w):
        ax.add_patch(Rectangle((x, yb - sw_y / 2), sw_x, sw_y, fc=fc, ec=ec, lw=0.9,
                               transform=ax.transAxes, clip_on=False))
        ax.text(x + sw_x + sw_gap, yb, lab, transform=ax.transAxes, va="center", ha="left",
                fontsize=fs, color=INK)
        x += uw + key_gap


def main():
    set_pub_style()
    PAPER.mkdir(parents=True, exist_ok=True)
    con = load_consistency()
    fit = load_fit()
    dp = load_donor_paired()

    fig = plt.figure(figsize=(7.2, 7.6))
    # two stacked matrix panels (left) sharing a column grid; the donor inset floats top-right of a.
    # Heights/positions chosen so each panel has: title band -> group-rule band -> column heads ->
    # matrix, with generous whitespace between bands and a clean gutter to the bottom legend.
    # Margins are kept uniform and generous (~0.05 fig-fraction, ~150px on the 2960px canvas) on all
    # four sides; the inset's right edge is pulled inward to RGT so it never crowds the frame.
    L, W = 0.115, 0.60                 # left margin 0.115; matrix block ends at 0.715
    RGT = 0.965                        # uniform right limit for the inset + the editorial tag
    INS_W = 0.195
    axA = fig.add_axes([L, 0.560, W, 0.270])
    axB = fig.add_axes([L, 0.135, W, 0.270])
    ax_ins = fig.add_axes([RGT - INS_W, 0.585, INS_W, 0.215])

    panel_agreement(axA, con)
    donor_inset(ax_ins, dp)
    panel_fit(axB, fit)

    # panel letters + titles — placed ABOVE the group-label band (which tops out near y=1.18 axes).
    # The per-panel in-cell value key is folded into the panel subtitle (the reference's single-
    # legend economy: no separate grey caption band beneath the legend).
    for ax, letter, title, sub in (
            (axA, "a", "Within-family verdict agreement",
             "cell value: ρ = within-family per-unit Spearman concordance"),
            (axB, "b", "Descriptive fit-matrix: does the family beat the universal floor?",
             "cell value: best-member gap vs universal floor (CI_lo = cluster-bootstrap lower bound)")):
        ax.text(-0.135, 1.295, letter, transform=ax.transAxes, fontsize=12, fontweight="bold",
                va="bottom", ha="left", color=INK)
        ax.text(-0.075, 1.300, title, transform=ax.transAxes, fontsize=8.6, fontweight="bold",
                va="bottom", ha="left", color=NAVY_DARK)
        ax.text(-0.075, 1.218, sub, transform=ax.transAxes, fontsize=6.4, color=GREY_MID,
                va="bottom", ha="left", style="italic")

    bottom_legend(fig, 0.040)

    # figure title + descriptive tag (the locked-charter PREREG framing). The title sits at the very
    # top of the canvas, clearing the panel-a title band below; the editorial tag is pulled in to RGT
    # so it clears the right frame by the same generous gap. A uniform savefig pad (below) then gives
    # all four sides an equal, generous border after the tight crop.
    fig.text(L, 0.968, "Within-family consistency and the descriptive fit-matrix",
             ha="left", va="bottom", fontsize=10.5, weight="bold", color=INK)
    fig.text(L, 0.946, "Of 13 evaluated family × task cells, 12 agree; the lone within-family "
             "split is the T2 donor OT pair.", ha="left", va="bottom", fontsize=7.2,
             color=GREY_MID, style="italic")
    fig.text(RGT, 0.968, "DESCRIPTIVE / EXPLORATORY", ha="right", va="bottom", fontsize=7.0,
             color=NAVY, weight="bold")
    fig.text(RGT, 0.947, "PREREG rule 5, not a hypothesis test", ha="right", va="bottom",
             fontsize=6.2, color=GREY_MID, style="italic")

    out = PAPER / "figure_within_family_fit.png"
    # uniform, generous border on all four sides (~0.30 in ≈ 120 px at 400 dpi) so no element runs to
    # the frame and the left/right/top/bottom margins read as even, matching the reference figure.
    fig.savefig(out, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.30)
    fig.savefig(out.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.30)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
