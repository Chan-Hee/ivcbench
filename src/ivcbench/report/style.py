"""Publication style for benchmark figures — Nature-grade defaults.

Okabe–Ito colorblind-safe palette, thin despined axes, outward ticks, sans-serif typography,
panel letters. Import set_pub_style() before plotting; use FAMILY_COLORS / SPLIT_COLORS and the
despine / panel_label helpers.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Jewel categorical register — colour-blind-safe, well-separated around the hue wheel, cohesive
# with the #0F4C75 navy ink. Replaces the washed Okabe–Ito set as the legacy categorical source.
# (Key kept for back-compat; values are the new jewel hexes so legacy figures inherit the system.)
OKABE_ITO = {
    "blue": "#3A4FB8", "orange": "#E8A317", "green": "#12A594", "vermillion": "#E8704F",
    "purple": "#C13E8E", "sky": "#2E8BC9", "yellow": "#F0C419", "grey": "#9AA3AD", "black": "#1a1a1a",
}

# method family -> color (model taxonomy, used in leaderboard)
FAMILY_COLORS = {
    "simple": OKABE_ITO["grey"], "latent": OKABE_ITO["purple"], "foundation": OKABE_ITO["orange"],
    "ot": OKABE_ITO["blue"], "graph": OKABE_ITO["green"], "hybrid": OKABE_ITO["vermillion"],
    "chemistry": OKABE_ITO["sky"],
}
# split -> color (categorical, assigned in order) — jewel register, well-separated hues
SPLIT_COLORS = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"],
                OKABE_ITO["vermillion"], OKABE_ITO["purple"]]

# canonical baseline ordering (family-grouped), intersected with what's present
BASELINE_ORDER = ["ctrl-pred", "cell-mean", "donor-shift", "linear-PCA", "FP-ridge", "scGen", "CPA",
                  "GEARS", "AttentionPert", "scGPT", "UCE", "CellOT", "CINEMA-OT", "STATE"]

# ============================================================================================
# CROSS-FIGURE SEMANTIC COLOUR SYSTEM — the single source of truth for the 5-figure plate.
# TWO DISJOINT, NON-OVERLAPPING colour namespaces. One colour = one concept in EVERY figure.
#
#   (1) AXIS_COLORS  — the four generalization axes (cell-context / perturbation / modality /
#                      donor). Used ONLY for axis CARDS (Fig1) and cluster/axis band HEADERS
#                      (Fig2). These are MUTED / DESATURATED hues so an axis card colour can
#                      never be mistaken for a saturated data mark.
#
#   (2) MODEL / FAMILY tokens — the model taxonomy and the conditioned-vs-simple contrast.
#                      Used ONLY for DATA MARKS (Fig2–5). These are SATURATED. One grey for the
#                      simple/floor family everywhere; one accent (purple) for conditioned/deep
#                      everywhere; per-family hues for the ranking spine; FP-ridge = the chemistry
#                      hue everywhere; the CITE markers (Fig4c) are a separate disjoint trio.
#
# The two namespaces are kept visually separable: AXIS hues are muted (low chroma / pastel-leaning
# ink), MODEL hues are saturated. NO model/family token reuses an axis hue and vice-versa, so the
# historical collisions are structurally impossible:
#   * the perturbation axis is the ONLY user of its muted-rust hue (the hybrid family now uses a
#     saturated teal, NOT the old #D55E00);
#   * the cell-context axis is the ONLY user of its muted-blue hue (the opt-transport family now
#     uses a saturated indigo, NOT the old #0072B2; scGen reads as the conditioned purple);
#   * the simple/floor family is ONE grey #9e9e9e everywhere; #BDBDBD is null/background only.
# ============================================================================================

# --- (1) generalization-axis colours — JEWEL TINTS, axis CARDS + band HEADERS only ------------
# One step LIGHTER and a touch lower-chroma than the data-mark register so an axis band reads as
# a calm header behind the saturated marks — but these are CLEAN jewel tints (real chroma), never
# the grey-washed / muddy-sienna hues that were rejected. They are the light siblings of the data
# register: indigo / coral / teal / plum, harmonising with the #0F4C75 navy ink.
AXIS_COLORS = {
    "cell-context": "#4F7CC0",   # bright cornflower-indigo (light sibling of the indigo data hue)
    "perturbation": "#E8704F",   # clean coral             — RESERVED for the perturbation axis only
    "modality":     "#2FA59A",   # clear teal
    "donor":        "#9A6FC0",   # vivid amethyst
}

# --- (2) model-family colours — SATURATED, the ranking-spine taxonomy (DATA MARKS only) ------
# Canonical in the ranking figure; reused for any per-family mark across the plate. No family
# colour equals any AXIS_COLORS value, so a family spine never reads as an axis band.
FAM_COLORS = {
    "simple":        "#9AA3AD",  # simple / linear floor family — the one cool slate-grey everywhere
    "latent":        "#C13E8E",  # latent / conditioned-deep family (scGen, CPA) = CONDITIONED magenta
    "hybrid":        "#12A594",  # hybrid (STATE) — jewel teal (NOT the perturb-axis coral)
    "foundation":    "#E8A317",  # foundation models (scGPT, UCE) — warm gold
    "graph":         "#8E3FB0",  # graph models (GEARS, AttentionPert) — jewel violet
    "chemistry":     "#2E8BC9",  # chemistry models (FP-ridge) — the chemistry sky-blue everywhere
    "shift":         "#B5762B",  # KO-embedding shift family — bronze (gold's deeper sibling)
    "opt-transport": "#3A4FB8",  # optimal-transport family (CINEMA-OT) — jewel indigo
}

# --- (2) reserved single-purpose MODEL tokens (the conditioned-vs-simple contrast) -----------
CONDITIONED = "#C13E8E"   # "conditioned / deep" model accent — one saturated magenta everywhere (= latent)
CONDITIONED_DARK = "#8E2566"  # legible-on-white text/callout variant of the conditioned accent
SIMPLE_GREY = "#9AA3AD"   # simple / baseline-floor family — one cool slate-grey everywhere
SIMPLE_DARK = "#5B6470"   # legible-on-white text/edge variant of the simple-floor grey
NULL_GREY   = "#C7CDD4"   # null / background / "exactly 0" — reserved strictly for null/background
CHEMISTRY   = "#2E8BC9"   # chemistry-family hue (FP-ridge) — one sky-blue everywhere (= FAM_COLORS["chemistry"])
HYBRID      = "#12A594"   # hybrid-family hue (STATE) — one jewel teal everywhere (= FAM_COLORS["hybrid"])
HYBRID_DARK = "#0B7468"   # legible-on-white text/edge variant of the hybrid hue

# --- (2) CITE-marker trio (Fig4c only) — a DISJOINT accent set, never a family/axis hue -------
# Three checkpoint/HLA markers, deliberately distinct from every family and axis colour above.
CITE_COLORS = {
    "PD-1":   "#12A594",  # jewel teal   -> PD-1 (CD279), recovered
    "HLA-I":  "#9A6FC0",  # amethyst     -> HLA class I (the HLA-A class-I marker)
    "HLA-I2": "#6D3F94",  # deep amethyst-> the second HLA class-I marker (HLA-E), same family hue
    "PD-L1":  "#E8704F",  # coral        -> PD-L1 (CD274), the un-recovered marker (warm = miss)
}

# --- (3) FRAMEWORK INFOGRAPHIC register (Figure 1 only) — GEARS-clean, navy-monochrome -----------
# Figure 1 is an EXPLANATORY framework infographic, not a data plot: its accent bars / pills / chips
# LABEL concepts the text already names, so saturated per-axis colour there is decoration, not
# information. Firing the four saturated AXIS_COLORS across the (a) cards, (b) metric chips, and
# (c) row anchors made the plate read busy ("6+ competing hues"). GEARS-clean rule for Figure 1:
# let NAVY be the single structural ink for EVERY card/metric accent bar and dataset-row anchor, and
# spend the four axis hues in EXACTLY ONE place — the (c) "Axes tested" pills, the one spot where a
# reader maps colour -> axis (decoded by the in-place key on the same row of the panel-c header).
# That collapses the whole plate to: navy everywhere + one compact four-hue legend used once.
# These are MUTED siblings of the AXIS_COLORS data hues (lower chroma) so the single pill strip reads
# as a calm key, never as saturated data marks; they do NOT replace AXIS_COLORS (which stay reserved
# for the real data semantics in Figure 2 + supplements).
FRAMEWORK_INK   = "#0F4C75"   # = NAVY: the one structural accent for all Fig-1 cards / bars / anchors
FRAMEWORK_PILLS = {           # the ONLY coloured marks in Fig 1 — the (c) axis pills + their key
    "cell-context": "#6E94C9",  # calm cornflower (muted sibling of AXIS_COLORS cell-context)
    "perturbation": "#E0876B",  # calm coral      (muted sibling of AXIS_COLORS perturbation)
    "modality":     "#5BB3A9",  # calm teal       (muted sibling of AXIS_COLORS modality)
    "donor":        "#A98BCB",  # calm amethyst   (muted sibling of AXIS_COLORS donor)
}

# --- editorial NAVY system — calibrated to the author's own figures (Section2_2 / OnePagers) ----
# The author's design language is a restrained navy-centric palette: one deep navy + a sequential
# ramp + a muted terracotta counter-pole, used for headers, bands and the floor diverging colormap
# so the plate reads as a designed editorial plate, not a default matplotlib science figure.
NAVY        = "#0F4C75"   # signature deep navy (the author's "strong" fill / structural ink)
NAVY_DARK   = "#0A3556"   # darker navy — header / title ink
SLATE_BAND  = "#33485C"   # one refined slate for neutral grouping bands (replaces ad-hoc greys)
NAVY_RAMP   = ["#EAF1F6", "#C2D7E6", "#8FB4D2", "#5A8BB8", "#2E6494", "#0F4C75"]  # clean light -> deep navy
CLAY_DARK   = "#C24E32"   # coral-rust counter-pole ink (under-floor negatives; clean, NOT muddy sienna)


def floor_diverging_cmap():
    """Diverging colormap for the universal-floor plate, in the author's palette.

    Muted terracotta (under floor) -> warm near-white (the floor line) -> the author's navy ramp
    (over floor). Lower chroma than a default RdBu so it sits as an editorial, not a science-default,
    plate; the navy over-floor pole is the author's signature #0F4C75.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "floor_div_v2",
        ["#C24E32", "#E08769", "#F0C3B2", "#F8EEE9",            # under floor: clean coral-rust -> near-white
         "#E3ECF3", "#8FB4D2", "#2E6494", "#0F4C75", "#0A3556"])  # over floor: light -> author navy


# --- ink + a small fixed secondary-grey ramp (replaces ad-hoc per-figure greys) ---
INK       = "#1a1a1a"     # near-black: all primary axis/annotation text + marker edges
GREY_MID  = "#555555"     # one mid-grey: secondary text / subtitles
GREY_LITE = "#888888"     # one light-grey: tertiary text / faint annotation

# --- shared typographic + frame tokens ---
FS_PANEL_LETTER = 12      # panel-letter font size, fixed across the whole plate set
FS_PANEL_TITLE  = 8.5     # panel-title font size (bold)
SUBTITLE_GREY   = "#555555"   # one grey for all italic panel subtitles
LEGEND_EC       = "#cccccc"   # one legend-frame edge colour everywhere
LEGEND_ALPHA    = 1.0         # one legend framealpha everywhere


def panel_title(ax, letter, title, sub=None, x_letter=-0.12, y=1.06, sub_dy=0.046,
                fs_letter=FS_PANEL_LETTER, fs_title=FS_PANEL_TITLE):
    """Left-aligned bold panel title adjacent to the panel letter, with optional italic subtitle.

    Nature-style: the letter sits at (x_letter, y) anchored bottom/right; the bold title sits at
    x=0 (axes left edge) on the same baseline; an optional italic grey subtitle sits just below.
    One helper, one offset/anchor/size, called from every per-axes figure for a shared grammar.
    """
    ax.text(x_letter, y, letter, transform=ax.transAxes, fontsize=fs_letter, fontweight="bold",
            va="bottom", ha="right", color=INK)
    ax.text(0.0, y, title, transform=ax.transAxes, fontsize=fs_title, fontweight="bold",
            va="bottom", ha="left", color=INK)
    if sub:
        ax.text(0.0, y - sub_dy, sub, transform=ax.transAxes, fontsize=6.8, color=SUBTITLE_GREY,
                va="bottom", ha="left", style="italic")


def style_legend(leg, ec=LEGEND_EC, alpha=LEGEND_ALPHA, lw=0.6):
    """Standardise a legend frame: one edge grey, one framealpha, fancybox off, white face."""
    fr = leg.get_frame()
    fr.set_edgecolor(ec); fr.set_linewidth(lw); fr.set_facecolor("white"); fr.set_alpha(alpha)
    return leg


def set_pub_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 350, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        # Arial/Helvetica if present, else Liberation Sans (Arial-metric-compatible) / Nimbus Sans
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType (editable text in vector output)
        "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.labelsize": 8, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.fontsize": 7, "legend.frameon": False, "legend.handlelength": 1.1,
        "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.titlepad": 6, "lines.linewidth": 1.2,
    })


def despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3, width=0.8)


def panel_label(ax, letter: str) -> None:
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="bottom", ha="right")
