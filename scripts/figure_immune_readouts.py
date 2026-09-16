#!/usr/bin/env python3
"""Current immune-readout figures from explicitly scoped, final-roster sources.

Produces Figure 3 and final Supplementary Figures S3, S4 and S5. No arbitrary success thresholds, false-zero
correlations, excluded CPA/scGen drug entries, or response-amplitude claims.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.transforms as mtransforms
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results/_paper"
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.report.style import INK_BODY, INK_HEAD, INK_NOTE, MAIN_FS  # noqa: E402

# data-mark colours only (points, lines, intervals); every piece of text takes the shared ink
# ladder from ivcbench.report.style so Figures 2 and 3 print the inks Figure 1 prints
NAVY, BLUE, GREY, ROSE = "#17324d", "#287fba", "#84909a", "#b34e68"
# The rosters are DERIVED from the readout table, never typed here. A hand-typed roster is how
# Figure 2 once plotted 24 of 35 census cells without anything failing: the list simply stopped
# being the census. immune_readout_audit.py is itself census-driven (it reads census_unit_scores.csv),
# so a model that entered the census and has a per-lineage readout appears here automatically, and
# roster_note() states in the caption which census cells could not be drawn and why.
# CINEMA-OT is a census entry on these splits with no per-lineage program readout deposited, so it
# cannot be scored here; it is named in that note rather than drawn as an empty row.
_ORDER = ["FP-ridge", "linear-shift-KOemb", "Biolord", "CPA", "CellFlow", "CellOT", "GEARS",
          "PRnet", "PertAdapt", "PerturbNet", "STATE", "AttentionPert", "scGen", "scPRAM",
          "scFoundation", "scGPT"]
_REFERENCES = {"cell-mean", "linear-PCA", "ctrl-pred", "donor-shift"}


def _roster(summary, units, task_key):
    """Models this figure draws for a task: a CENSUS cell on that task WITH a deposited readout.

    Both halves are needed. Without the readout half the plate has empty rows; without the census
    half it draws a cell the panel does not report -- biolord has an OP3 readout but is a T5u
    census entry, not a T5c one, and the hand-typed roster drew it under the T5c heading.
    """
    have = set(summary[summary.task_key == task_key].model) - _REFERENCES
    have &= set(units[units.task_key == task_key].model)
    known = [m for m in _ORDER if m in have]
    return known + sorted(have - set(known))      # anything new still gets drawn, at the end


T3_MODELS: list[str] = []
OP3_MODELS: list[str] = []
PROGRAMS = [
    "TCR_activation",
    "IL2_STAT5",
    "proliferation",
    "effector_cytokine",
    "Treg_exhaustion",
]
LABELS = [
    "TCR activation",
    "IL2\u2013STAT5",
    "Proliferation",
    "Effector cytokine",
    "Treg / exhaustion",
]
DATASETS = ["shifrut", "schmidt", "mccutcheon_CRISPRi", "mccutcheon_CRISPRa", "chen"]
DS_LABELS = [
    "Shifrut (2)",
    "Schmidt (7)",
    "McCutcheon i (2)",
    "McCutcheon a (2)",
    "Chen (30)",
]


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": MAIN_FS["tick"],
            "axes.labelsize": MAIN_FS["axis"],
            "axes.titlesize": MAIN_FS["title"],
            "xtick.labelsize": MAIN_FS["tick"],
            "ytick.labelsize": MAIN_FS["tick"],
            "legend.fontsize": MAIN_FS["key"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # the shared text-ink ladder. text.color never reaches tick labels, spines or tick
            # marks (they printed matplotlib black before), so those are set explicitly.
            "text.color": INK_BODY,
            "axes.labelcolor": INK_BODY,
            "axes.titlecolor": INK_HEAD,
            "axes.edgecolor": INK_BODY,
            "xtick.color": INK_BODY,
            "ytick.color": INK_BODY,
            "xtick.labelcolor": INK_BODY,
            "ytick.labelcolor": INK_BODY,
            "legend.labelcolor": INK_NOTE,
            "axes.unicode_minus": True,
        }
    )


# panel-header geometry (inches), matching Figure 2: a standalone bold letter on the panel's
# outer left edge, the bold title 0.15 in to its right, one grey subtitle line under the title
TITLE_DX = 0.15  # title left edge, right of the letter's left edge
SUB_DY = 0.09  # subtitle baseline above the axes top
TITLE_PITCH = 0.205  # title baseline above the subtitle baseline (Figure 2's pitch)
TITLE_DY_NOSUB = 0.17  # title baseline above the axes top when there is no subtitle


def title(ax, letter, text, sub=None):
    """Queue a panel header; it is drawn by _place_titles once the tick labels exist, so the
    letters of one column all sit on the column's outermost left edge (the widest tick-label
    set decides it) and share one x, as the letters of Figure 2 do."""
    ax.figure.__dict__.setdefault("_panel_titles", []).append((ax, letter, text, sub))


def _place_titles(fig):
    items = fig.__dict__.pop("_panel_titles", [])
    if not items:
        return
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    to_in = fig.dpi_scale_trans.inverted()
    left = {ax: ax.get_tightbbox(rend).transformed(to_in).x0 for ax, *_ in items}
    cols: dict = {}
    for ax in left:
        cols.setdefault(round(ax.get_position().x0, 3), []).append(ax)
    for axs in cols.values():
        x = min(left[a] for a in axs)
        for a in axs:
            left[a] = x
    for ax, letter, text, sub in items:
        base = mtransforms.blended_transform_factory(fig.dpi_scale_trans, ax.transAxes)
        y_title = SUB_DY + TITLE_PITCH if sub else TITLE_DY_NOSUB
        tr = mtransforms.offset_copy(base, fig=fig, x=0, y=y_title, units="inches")
        kw = dict(ha="left", va="baseline", color=INK_HEAD, fontweight="bold")
        fig.text(left[ax], 1.0, letter, transform=tr, fontsize=MAIN_FS["letter"], **kw)
        fig.text(left[ax] + TITLE_DX, 1.0, text, transform=tr, fontsize=MAIN_FS["title"], **kw)
        if sub:
            tr = mtransforms.offset_copy(base, fig=fig, x=0, y=SUB_DY, units="inches")
            fig.text(left[ax] + TITLE_DX, 1.0, sub, transform=tr, fontsize=MAIN_FS["subtitle"],
                     ha="left", va="baseline", color=INK_NOTE)


def _keys_below(fig, rows, gap=0.10):
    """One key per panel, under its x-axis, its left edge on the axes' left edge; the keys of
    one panel row share one top edge (the row's deepest tick label or axis label decides it).
    A str in place of handles is drawn as a plain text key."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    to_in = fig.dpi_scale_trans.inverted()
    fh = fig.get_size_inches()[1]
    for row in rows:
        bottom = min(ax.get_tightbbox(rend).transformed(to_in).y0 for ax, *_ in row)
        y = (bottom - gap) / fh
        for ax, handles, kw in row:
            x = ax.get_position().x0
            if isinstance(handles, str):
                fig.text(x, y, handles, fontsize=MAIN_FS["key"], color=INK_NOTE, ha="left",
                         va="top")
            else:
                ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, y),
                          bbox_transform=fig.transFigure, frameon=False, borderpad=0,
                          borderaxespad=0, handletextpad=0.5, labelspacing=0.35, **kw)


def _assert_layout(fig):
    """Geometry proof: no text block leaves the canvas, and no panel's header, axes (with its
    tick labels) or key overlaps another panel's."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fw, fh = fig.get_size_inches()
    to_in = fig.dpi_scale_trans.inverted()
    blocks = []  # (name, bbox, owner); an axes' tight bbox already contains its own legend
    for t in fig.texts:
        bb = t.get_window_extent(rend).transformed(to_in)
        assert bb.x0 > -0.02 and bb.y0 > -0.02 and bb.x1 < fw + 0.55 and bb.y1 < fh + 0.55, (
            f"text {t.get_text()[:30]!r} leaves the canvas: {bb}")
        blocks.append((t.get_text()[:30], bb, None))
    for ax in fig.axes:
        blocks.append(("axes", ax.get_tightbbox(rend).transformed(to_in), ax))
        if ax.get_legend() is not None:
            blocks.append(
                ("legend", ax.get_legend().get_window_extent(rend).transformed(to_in), ax)
            )
    for i, (na, a, oa) in enumerate(blocks):
        for nb, b, ob in blocks[i + 1:]:
            if oa is not None and oa is ob:
                continue
            hit = a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1
            assert not hit, f"layout collision: {na!r} overlaps {nb!r}"


def save(fig, stem, tiff=False):
    _place_titles(fig)
    fig.savefig(
        PAPER / (stem + ".png"), dpi=350, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(PAPER / (stem + ".pdf"), bbox_inches="tight", facecolor="white")
    if tiff:
        fig.savefig(
            PAPER / (stem + ".tiff"),
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    plt.close(fig)
    print(stem, flush=True)


def protein_panel(ax):
    frame = pd.read_csv(PAPER / "c4_surface_marker_CIs.csv")
    frame = frame[frame.held_frac_pct == 50].sort_values("obsDelta_mean")
    for i, row in enumerate(frame.itertuples()):
        color = (
            ROSE if row.marker == "CD274" else BLUE if row.marker == "CD279" else GREY
        )
        # the observed point sat under the open prediction diamond in 13 of 20 rows, so the
        # interval appeared to belong to the constant prediction; offset them onto two half-rows
        ax.plot([row.ci_lo, row.ci_hi], [i - 0.16, i - 0.16], color=color, lw=1.2, zorder=2)
        ax.plot(row.obsDelta_mean, i - 0.16, "o", color=color, ms=3.2, zorder=4)
        ax.plot(row.predDelta, i + 0.18, "D", mfc="white", mec=color, ms=3, zorder=3)
    ax.set_yticks(
        range(len(frame)),
        [
            str(a).split(" (")[0]
            for a, m in zip(frame.alias, frame.marker)
        ],
    )
    ax.axvline(0, color=GREY, lw=0.6)
    ax.set_xlabel("protein shift (library-log units)")
    return [  # the key, drawn under the panel by _keys_below
        Line2D([], [], color=GREY, marker="o", lw=1, ms=3,
               label="observed ± conditional 95% interval"),
        Line2D([], [], color=GREY, marker="D", mfc="white", lw=0, ms=3,
               label="cell-mean floor prediction"),
    ]


def op3_matrix(ax, frame):
    """Program concordance per model, scored on the lineages where it is defined.

    The earlier version required all four coarse lineages and blanked a cell otherwise, so cells
    that Table S7 reports as estimable in two of four were drawn as "NA-O: constant observed
    target" - a reason that was not theirs. The mean is now taken over the estimable lineages, the
    count is printed where it is below four, and a cell is marked NA only when no lineage is
    estimable, with the label taken from that cell's own reasons.
    """
    cols = ["type_I_IFN", "inflammatory_NFkB", "effector_lymphocyte"]
    sub = frame[(frame.task_key == "T5c") & frame.program.isin(cols)]
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#eeeeee")
    matrix = np.full((len(OP3_MODELS), len(cols)), np.nan)
    note = {}
    for i, model in enumerate(OP3_MODELS):
        for j, prog in enumerate(cols):
            cell = sub[(sub.model == model) & (sub.program == prog)]
            est = cell[cell.correlation_status == "estimable (descriptive)"]
            if len(est):
                matrix[i, j] = est.program_corr.mean()
                if len(est) < len(cell):
                    note[(i, j)] = f"{len(est)}/{len(cell)}"
            else:
                why = set(cell.correlation_status)
                note[(i, j)] = ("NA-O" if why == {"constant observed rank-program score"}
                                else "NA-P" if why == {"constant predicted rank-program score"}
                                else "NA")
    ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    fs = MAIN_FS["tick"]  # in-cell values, the size Figure 2 prints its cells at
    for i in range(len(OP3_MODELS)):
        for j in range(len(cols)):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, note[(i, j)], ha="center", va="center", color=INK_BODY,
                        fontsize=fs)
            else:
                label = f"{v:.2f}".replace("-", "\u2212")
                if (i, j) in note:
                    label += f"\n{note[(i, j)]}"
                ax.text(j, i, label, ha="center", va="center",
                        color="white" if abs(v) > 0.65 else INK_BODY, fontsize=fs)
    ax.set_xticks(range(len(cols)), ["Type-I IFN", "NF-\u03baB", "Effector\n(lymphocyte)"])
    ax.set_yticks(
        range(len(OP3_MODELS)),
        [
            x.replace("Biolord", "biolord")
            + (
                " \u2020"
                if x in ["scGPT", "scFoundation"]
                else " *" if x in ["FP-ridge", "CINEMA-OT"] else ""
            )
            for x in OP3_MODELS
        ],
        fontsize=MAIN_FS["name"],  # the model roster, at the size Figure 2 prints its rows
    )
    ax.tick_params(length=0)
    # The six-line footnote that used to sit here (mean over estimable lineages, the NA codes,
    # "never plotted as zero") is carried by the subtitle and the figure legend; only the
    # symbol key for the row labels stays on the plate.
    return "\u2020 adapted interface     * diagnostic comparator"


def t3_observability(ax, summary):
    """Show estimable comparisons; a constant target is not a zero model score."""
    selected = summary[(summary.task_key == "T3") & summary.model.isin(T3_MODELS)]
    selected = selected.assign(
        estimable=selected.correlation_status == "estimable (descriptive)"
    )
    counts = (
        selected.groupby(["unit", "program"])
        .estimable.sum()
        .unstack()
        .loc[DATASETS, PROGRAMS]
    )
    target_constant = (
        selected.assign(
            constant=selected.correlation_status
            == "constant observed rank-program score"
        )
        .groupby(["unit", "program"])
        .constant.all()
        .unstack()
        .loc[DATASETS, PROGRAMS]
    )
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eeeeee")
    ax.imshow(
        np.ma.masked_where(target_constant, counts),
        cmap=cmap,
        vmin=0,
        vmax=len(T3_MODELS),
        aspect="auto",
    )
    for i in range(5):
        for j in range(5):
            value = int(counts.iloc[i, j])
            label = (
                "constant"
                if target_constant.iloc[i, j]
                else f"{value}/{len(T3_MODELS)}"
            )
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="white" if value > 4 else NAVY,
                fontsize=5.6,
            )
    # five program names do not fit side by side in a half-width panel
    ax.set_xticks(range(5), LABELS, fontsize=6, rotation=30, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(range(5), DS_LABELS, fontsize=6.5)
    ax.tick_params(length=0)
    ax.text(
        0,
        -0.25,
        "Rows: dataset-arms (held-gene counts in parentheses).\n22/25 targets are"
        " constant; not evidence of model failure.",
        transform=ax.transAxes,
        fontsize=6.6,
        va="top",
    )


def aggregation_panel(ax):
    scores = pd.read_csv(PAPER / "immune_readout_per_stratum.csv")
    frame = scores[
        (scores.task_key == "T3")
        & (scores.unit == "schmidt")
        & (scores.program == "TCR_activation")
        & (scores.model == "cell-mean")
    ].sort_values("stratum")
    x = np.arange(len(frame))
    (mean_profile,) = ax.plot(
        x,
        frame.observed_program_score,
        "o-",
        color=NAVY,
        ms=4,
        label="rank score of the observed mean profile",  # the legend's wording
    )
    (per_cell,) = ax.plot(
        x,
        frame.observed_per_cell_score_mean,
        "s--",
        color=ROSE,
        ms=4,
        label="mean of the per-cell scores",
    )
    ax.set_xticks(
        x,
        frame.stratum.str.replace("perturbation=", "", regex=False),
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("TCR-activation rank score")
    ax.set_ylim(bottom=-0.002)
    return [mean_profile, per_cell]


def magnitude_panel(ax):
    """Direction is not magnitude.

    The estimability grid this panel replaced spent a fifth of the figure saying one number
    (22 of 25 targets constant), which the text states and Table S12 and Figure S6 carry in full.
    The calibration result had no figure at all, although it is the sharpest thing the compound
    axis shows: a model can point the right way and still be three times too large. Pearson-Delta
    cannot see that, because it is invariant to the scale of the predicted shift.
    """
    src = PAPER / "supplementary_tables/Supplementary_Table_S22.csv"
    d = pd.read_csv(src)
    num = lambda c: pd.to_numeric(
        d[c].astype(str).str.replace("\u2212", "-", regex=False), errors="coerce")
    d["ratio"], d["slope"] = num("Predicted / observed response norm"), num("Calibration slope")

    FACE = {"native": BLUE, "adapted": BLUE, "diagnostic": ROSE, "floor member": GREY}
    ax.axvline(1.0, color=GREY, lw=0.7, ls=":", zorder=1)
    ax.axhline(1.0, color=GREY, lw=0.7, ls=":", zorder=1)
    ax.plot([1.0], [1.0], marker="+", ms=7, mew=1.2, color=NAVY, zorder=4)
    fs = MAIN_FS["tick"]  # in-panel annotations
    ax.annotate("calibrated", (1.0, 1.0), textcoords="offset points", xytext=(6, -9),
                fontsize=fs, color=INK_BODY)

    plotted = d[d.slope.notna() & d.ratio.notna()]
    for _, r in plotted.iterrows():
        adapted = r["Role"] == "adapted"
        ax.scatter(r.ratio, r.slope, s=26, zorder=3,
                   facecolor="white" if adapted else FACE.get(r["Role"], BLUE),
                   edgecolor=FACE.get(r["Role"], BLUE), linewidths=0.9)
    # The understating entries sit close together against the left spine, so their labels go into
    # the empty space around them rather than off the axis; a white halo keeps them legible where
    # they cross the dotted guides. These four are the point of the panel: they are the only
    # entries below x = 1, and three of them are exactly the three that clear the floor on
    # response direction. The overstating band near y = 0 is left unlabelled -- seven entries
    # between 1.8 and 3.0 that all read the same way -- with its two ends named.
    halo = dict(boxstyle="square,pad=0.12", fc="white", ec="none")
    for name, dx, dy, ha in [("linear-PCA", 0, 7, "center"), ("scFoundation", 9, -2, "left"),
                             ("FP-ridge", 0, -11, "center"), ("scGPT", 9, -2, "left"),
                             ("PRnet", 0, 9, "center"),
                             ("cell-mean", 0, -11, "center")]:
        r = plotted[plotted.Entry == name]
        if len(r):
            ax.annotate(name, (r.ratio.iloc[0], r.slope.iloc[0]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=fs, color=INK_BODY, ha=ha,
                        bbox=halo, zorder=5)

    ax.set_xlim(-0.15, 3.35)
    ax.set_ylim(-0.12, 1.12)
    ax.set_xlabel("predicted / observed response norm")
    ax.set_ylabel("calibration slope")
    # The reading of the panel (eight conditioned predictors overstate the response 1.8-3.0x with
    # slopes near zero and all fall short on direction; the entries that understate it keep slopes
    # well above zero and include all three that clear the floor) is prose for the Results and the
    # legend, not for the plate.
    return [  # the colour key, drawn under the panel by _keys_below
        Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BLUE, ms=5, label="conditioned"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=BLUE, ms=5, label="adapted interface"),
        Line2D([], [], marker="o", ls="", mfc=ROSE, mec=ROSE, ms=5, label="diagnostic"),
        Line2D([], [], marker="o", ls="", mfc=GREY, mec=GREY, ms=5, label="floor member"),
    ]


def figure3(summary, macro):
    """Compose the main readout argument; the full protein panel appears once."""
    fig, axes = plt.subplots(2, 2, figsize=(6.33, 7.07))  # 174 mm live area
    fig.subplots_adjust(
        left=0.14, right=0.98, top=0.95, bottom=0.13, wspace=0.72, hspace=0.62
    )
    (a, b), (c, d) = axes
    key_a = aggregation_panel(a)
    title(a, "a", "Aggregation changes the target",
          "Schmidt: all 7 held targets; observed data only")
    key_b = op3_matrix(b, summary)
    title(b, "b", "OP3 program concordance", "OP3 held cell types; n/4 = estimable lineages")
    key_c = magnitude_panel(c)
    title(c, "c", "Direction is not magnitude", "OP3 held cell types; four coarse lineages")
    key_d = protein_panel(d)
    title(d, "d", "Checkpoint assay context", "Frangieh protein fit; 124 held KOs")
    _place_titles(fig)
    _keys_below(fig, [
        [(a, key_a, {}), (b, key_b, {})],
        [(c, key_c, dict(ncol=2, columnspacing=1.2)), (d, key_d, {})],
    ])
    _assert_layout(fig)
    save(fig, "figure_immune_blindspot", tiff=True)


def figure_s3(summary, units):
    """Supplementary Figure S3: matched OP3 lineage scores on two readouts."""
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.9))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.19, top=0.88, wspace=0.65)
    frame = summary[
        (summary.task_key == "T5c")
        & (summary.program == "type_I_IFN")
        & summary.model.isin(OP3_MODELS)
    ]
    colors = dict(B="#287fba", Mono="#b34e68", NK="#33866e", T_cells="#9b7435")
    for i, model in enumerate(OP3_MODELS):
        sub = frame[frame.model == model]
        for row in sub.itertuples():
            # Solid markers hid each other where two lineages land close together: CellOT's B
            # (0.757) sat completely under its T cells (0.759) in panel b, so the row showed
            # three markers for four estimable lineages while the caption counts them. Hollow
            # markers keep the exact positions and let both rings be seen.
            axes[0].plot(row.pearson_delta, i, "o", ms=4.6, mfc="none",
                         mec=colors[row.unit], mew=1.1, alpha=0.95)
            if row.correlation_status == "estimable (descriptive)":
                axes[1].plot(row.program_corr, i, "o", ms=4.6, mfc="none",
                             mec=colors[row.unit], mew=1.1, alpha=0.95)
        if not (sub.correlation_status == "estimable (descriptive)").any():
            axes[1].axhspan(i - 0.42, i + 0.42, color="#EFEFEF", zorder=0, lw=0)
            axes[1].text(
                0.5,
                i,
                "not estimable",
                transform=mtransforms.blended_transform_factory(
                    axes[1].transAxes, axes[1].transData
                ),
                ha="center",
                va="center",
                color=GREY,
                fontsize=6,
                style="italic",
            )
    for ax in axes:
        ax.set_yticks(range(len(OP3_MODELS)),
                      [m.replace("Biolord", "biolord") for m in OP3_MODELS])
        ax.invert_yaxis()
    t5c = units[units.task_key == "T5c"]
    q = t5c.groupby("model").pearson_delta.mean()
    floor = max(q["cell-mean"], q["linear-PCA"])
    # the binding floor is per lineage; one line at their mean puts some lineages on the wrong side
    per = t5c[t5c.model.isin(["cell-mean", "linear-PCA"])].groupby("unit").pearson_delta.max()
    axes[0].axvline(floor, color=GREY, ls="--", lw=1)
    # the ticks used to sit at ymin=0.94, which is the top data row; give them their own band.
    # BOTH panels take these limits: extending only panel a moved its eleven rows down relative
    # to panel b by up to 0.17 in on the page, so a reader tracking one model across the two
    # panels was reading two different heights for the same row.
    for _ax in axes:
        _ax.set_ylim(len(OP3_MODELS) - 0.4, -1.35)
    for unit, val in per.items():
        axes[0].plot([val, val], [-1.25, -0.95], color=colors.get(unit, GREY), lw=1.2)
    # right-aligned to the panel: left-aligned it began outside the axes, left even of the panel
    # letter, and was the leftmost ink in the whole plate
    axes[0].text(1.0, 1.02, "ticks above the panel: per-lineage binding floor; dashed: their mean",
                 transform=axes[0].transAxes, ha="right", fontsize=6, color=GREY)
    axes[0].set_xlabel("Gene-response pattern correlation (Pearson-Δ)")
    axes[1].set_xlabel("Type-I IFN program correlation\nacross compounds")
    axes[1].set_xlim(-1.02, 1.02)
    title(axes[0], "a", "Expression-pattern fidelity")
    title(axes[1], "b", "Program concordance")
    fig.legend(
        handles=[
            # hollow, like the markers in the panels
            Line2D([], [], marker="o", ls="", mfc="none", mec=c, mew=1.1, ms=6,
                   label=k.replace("_", " "))
            for k, c in colors.items()
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7,
    )
    save(fig, "figure_cellcontext")


def figure_s4():
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    fig.subplots_adjust(left=0.28, right=0.95, top=0.82, bottom=0.26)
    frame = pd.read_csv(PAPER / "c4_rna_vs_surface_decoupling.csv")
    frame = frame[frame.surface_marker == "CD274"]
    labels, values = [], []
    for row in frame.itertuples():
        for modality, prefix in [("RNA", "rna"), ("Surface", "surf")]:
            mean = getattr(row, prefix + "_mean")
            lo, hi = getattr(row, prefix + "_ci_lo"), getattr(row, prefix + "_ci_hi")
            labels.append(f"{modality}, {row.held_frac_pct}%")
            values.append((mean, lo, hi, BLUE if modality == "RNA" else ROSE))
    for i, (mean, lo, hi, color) in enumerate(values):
        ax.plot([lo, hi], [i, i], color=color, lw=2)
        ax.plot(mean, i, "o", color=color, ms=5)
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.axvline(0, color=GREY, lw=0.7)
    ax.set_xlabel("Observed CD274 shift (modality-specific units)")
    ax.set_title("Frangieh: observed RNA and surface CD274 shifts", fontsize=10, pad=14)
    fig.text(
        0.5,
        0.03,
        "Separate modality fits; intervals condition on the shared"
        " control.\nOverlapping KO holdouts are not independent replications.",
        fontsize=7,
        ha="center",
    )
    save(fig, "figS_c4_pdl1_assay_power")


def figure_s5():
    frame = pd.read_csv(PAPER / "chen_checkpoint_normalization.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.82, bottom=0.2, wspace=0.2)
    for ax, norm, letter in zip(axes, ["library-log", "centered log1p"], ["a", "b"]):
        sub = frame[frame.normalization == norm].sort_values(
            ["marker", "held_fraction"]
        )
        for i, row in enumerate(sub.itertuples()):
            color = BLUE if row.marker == "PD-1" else ROSE
            ax.plot([row.ci_lo, row.ci_hi], [i, i], color=color, lw=2)
            ax.plot(row.mean, i, "o", color=color, ms=5)
        ax.set_yticks(
            range(4), [f"{r.marker}, {r.held_fraction}%" for r in sub.itertuples()]
        )
        ax.axvline(0, color=GREY, lw=0.7)
        ax.set_xlabel("Observed marker shift")
        title(ax, letter, norm)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Chen checkpoint readouts depend on normalization", fontsize=10, color=INK_HEAD
    )
    save(fig, "figS_chen_checkpoint_replication")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=["3", "S3", "S4", "S5"],
        default=["3", "S3", "S4", "S5"],
    )
    args = parser.parse_args(argv)
    style()
    if any(f in args.figures for f in ("3", "S3")):
        summary = pd.read_csv(PAPER / "immune_readout_summary.csv")
        macro = pd.read_csv(PAPER / "immune_readout_op3_macro.csv")
        units = pd.read_csv(PAPER / "census_unit_scores.csv")
        global T3_MODELS, OP3_MODELS
        T3_MODELS = _roster(summary, units, "T3")
        OP3_MODELS = _roster(summary, units, "T5c")
        # Say out loud which census cells this figure cannot draw. Silence here is what let
        # Figure 2 ship missing eleven of thirty-five cells; the caption has to carry this.
        for tk, drawn in (("T3", T3_MODELS), ("T5c", OP3_MODELS)):
            cen = set(units[units.task_key == tk].model) - _REFERENCES
            missing = sorted(cen - set(drawn))
            print(f"[Figure 3] {tk}: drawing {len(drawn)} of {len(cen)} census cells"
                  + (f"; NO per-lineage readout deposited for {', '.join(missing)}"
                     if missing else "; every census cell drawn"))
    for figure in args.figures:
        if figure == "3":
            figure3(summary, macro)
        elif figure == "S3":
            figure_s3(summary, units)
        elif figure == "S4":
            figure_s4()
        elif figure == "S5":
            figure_s5()
        else:
            raise ValueError(figure)


if __name__ == "__main__":
    main()
