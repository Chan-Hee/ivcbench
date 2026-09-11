#!/usr/bin/env python3
"""Current immune-readout figures from explicitly scoped, final-roster sources.

Produces Figure 3 and final Supplementary Figures S3, S4 and S5. No arbitrary success thresholds, false-zero
correlations, excluded CPA/scGen drug entries, or response-amplitude claims.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.transforms as mtransforms
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results/_paper"
NAVY, BLUE, GREY, ROSE = "#17324d", "#287fba", "#84909a", "#b34e68"
T3_MODELS = ["AttentionPert", "Biolord", "CellFlow", "GEARS", "PerturbNet", "scGPT"]
# CINEMA-OT is a census entry on this split, but no per-lineage program readout was deposited for
# it, so it cannot be scored here; both legends say so rather than drawing an empty row.
OP3_MODELS = [
    "FP-ridge",
    "Biolord",
    "CellFlow",
    "CellOT",
    "PRnet",
    "scPRAM",
    "scFoundation",
    "scGPT",
]
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
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlecolor": NAVY,
            "text.color": NAVY,
            "axes.unicode_minus": True,
        }
    )


def title(ax, letter, text):
    ax.set_title(f"{letter}  {text}", loc="left", fontweight="bold", pad=12)


def save(fig, stem, tiff=False):
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
    ax.set_xlabel("Protein shift (library-log units)")
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color=GREY,
                marker="o",
                lw=1,
                ms=3,
                label="Observed ± conditional 95% interval",
            ),
            Line2D(
                [],
                [],
                color=GREY,
                marker="D",
                mfc="white",
                lw=0,
                ms=3,
                label="Cell-mean floor prediction",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(-0.05, -0.31),
        frameon=False,
        fontsize=6.6,
    )
    ax.text(
        0,
        1.015,
        "Frangieh protein fit; 124 held KOs",
        transform=ax.transAxes,
        fontsize=6.6,
    )


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
    for i in range(len(OP3_MODELS)):
        for j in range(len(cols)):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, note[(i, j)], ha="center", va="center", color=NAVY, fontsize=7)
            else:
                label = f"{v:.2f}".replace("-", "\u2212")
                if (i, j) in note:
                    label += f"\n{note[(i, j)]}"
                ax.text(j, i, label, ha="center", va="center",
                        color="white" if abs(v) > 0.65 else NAVY, fontsize=7)
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
    )
    ax.tick_params(length=0)
    ax.text(
        0,
        -0.19,
        "Mean over the lineages where the score is defined;\nn/4 printed where fewer than four."
        "\nNA-O: constant observed target; NA-P: constant\nprediction; NA: both, in different lineages.\nNever plotted as zero."
        "\n\u2020 Adapted   * Diagnostic comparator",
        transform=ax.transAxes,
        fontsize=6.6,
        va="top",
    )


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
    ax.plot(
        x,
        frame.observed_program_score,
        "o-",
        color=NAVY,
        ms=4,
        label="Score of observed mean profile",
    )
    ax.plot(
        x,
        frame.observed_per_cell_score_mean,
        "s--",
        color=ROSE,
        ms=4,
        label="Mean observed per-cell score",
    )
    ax.set_xticks(
        x,
        frame.stratum.str.replace("perturbation=", "", regex=False),
        rotation=45,
        ha="right",
        fontsize=6.5,
    )
    ax.set_ylabel("TCR-activation rank score")
    ax.set_ylim(bottom=-0.002)
    ax.text(
        0,
        1.015,
        "Schmidt: all 7 held targets; observed data only",
        transform=ax.transAxes,
        fontsize=6.6,
    )
    ax.legend(
        loc="upper left", bbox_to_anchor=(-0.05, -0.31), fontsize=6, frameon=False
    )


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
    ax.annotate("calibrated", (1.0, 1.0), textcoords="offset points", xytext=(6, -9),
                fontsize=5.8, color=NAVY)

    plotted = d[d.slope.notna() & d.ratio.notna()]
    for _, r in plotted.iterrows():
        adapted = r["Role"] == "adapted"
        ax.scatter(r.ratio, r.slope, s=26, zorder=3,
                   facecolor="white" if adapted else FACE.get(r["Role"], BLUE),
                   edgecolor=FACE.get(r["Role"], BLUE), linewidths=0.9)
    # the three understating entries sit close together against the left spine, so their labels
    # go into the empty space to their right rather than off the axis
    for name, dx, dy, ha in [("linear-PCA", 7, 2, "left"), ("CINEMA-OT", 0, -11, "center"),
                             ("FP-ridge", 8, -1, "left"), ("PRnet", 6, 1, "left"),
                             ("cell-mean", 0, -11, "center")]:
        r = plotted[plotted.Entry == name]
        if len(r):
            ax.annotate(name, (r.ratio.iloc[0], r.slope.iloc[0]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=5.8, color=NAVY, ha=ha)

    ax.set_xlim(-0.15, 3.35)
    ax.set_ylim(-0.12, 1.12)
    ax.set_xlabel("predicted / observed response norm", fontsize=6.6)
    ax.set_ylabel("calibration slope", fontsize=6.6)
    ax.tick_params(labelsize=6)
    zero = d[d.slope.isna() & (d.ratio == 0)]
    note = ("Six conditioned predictors overstate the response 1.5 to 3.0 times with slopes at or near\n"
            "zero; the three that understate it keep slopes well above zero."
            + (" scPRAM predicts no shift." if len(zero) else ""))
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BLUE, ms=5, label="conditioned"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=BLUE, ms=5, label="adapted interface"),
        Line2D([], [], marker="o", ls="", mfc=ROSE, mec=ROSE, ms=5, label="diagnostic"),
        Line2D([], [], marker="o", ls="", mfc=GREY, mec=GREY, ms=5, label="floor member"),
    ], loc="upper left", bbox_to_anchor=(-0.02, -0.20), ncol=2, fontsize=6,
        frameon=False, handletextpad=0.3, columnspacing=1.1, labelspacing=0.25)
    ax.text(0.0, -0.42, note, transform=ax.transAxes, va="top", fontsize=5.8, color=GREY)


def figure3(summary, macro):
    """Compose the main readout argument; the full protein panel appears once."""
    fig, axes = plt.subplots(2, 2, figsize=(6.33, 7.07))  # 174 mm live area
    fig.subplots_adjust(
        left=0.14, right=0.98, top=0.95, bottom=0.13, wspace=0.85, hspace=0.78
    )
    aggregation_panel(axes[0, 0])
    title(axes[0, 0], "a", "Aggregation changes the target")
    op3_matrix(axes[0, 1], summary)
    title(axes[0, 1], "b", "OP3 program concordance")
    magnitude_panel(axes[1, 0])
    title(axes[1, 0], "c", "Direction is not magnitude")
    protein_panel(axes[1, 1])
    title(axes[1, 1], "d", "Checkpoint assay context")
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
            axes[0].plot(
                row.pearson_delta, i, "o", color=colors[row.unit], ms=4, alpha=0.8
            )
            if row.correlation_status == "estimable (descriptive)":
                axes[1].plot(
                    row.program_corr, i, "o", color=colors[row.unit], ms=4, alpha=0.8
                )
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
    # the ticks used to sit at ymin=0.94, which is the top data row; give them their own band
    axes[0].set_ylim(len(OP3_MODELS) - 0.4, -1.35)
    for unit, val in per.items():
        axes[0].plot([val, val], [-1.25, -0.95], color=colors.get(unit, GREY), lw=1.2)
    axes[0].text(0.99, 1.02, "ticks above the panel: per-lineage binding floor; dashed: their mean",
                 transform=axes[0].transAxes, ha="right", fontsize=6, color=GREY)
    axes[0].set_xlabel("Gene-response pattern correlation (Pearson-Δ)")
    axes[1].set_xlabel("Type-I IFN program correlation across compounds")
    axes[1].set_xlim(-1.02, 1.02)
    title(axes[0], "a", "Expression-pattern fidelity")
    title(axes[1], "b", "Program concordance")
    fig.legend(
        handles=[
            Line2D([], [], color=c, marker="o", ls="", label=k.replace("_", " "))
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
        "Chen checkpoint readouts depend on normalization", fontsize=10, color=NAVY
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
