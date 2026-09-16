#!/usr/bin/env python3
"""Figure 3, in the arrangement the submitted version used.

The submitted figure asks four questions in a 2x2: which surface markers are recovered, which
immune programs are recovered, whether program agreement and transcriptome agreement move together,
and whether the advantage is uniform across lineages. Reviewers raised none of them, so the
revision keeps the questions, the panel order and the visual grammar, and changes only the data:
the roster is the census this revision reports, and every value is read from the deposited
per-unit scores rather than from a stored copy.

The arrangement follows revision_codex/workflow/figures.py, which restored the submitted panels
after an earlier round had replaced them.

    python scripts/figure3_blindspot.py [--out-dir results/_paper]
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "results" / "_paper"

import sys
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.report.style import INK_BODY, INK_HEAD, INK_NOTE, MAIN_FS  # noqa: E402

# The shared ink ladder and type scale, so this figure matches Figures 1 and 2. Setting text.color
# alone is not enough: it never reaches tick labels, spines or tick marks, which print matplotlib
# black. figure_consistency.py fails the build on exactly that leak.
# PD-L1 was orange here and rose in Figures S4 and S5, so the same marker wore two colours
# across the set. The rose is the one the supplementary captions name.
NAVY, BLUE, ORANGE, GREY, GREEN = INK_HEAD, "#226ca0", "#B34E68", "#99a4ae", "#33866E"
INK = INK_NOTE
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": MAIN_FS["tick"],
    "axes.labelsize": MAIN_FS["axis"], "axes.titlesize": MAIN_FS["title"],
    "xtick.labelsize": MAIN_FS["tick"], "ytick.labelsize": MAIN_FS["tick"],
    "legend.fontsize": MAIN_FS["key"],
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True, "savefig.facecolor": "white",
    # This plate is authored AT the 174 mm live width, so the pad cannot grow: a 0.14 in pad
    # took it to 175.4 mm. The bottom clearance comes from lifting the footnote instead.
    "savefig.bbox": "tight", "savefig.pad_inches": 0.10,
    "text.color": INK_BODY, "axes.labelcolor": INK_BODY, "axes.titlecolor": INK_HEAD,
    "axes.edgecolor": INK_BODY, "xtick.color": INK_BODY, "ytick.color": INK_BODY,
    "xtick.labelcolor": INK_BODY, "ytick.labelcolor": INK_BODY, "legend.labelcolor": INK_NOTE,
})


def rd(name):
    with open(P / f"{name}.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fmt(x):
    return f"{x:.3f}".replace("-", "−")


def label(ax, letter, title):
    ax.set_title(title, loc="left", pad=13, color=NAVY)
    ax.text(-0.135, 1.05, letter, transform=ax.transAxes, fontsize=MAIN_FS["header"],
            fontweight="bold", color=NAVY)


def tidy(ax):
    ax.grid(axis="x", alpha=0.16)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- a. surface markers
def marker_panel(ax):
    d = [r for r in rd("c4_surface_marker_CIs") if int(float(r["held_frac_pct"])) == 50]
    d.sort(key=lambda r: float(r["obsDelta_mean"]))
    y = np.arange(len(d))
    labels = []
    for j, r in enumerate(d):
        obs, pred = float(r["obsDelta_mean"]), float(r["predDelta"])
        # A highlighted row used to draw FOUR markers: the generic blue diamond and grey dot,
        # then a larger coloured pair on top. At PD-L1, where observed and predicted nearly meet,
        # that overlay erased two thirds of the diamond it exists to draw attention to. One
        # diamond and one dot per row, in the row's own colour, larger where it is highlighted.
        hl = r["marker"] in ("CD279", "CD274")
        c = (GREEN if r["marker"] == "CD279" else ORANGE) if hl else BLUE
        ax.plot([obs, pred], [j, j], color="#bdc6ce", lw=1, zorder=1)
        ax.scatter(pred, j, s=40 if hl else 22, facecolor="none", edgecolor=c,
                   linewidths=1.4 if hl else 1.0, marker="D", zorder=2)
        ax.scatter(obs, j, s=20 if hl else 11, color=c if hl else GREY, zorder=3)
        name = r["alias"]
        if hl:
            # Every other row in this panel is labelled with its CD alias, and so are the same
            # two markers in Figures S4a and S5.
            name = "PD-1 (CD279)" if r["marker"] == "CD279" else "PD-L1 (CD274)"
        labels.append(name)
    ax.scatter([], [], s=11, color=GREY, label="Observed")
    ax.scatter([], [], s=22, facecolor="none", edgecolor=BLUE, linewidths=1.0,
               marker="D", label="Predicted")
    ax.axvline(0, color=GREY, ls="--", lw=0.8)
    ax.set_yticks(y, labels, fontsize=MAIN_FS["micro"])
    ax.set_ylim(-1, len(d))
    ax.set_xlabel("Normalized protein mean shift (50% holdout)")
    # the markers run bottom-left to top-right, so the upper left of the panel is empty;
    # below the axis the key collided with panel c's title
    ax.legend(loc="upper left", fontsize=MAIN_FS["key"], frameon=False,
              handletextpad=0.4, borderaxespad=0.5)
    tidy(ax)


# ---------------------------------------------------------------- b. program concordance
# The tag is the TASK, as it is in Figure S6, Table S12 and the census. This list used to mix
# the two vocabularies inside one panel -- three programs tagged with the dataset (OP3) and five
# with the task (T3) -- and named Treg_exhaustion differently from Figure S6.
NAMES = [
    ("type_I_IFN", "Type-I IFN (T5c)"),
    ("inflammatory_NFkB", "NF-κB (T5c)"),
    ("effector_lymphocyte", "Effector lymphocyte (T5c)"),
    ("TCR_activation", "TCR activation (T3)"),
    ("IL2_STAT5", "IL2–STAT5 (T3)"),
    ("proliferation", "Proliferation (T3)"),
    ("effector_cytokine", "Effector cytokine (T3)"),
    ("Treg_exhaustion", "Treg / exhaustion (T3)"),
]


# Which task each program is scored on, and therefore which summary carries it. Panel b lists
# three OP3 programs and five T3 programs; the OP3 macro file holds only the first three.
PROGRAM_TASK = {"type_I_IFN": "T5c", "inflammatory_NFkB": "T5c", "effector_lymphocyte": "T5c",
                "TCR_activation": "T3", "IL2_STAT5": "T3", "proliferation": "T3",
                "effector_cytokine": "T3", "Treg_exhaustion": "T3"}


def complete_macro():
    """Entries whose program correlation is defined on EVERY required unit of the split.

    The T5c entries come from the deposited OP3 macro, which already carries the estimable count.
    The five T3 programs had no source at all: they were looked up in that same OP3 file, found
    absent, and drawn NA. The NA is right today -- no model is estimable on all five T3 dataset
    arms -- but it was right by accident, and a future T3 result would have gone on being drawn NA.
    T3 completeness is computed here from immune_readout_summary, the file that records it.
    """
    out = collections.defaultdict(list)
    for r in rd("immune_readout_op3_macro"):
        if r["program_corr"] and r["n_estimable_lineages"] == r["n_lineages"]:
            out[r["program"]].append((r["model"], float(r["program_corr"]),
                                      float(r["pearson_delta"])))

    t3 = [r for r in rd("immune_readout_summary") if r["task_key"] == "T3"]
    units = {r["unit"] for r in t3}
    by = collections.defaultdict(dict)
    for r in t3:
        if r["program_corr"]:
            by[(r["program"], r["model"])][r["unit"]] = (float(r["program_corr"]),
                                                         float(r["pearson_delta"]))
    for (program, model), per_unit in by.items():
        if PROGRAM_TASK.get(program) != "T3" or set(per_unit) != units:
            continue                      # an incomplete macro is not a macro
        vals = list(per_unit.values())
        out[program].append((model, sum(v[0] for v in vals) / len(vals),
                             sum(v[1] for v in vals) / len(vals)))
    return out


def program_panel(ax):
    full = complete_macro()
    y = np.arange(len(NAMES))
    for j, (key, name) in enumerate(NAMES):
        entries = full.get(key, [])
        if not entries:
            ax.text(0.025, j, "NA — no complete estimable macro", va="center",
                    color=INK, fontsize=MAIN_FS["name"])
            continue
        model, value, _ = max(entries, key=lambda e: e[1])
        ax.barh(j, value, color=BLUE, height=0.55)
        ax.text(value + 0.025, j, f"{fmt(value)}  {model}", va="center", fontsize=MAIN_FS["name"])
    ax.set_yticks(y, [n for _, n in NAMES], fontsize=MAIN_FS["name"])
    ax.set(xlim=(0, 1.2), ylim=(len(NAMES) - 0.3, -0.7),
           xlabel="Program concordance (complete task macro)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    tidy(ax)


# ---------------------------------------------------------------- c. program vs transcriptome
def program_scatter(ax):
    entries = complete_macro().get("type_I_IFN", [])
    entries.sort(key=lambda e: -e[1])
    unc = {(r["task_key"], r["model"]): r for r in rd("census_uncertainty")}
    floor = float(unc[("T5c", "FP-ridge")]["binding_floor_score"])
    # The emphasis is derived, not chosen: the stars are the entries that clear this split's floor
    # in EVERY one of its four lineage units -- the census verdict "positive in all units" -- which
    # is the rule the caption states. Note this is NOT "above the dashed line": the line is the
    # mean of the four per-lineage floors, and scFoundation sits above it while falling below the
    # floor in B cells (0.284 vs 0.300). The previous rule was the largest Pearson-Delta plus a
    # hard-coded "scGPT", which encoded nothing a reader could name.
    lead_set = {m for (t, m), r in unc.items() if t == "T5c"
                and int(r["units_above_floor"].split("/")[0]) == int(r["n_units"])}
    for model, corr, pd_ in entries:
        lead = model in lead_set
        ax.scatter(corr, pd_, s=80 if lead else 38, marker="*" if lead else "o",
                   color=BLUE if lead else "#456c80", zorder=3)
    # Labels are placed by hand: the nine entries cluster between 0.51 and 0.83 on x, so a uniform
    # offset overlaps. Each goes into the empty side of its own point, and the two on the left edge
    # are right-aligned so the text runs away from the axis rather than back over the marker.
    OFF = {"FP-ridge": (8, 4, "left"), "scGPT": (8, -10, "left"),
           # CellOT (0.622) and scGen (0.708) are 0.086 apart on an axis 0.53 wide, so a
           # right-hand label on CellOT landed 112 px from scGen's marker and 154 px from its
           # own: the nearest label to the scGen dot was "CellOT". Moved above its own marker.
           "scFoundation": (9, -3, "left"), "CellOT": (0, 9, "center"),
           "scGen": (9, -3, "left"), "CellFlow": (9, -3, "left"),
           "scPRAM": (0, 9, "center"), "PerturbNet": (0, -11, "center"),
           # to the right of its dot, "PRnet" ended 0.078 in from the CellFlow marker against
           # 0.063 in from its own; to the left the panel is empty as far as the axis.
           # left of its dot the label was bisected by the y spine; below it the panel is empty
           "PRnet": (0, -11, "center")}
    for model, corr, pd_ in entries:
        dx, dy, ha = OFF.get(model, (8, 3, "left"))
        ax.annotate(model, (corr, pd_), xytext=(dx, dy), textcoords="offset points",
                    fontsize=MAIN_FS["name"], ha=ha, color=BLUE if model in lead_set else "#456c80")
    ax.axhline(floor, ls="--", color=GREY, lw=1)
    ax.set(xlim=(0.42, 0.95), ylim=(0, 0.48), xlabel="Type-I IFN program concordance",
           ylabel="Response-direction Pearson-\u0394")
    # inside the axes, clear of the y-axis label and of every point
    # the caption calls this line "that split's binding universal floor"; the panel called it a
    # transcriptome reference, a term that appears nowhere else in the package
    ax.text(0.432, floor + 0.014, "Binding universal floor " + fmt(floor), color=INK,
            fontsize=MAIN_FS["name"])
    tidy(ax)


# ---------------------------------------------------------------- d. per-lineage
ALIAS = {"Mono_CD14": "CD14 monocyte", "Mono_FCGR3A": "CD16 monocyte", "Mk": "Megakaryocyte",
         "CD4T": "CD4 T", "CD8T": "CD8 T", "T_cells": "T cells", "Mono": "Mono"}


def lineage_panel(ax):
    by = collections.defaultdict(dict)
    for r in rd("census_unit_scores"):
        by[(r["task_key"], r["unit"])][r["model"]] = float(r["pearson_delta"])
    rows, labels = [], []
    for task, model, tag in (("T1", "scGen", "Kang · "), ("T5c", "FP-ridge", "OP3 · ")):
        for (t, unit), d in sorted(by.items()):
            if t != task or model not in d:
                continue
            ref = max(d.get("cell-mean", -9.0), d.get("linear-PCA", -9.0))
            rows.append((task, d[model], ref))
            labels.append(tag + ALIAS.get(unit, unit))
    for j, (task, pred, ref) in enumerate(rows):
        ax.plot([ref, pred], [j, j], color="#b5bdc7", lw=1.5, zorder=1)
        # over the navy result the ring split it into fragments on the two rows where they
        # nearly coincide; the hollow ring is the larger marker, so it belongs underneath
        ax.scatter(ref, j, facecolor="none", edgecolor=GREY, linewidths=1.1, s=34, zorder=2)
        # One colour for the model, as in panel a: colouring by DATASET here made FP-ridge a
        # blue star in (c) and a teal dot in (d), scGen a slate dot in (c) and a blue dot in
        # (d), and teal mean PD-1 in (a) and FP-ridge-on-OP3 in (d). The two blocks are
        # already separated by the rule and named by their own y labels.
        ax.scatter(pred, j, color=BLUE, s=23, zorder=3)
    # Blue is scGen above the rule and FP-ridge below it; the caption said so and the panel did
    # not. Both blocks stop well short of the right edge (0.881 and 0.439 on a 0-1 axis), so the
    # name goes there, on the block's first row, in the colour it names.
    # On the block's first row the name landed on that row's own grey marker (Kang B's reference
    # is 0.829), so each sits in the blank strip that opens the block instead: above the first row
    # for scGen, between the dividing rule and the first OP3 row for FP-ridge.
    _split = sum(1 for r in rows if r[0] == "T1")
    # corner-anchored, "FP-ridge" sat level with OP3 B and read as annotating that one row;
    # each name is centred on the block it names instead
    for y, name in (((_split - 1) / 2, "scGen"), ((_split + len(rows) - 1) / 2, "FP-ridge")):
        ax.text(0.985, y, name, ha="right", va="center", fontsize=MAIN_FS["name"],
                color=BLUE, fontweight="bold")
    ax.set_yticks(range(len(rows)), labels, fontsize=MAIN_FS["name"])
    ax.set(ylim=(len(rows) - 0.3, -0.7), xlim=(0, 1),
           # "stronger of the two simple references" is accurate but too long for this panel: it
           # ran to the canvas edge with zero margin and its closing parenthesis was cut. Same
           # meaning, short enough to fit.
           xlabel="Pearson-Δ (grey: the stronger simple reference)")
    ax.axhline(sum(1 for r in rows if r[0] == "T1") - 0.5, color="#d6dce2", lw=0.8)
    tidy(ax)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(P))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 161 x 168 mm: the width Figure 2 and the previous Figure 3 both use, inside the 174 mm
    # live area. The arrangement this restores was drawn at 12 x 10 inches for the screen; at the
    # printed width the margins have to be retuned, not merely scaled, because panel a carries
    # twenty marker names and panel d twelve lineage labels.
    fig = plt.figure(figsize=(6.85, 6.7))
    # wspace 0.52 left panel b's long program names ("Regulatory / exhaustion (T3)") and panel d's
    # lineage labels drawn INSIDE the axes box of the panel to their left -- panel a's x=0.2
    # gridline ran through the first glyphs of "Effector lymphocyte (OP3)".
    gs = fig.add_gridspec(2, 2, left=0.145, right=0.955, bottom=0.095, top=0.875,
                          wspace=0.62, hspace=0.34)
    axes = [fig.add_subplot(gs[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    marker_panel(axes[0]); label(axes[0], "a", "Surface-marker shifts")
    program_panel(axes[1]); label(axes[1], "b", "Immune-program concordance")
    program_scatter(axes[2]); label(axes[2], "c", "Program and transcriptome agreement")
    lineage_panel(axes[3]); label(axes[3], "d", "Prediction varies by lineage")
    fig.text(0.015, 0.978, "Immune readouts retain distinct marker, program and lineage information",
             fontsize=MAIN_FS["title"], color=NAVY, fontweight="bold", va="top")
    # 0.012 of a 6.7 in plate put the footnote's descenders 0.063 in from the trim
    fig.text(0.155, 0.022,
             "Protein reference is fitted within modality. Undefined program correlations are not "
             "zero recovery.\nLocal point advantages do not establish task-wide or "
             "multiplicity-adjusted support.", fontsize=MAIN_FS["micro"], color=INK)

    for ext, kw in (("png", dict(dpi=240)), ("pdf", {}),
                    ("tiff", dict(dpi=600, pil_kwargs={"compression": "tiff_lzw"}))):
        fig.savefig(out / f"figure_immune_blindspot.{ext}", **kw)
    plt.close(fig)
    print(f"wrote {out}/figure_immune_blindspot.{{png,pdf,tiff}}")
    _macro = complete_macro()
    ent = _macro.get("type_I_IFN", [])
    _complete = sum(1 for v in _macro.values() if v)
    print(f"  panel b: {_complete} of {len(NAMES)} programs has a complete estimable macro")
    print(f"  panel c: {len(ent)} entries with a complete type-I IFN macro")


if __name__ == "__main__":
    main()
