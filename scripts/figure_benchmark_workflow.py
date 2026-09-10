#!/usr/bin/env python3
"""Figure 1: the executed workflow, including preprocessing and inference limits."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/_paper"
NAVY, GREY, LINE = "#17324d", "#637180", "#dbe3ea"


def box(ax, x, y, width, height, color="#f7f9fb", edge=LINE):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            fc=color,
            ec=edge,
            lw=0.8,
        )
    )


def main():
    census = pd.read_csv(OUT / "cross_cluster_headline.csv")
    counts = census.status.value_counts()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "text.color": NAVY,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.99)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(
        0.015,
        0.985,
        "Immune perturbation prediction: from task to evidence",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.015,
        0.929,
        "Five tasks · three intervention classes · task-specific information"
        " boundaries",
        fontsize=9,
        color=GREY,
    )
    tasks = [
        ("T1", "Kang", "IFN-β PBMCs", "Cell-context transfer", "#3274a3"),
        ("T2", "Soskic", "CD4 T-cell activation", "Donor transfer", "#3274a3"),
        (
            "T3",
            "Primary-T CRISPR",
            "Four studies / five arms",
            "Unseen genes",
            "#43806b",
        ),
        ("T4", "Frangieh", "Melanoma Perturb-CITE", "Unseen KOs; RNA", "#43806b"),
        ("T5", "OP3", "PBMC compounds", "Cell context / compounds", "#b38532"),
    ]
    for i, (task, name, system, split, color) in enumerate(tasks):
        x = 0.02 + 0.197 * i
        box(ax, x, 0.725, 0.175, 0.161)
        ax.plot([x + 0.006, x + 0.006], [0.751, 0.868], color=color, lw=2.5)
        ax.text(x + 0.017, 0.857, f"{task}  {name}", fontsize=8.6, fontweight="bold")
        ax.text(x + 0.017, 0.814, system, fontsize=7.3, color=GREY)
        ax.text(x + 0.017, 0.766, split, fontsize=7.1)
    ax.text(
        0.5,
        0.681,
        "Shared input preparation: QC, feature selection and task-specific"
        " normalization",
        ha="center",
        fontsize=8,
        color=GREY,
    )
    ax.annotate(
        "",
        (0.5, 0.618),
        (0.5, 0.66),
        arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1),
    )
    cards = [
        (
            0.02,
            0.284,
            "1  Define the held-out unit",
            (
                "Cell context     T1; T5c\nDonor                T2\nGene / KO        "
                " T3; T4\nCompound        T5u"
            ),
            (
                "Matched control cells may be used\nas inference input, not held"
                " responses."
            ),
        ),
        (
            0.357,
            0.278,
            "2  Fit and document the interface",
            (
                f"{census.model.nunique()} method/comparator"
                f" groups\n{counts['native']} native"
                f" evaluations\n{counts['adapted']} adapted"
                f" evaluations\n{counts['diagnostic']} diagnostic evaluations"
            ),
            "Model fitting, PCA and reference\nestimation use training-fold cells.",
        ),
        (
            0.689,
            0.286,
            "3  Score matched predictions",
            (
                "Gene-response pattern: Pearson-Δ\nImmune programs: rank-score"
                " readout\nProteins: separate marker analysis\nDistributions: auxiliary"
                " energy distance"
            ),
            (
                "Cell-mean and linear-PCA references;\npaired units and explicit"
                " missing scores."
            ),
        ),
    ]
    for x, width, heading, lines, note in cards:
        box(ax, x, 0.286, width, 0.31)
        ax.text(x + 0.009, 0.558, heading, fontsize=8.4, fontweight="bold")
        ax.plot([x + 0.009, x + width - 0.009], [0.537, 0.537], color=LINE, lw=0.8)
        ax.text(x + 0.009, 0.509, lines, va="top", fontsize=8, linespacing=1.85)
        ax.text(
            x + 0.009, 0.333, note, va="top", fontsize=7.1, color=GREY, linespacing=1.45
        )
    for left, right in [(0.313, 0.344), (0.646, 0.676)]:
        ax.annotate(
            "",
            (right, 0.441),
            (left, 0.441),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1),
        )
    box(ax, 0.02, 0.115, 0.955, 0.113, color="#edf3f7")
    ax.text(
        0.035,
        0.192,
        "Interpret the margin, not just the rank",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0.035,
        0.15,
        "Above both references?  →  Check paired uncertainty, multiplicity, readout"
        " estimability and external validity.",
        fontsize=8,
    )
    ax.text(
        0.02,
        0.06,
        "Scope: shared upstream feature selection is not fully inductive. T4 does not"
        " hold out the protein modality.",
        fontsize=7.5,
        color=GREY,
    )
    ax.text(
        0.02,
        0.028,
        "Compact prediction profiles reproduce Pearson-Δ; they do not retain the cell"
        " clouds needed for energy-distance replay.",
        fontsize=7.5,
        color=GREY,
    )
    for suffix in [".png", ".pdf", ".tiff"]:
        kw = dict(dpi=600 if suffix == ".tiff" else 350, facecolor="white")
        if suffix == ".tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(OUT / ("Figure1" + suffix), **kw)
    plt.close(fig)
    print("Figure1: executed workflow and explicitly scoped evidence")


if __name__ == "__main__":
    main()
