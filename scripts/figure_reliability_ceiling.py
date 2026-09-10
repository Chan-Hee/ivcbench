#!/usr/bin/env python3
"""Plot exact-held-target repeatability, not a prediction ceiling.

The historical output stem avoids duplicate supplementary artwork. The new
source matches census targets and masks, with disjoint control half-samples.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ivcbench.report.style import set_pub_style, despine, NAVY, SIMPLE_GREY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/_paper"
LABELS = {
    "T1": "T1 · lineages",
    "T2": "T2 · donors",
    "T3": "T3 · held genes",
    "T4": "T4 · held KOs",
    "T5c": "T5c · cell contexts",
    "T5u": "T5u · held compounds",
}


def main():
    data = (
        pd.read_csv(OUT / "target_repeatability_summary.csv")
        .set_index("task_key")
        .loc[list(LABELS)]
    )
    set_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    fig.subplots_adjust(left=0.26, right=0.96, bottom=0.27, top=0.83)
    for index, (task, row) in enumerate(data.iterrows()):
        y = len(data) - index - 1
        ax.plot(
            [row.independent_control_r, row.shared_control_r], [y, y], color="0.8", lw=1
        )
        for metric, offset, marker, color, label in [
            ("independent_control_r", -0.10, "o", NAVY, "Disjoint control halves"),
            ("shared_control_r", 0.10, "D", SIMPLE_GREY, "Same pooled control"),
        ]:
            mean = row[metric]
            lo, hi = row[metric + "_partition_lo"], row[metric + "_partition_hi"]
            ax.hlines(y + offset, lo, hi, color=color, lw=1.5)
            ax.scatter(
                mean,
                y + offset,
                s=28,
                marker=marker,
                color=color,
                label=label if index == 0 else None,
                zorder=3,
            )
        ax.text(
            1.01,
            y,
            f"{int(row.n_estimable_strata)}/{int(row.n_strata)}",
            ha="right",
            va="center",
            fontsize=8,
        )
    labels = [
        f"{LABELS[t]}\n{int(data.loc[t, 'n_units'])} primary units"
        for t in reversed(list(LABELS))
    ]
    ax.set_yticks(np.arange(len(data)), labels)
    ax.set_xlim(
        min(-0.05, float(data.independent_control_r_partition_lo.min()) - 0.05), 1.06
    )
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_ylim(-0.65, len(data) - 0.35)
    ax.set_xlabel("Split-half response correlation (matched score masks)")
    ax.text(1.01, len(data) - 0.38, "Strata", ha="right", va="bottom", fontsize=8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.45, 1.20),
        frameon=False,
        ncol=2,
        fontsize=8,
    )
    despine(ax)
    fig.suptitle(
        "Held-target repeatability and shared-control sensitivity",
        y=0.98,
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.15,
        "Exact held targets; 100 treated/control partitions; equal primary-unit"
        " weighting.",
        ha="center",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.10,
        "Bars: 2.5–97.5% partition ranges, not biological confidence intervals.",
        ha="center",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.05,
        "Neither statistic is a prediction ceiling or a correction for attenuation.",
        ha="center",
        fontsize=8,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figure_reliability_ceiling.{ext}", dpi=300)
    plt.close(fig)
    print(
        "S1: six exact-held-target settings; independent and shared control comparison"
    )


if __name__ == "__main__":
    main()
