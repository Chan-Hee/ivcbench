#!/usr/bin/env python3
"""Rebuild the paired seed-0 CellOT/scGPT donor curve from complete final shards.

This historical curve uses the stronger cell-mean/donor-shift reference per donor,
not the split-fixed cell-mean/linear-PCA census floor. Ten evaluation donors and
the seed-0 donor subsets are matched. scGPT's available training pool grows, but
its runner caps stimulated training cells at 8,000 and fine-tunes for ten epochs.
Intervals resample the ten evaluation donors conditional on these fitted models.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from census_units import paired_summary
from ivcbench.report.style import set_pub_style, despine, NAVY, CLAY_DARK, SIMPLE_GREY

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/newdata"
OUT = ROOT / "results/_paper"
GRID = [8, 16, 32, 64, 96]


def read_arm(files, score):
    frames = [
        pd.read_csv(path).assign(source_file=str(path.relative_to(ROOT)))
        for path in files
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[(frame.metric == "pearson_delta") & (frame.seed == 0)].copy()
    key = ["n_train_donors", "eval_donor"]
    if frame.duplicated(key).any():
        raise ValueError(
            "Overlapping curve shards: do not silently select a pilot or drop"
            " duplicates"
        )
    if set(frame.n_train_donors) != set(GRID) or len(frame) != 50:
        raise ValueError("Incomplete 5 x 10 donor curve")
    donors = set(frame.eval_donor)
    if len(donors) != 10 or any(
        set(group.eval_donor) != donors for _, group in frame.groupby("n_train_donors")
    ):
        raise ValueError("Evaluation donors vary across the grid")
    columns = key + [
        score,
        "primary_baseline",
        "baseline_score",
        "n_train_cells",
        "n_test",
        "n_ctrl",
        "n_response_genes",
        "source_file",
    ]
    return frame[columns].set_index(key).sort_index()


def load_matched():
    cellot_files = [DATA / "cellot_donor_learning_curve.csv"]
    scgpt_files = [DATA / f"scgpt_donor_curve_shard{i}.csv" for i in range(3)]
    cellot = read_arm(cellot_files, "cellot_score")
    scgpt = read_arm(scgpt_files, "scgpt_score")
    if not cellot.index.equals(scgpt.index):
        raise ValueError("CellOT/scGPT evaluation keys differ")
    for column in (
        "primary_baseline",
        "baseline_score",
        "n_train_cells",
        "n_test",
        "n_ctrl",
        "n_response_genes",
    ):
        if not np.array_equal(cellot[column].to_numpy(), scgpt[column].to_numpy()):
            raise ValueError(f"CellOT/scGPT {column} differs")
    matched = cellot.join(scgpt[["scgpt_score", "source_file"]], rsuffix="_scgpt")
    matched["cellot_margin"] = matched.cellot_score - matched.baseline_score
    matched["scgpt_margin"] = matched.scgpt_score - matched.baseline_score
    return matched, cellot_files + scgpt_files


def main():
    matched, inputs = load_matched()
    rows = []
    for donors, group in matched.groupby(level=0):
        record = dict(
            n_train_donors=int(donors),
            n_eval_donors=len(group),
            seed=0,
            available_training_cells=int(group.n_train_cells.median()),
            cellot_score=float(group.cellot_score.mean()),
            scgpt_score=float(group.scgpt_score.mean()),
            reference_score=float(group.baseline_score.mean()),
            reference="per-donor max(cell-mean, donor-shift)",
        )
        for model in ("cellot", "scgpt"):
            summary = paired_summary(group[model + "_score"], group.baseline_score)
            for key in ("margin", "ci_lo", "ci_hi", "units_above_floor"):
                record[model + "_" + key.replace("floor", "reference")] = summary[key]
        rows.append(record)
    summary = pd.DataFrame(rows)
    matched.reset_index().to_csv(OUT / "matched_donor_curve_per_donor.csv", index=False)
    summary.to_csv(OUT / "matched_donor_curve_summary.csv", index=False)
    changes = []
    first, last = matched.xs(8), matched.xs(96)
    for model in ("cellot", "scgpt"):
        for metric in ("score", "margin"):
            diff = last[model + "_" + metric] - first[model + "_" + metric]
            record = paired_summary(diff.to_numpy(), np.zeros(len(diff)))
            changes.append(
                {
                    "model": model,
                    "quantity": metric,
                    "change_96_minus_8": record["margin"],
                    "ci_lo": record["ci_lo"],
                    "ci_hi": record["ci_hi"],
                    "n_eval_donors": len(diff),
                }
            )
    pd.DataFrame(changes).to_csv(
        OUT / "matched_donor_curve_endpoint_changes.csv", index=False
    )
    provenance = dict(
        seed=0,
        n_eval_donors=10,
        matched_fields=[
            "donor",
            "reference",
            "available cells",
            "test/control counts",
            "gene-mask count",
        ],
        reference=(
            "stronger cell-mean/donor-shift member per donor, not the census universal"
            " floor"
        ),
        scgpt_configuration=(
            "end-to-end fine-tuning, 10 epochs, at most 8,000 stimulated training"
            " cells; all grid points share this cap"
        ),
        uncertainty=(
            "4,000 evaluation-donor bootstrap draws, seed 0; fitted models and training"
            " subset fixed"
        ),
        limitation=(
            "one donor-subset seed; training subsets are not nested; does not exclude"
            " optimization or cell-budget effects"
        ),
        sources=[
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in inputs
        ],
    )
    (OUT / "matched_donor_curve_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.85))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.25, top=0.82, wspace=0.32)
    x = summary.n_train_donors.to_numpy()
    for model, label, color, marker in (
        ("cellot", "CellOT", NAVY, "o"),
        ("scgpt", "scGPT", CLAY_DARK, "s"),
    ):
        axes[0].plot(
            x,
            summary[model + "_score"],
            marker=marker,
            color=color,
            label=label,
            lw=1.5,
            ms=4,
        )
        axes[1].plot(
            x,
            summary[model + "_margin"],
            marker=marker,
            color=color,
            label=label,
            lw=1.5,
            ms=4,
        )
        axes[1].fill_between(
            x,
            summary[model + "_ci_lo"],
            summary[model + "_ci_hi"],
            color=color,
            alpha=0.13,
            linewidth=0,
        )
    axes[0].plot(
        x,
        summary.reference_score,
        "--^",
        color=SIMPLE_GREY,
        label="Matched simple reference",
        lw=1.3,
        ms=4,
    )
    axes[1].axhline(0, color=SIMPLE_GREY, lw=0.9, ls="--")
    axes[0].set_ylabel("Response-direction Pearson Δ")
    axes[1].set_ylabel("Model − matched reference")
    axes[0].set_title("a  Recorded scores", loc="left", fontweight="bold", fontsize=9)
    axes[1].set_title("b  Paired margins", loc="left", fontweight="bold", fontsize=9)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(x, [str(v) for v in x])
        axis.set_xlabel("Available training donors")
        despine(axis)
    axes[0].legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle(
        "Donor-pool size under fixed training configurations",
        fontsize=10,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.095,
        "Seed 0; same ten evaluation donors. Bands: conditional 95% donor-bootstrap"
        " intervals.",
        ha="center",
        fontsize=7,
    )
    fig.text(
        0.5,
        0.045,
        "scGPT uses ≤8,000 stimulated training cells. Reference: per-donor cell-mean /"
        " donor-shift maximum.",
        ha="center",
        fontsize=7,
    )
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"figure_donor_learning_curve.{extension}", dpi=300)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(pd.DataFrame(changes).to_string(index=False))


if __name__ == "__main__":
    main()
