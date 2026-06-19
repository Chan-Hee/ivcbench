#!/usr/bin/env python
"""Summarize the C3 nearest-gene baseline vs the floor and the deep models, from deposited data only.

Reads results/C3/results_raw.csv (floor + GEARS/AttentionPert + CINEMA-OT) and
results/C3/nearest_gene_baseline.csv (the two nearest-gene priors), and writes:
  results/_paper/c3_nearest_gene_summary.csv  — per-split mean Pearson-Δ (over 5 datasets) for each
                                                method + a "beats cell-mean floor" count over the 15
                                                (dataset, split) cells.
Prints the headline numbers used in the report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["C3_true_lo_gene_10", "C3_true_lo_gene_25", "C3_true_lo_gene_50"]
PCT = {"C3_true_lo_gene_10": 10, "C3_true_lo_gene_25": 25, "C3_true_lo_gene_50": 50}


def main():
    dep = pd.read_csv(ROOT / "results/C3/results_raw.csv")
    dep = dep[dep.ran == True][["baseline", "dataset", "split", "pearson_delta"]]
    nng = pd.read_csv(ROOT / "results/C3/nearest_gene_baseline.csv")
    nng = nng[["baseline", "dataset", "split", "pearson_delta"]]
    grid = pd.concat([dep, nng], ignore_index=True)

    # per (dataset, split) cell-mean floor (the universal floor's strongest member on C3)
    piv = grid.pivot_table(index=["dataset", "split"], columns="baseline",
                           values="pearson_delta", aggfunc="mean")
    floor = piv[["cell-mean", "linear-PCA"]].max(axis=1)

    methods = ["cell-mean", "linear-PCA", "CINEMA-OT", "nearest-gene-coexpr", "nearest-gene-go",
               "GEARS", "AttentionPert"]
    out_rows = []
    for m in methods:
        if m not in piv.columns:
            continue
        row = {"method": m}
        for sp in SPLITS:
            vals = piv[m].xs(sp, level="split")
            row[f"mean_pd_{PCT[sp]}"] = float(vals.mean())
        # beats-floor count over all 15 (dataset, split) cells
        beats = (piv[m] > floor)
        row["beats_floor_cells"] = int(beats.sum())
        row["n_cells"] = int(beats.notna().sum())
        # mean over all splits/datasets
        row["mean_pd_all"] = float(piv[m].mean())
        out_rows.append(row)
    summ = pd.DataFrame(out_rows)
    out = ROOT / "results/_paper/c3_nearest_gene_summary.csv"
    summ.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print(f"\n[wrote] {out}")

    # headline lines
    def g(method, col):
        s = summ[summ.method == method][col]
        return s.iloc[0] if len(s) else float("nan")
    print("\n--- HEADLINE ---")
    print(f"cell-mean floor (mean over all 15 cells):     {float(g('cell-mean','mean_pd_all')):.3f}")
    print(f"nearest-gene co-expr (mean over all cells):   {float(g('nearest-gene-coexpr','mean_pd_all')):.3f} "
          f"(beats floor {int(g('nearest-gene-coexpr','beats_floor_cells'))}/{int(g('nearest-gene-coexpr','n_cells'))})")
    print(f"nearest-gene GO-Jaccard (mean over all cells):{float(g('nearest-gene-go','mean_pd_all')):.3f} "
          f"(beats floor {int(g('nearest-gene-go','beats_floor_cells'))}/{int(g('nearest-gene-go','n_cells'))})")
    for m in ["GEARS", "AttentionPert"]:
        if (summ.method == m).any():
            print(f"{m:14s} (mean over all cells):          {float(g(m,'mean_pd_all')):.3f} "
                  f"(beats floor {int(g(m,'beats_floor_cells'))}/{int(g(m,'n_cells'))})")


if __name__ == "__main__":
    main()
