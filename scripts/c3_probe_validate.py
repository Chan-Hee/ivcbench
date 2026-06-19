#!/usr/bin/env python
"""Validate the per-gene probe: macro-average of recomputed floor per-gene predictability must
match the deposited results_raw.csv floor macro (within rounding). Confirms we recovered exactly
the same units/metric the benchmark deposited, so the per-gene decomposition is faithful."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pg = pd.read_csv(ROOT / "results/C3/predictability_probe_pergene.csv")
dep = pd.read_csv(ROOT / "results/C3/results_raw.csv")
dep = dep[dep.ran == True]

# floor macro from deposited = max over simple shift baselines per (dataset,split)
SIMPLE = ["cell-mean", "donor-shift", "linear-PCA"]
print(f"{'dataset':18s} {'hold':4s} {'recomputed_macro':>16s} {'deposited_floor':>15s} {'Δ':>8s}")
maxdiff = 0.0
for (ds, h), g in pg.groupby(["dataset", "hold"]):
    # per gene predictability already = max of the 3 shift baselines; macro = mean over held genes
    recomputed_macro = g.predictability.mean()
    # but deposited floor = max over baselines of (mean over genes of that baseline) — recompute matching
    colmap = {"cell-mean": "pd_cell_mean", "donor-shift": "pd_donor_shift", "linear-PCA": "pd_linear_pca"}
    recomp_per_base = {b: g[colmap[b]].mean() for b in SIMPLE}
    recomputed_floor = max(recomp_per_base.values())
    sp = f"C3_true_lo_gene_{h}"
    dsub = dep[(dep.dataset == ds) & (dep.split == sp) & (dep.baseline.isin(SIMPLE))]
    dep_floor = dsub.pearson_delta.max()
    d = abs(recomputed_floor - dep_floor)
    maxdiff = max(maxdiff, d)
    print(f"{ds:18s} {str(h):4s} {recomputed_floor:16.4f} {dep_floor:15.4f} {d:8.4f}")
print(f"\nmax |recomputed_floor - deposited_floor| = {maxdiff:.5f}  "
      f"({'PASS' if maxdiff < 1e-3 else 'CHECK'})")
print(f"\nNote: per-gene 'predictability' = max-over-3-baselines PER GENE (>= the macro floor, which is "
      f"max-over-baselines-of-mean). Both are reported.")
