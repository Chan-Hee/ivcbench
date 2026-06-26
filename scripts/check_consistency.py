#!/usr/bin/env python3
"""Deposit consistency gate — the census must always equal what the bundles re-score to.

Single source of truth = the deposited prediction bundles. `assemble_cross_cluster.build()`
re-scores them (GPU-free) and assembles the 35-cell headline; this gate rebuilds that table
from scratch and fails if the committed `results/_paper/cross_cluster_headline.csv` (or the
within-family table) has drifted from it. Run by `make test`, so a hand-edited number, a stale
bundle, or a metric change can never silently land a census the reader cannot reproduce.

    python scripts/check_consistency.py        # exit 0 = consistent, 1 = drift (prints the cells)

It checks three things: (1) every committed headline number equals the freshly re-scored value
to 1e-9; (2) the prediction layer is exactly the expected size with no NaN; (3) the floor-clearing
verdicts are intact.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assemble_cross_cluster import OUT, build  # noqa: E402

EXPECTED_BUNDLES = 1465          # 1469 deposited .npz minus the 4 predictions/example format demos
EXPECTED_CELLS = 35
FLOOR_CLEARERS = {("C2", "CellOT"), ("C5", "FP-ridge")}  # the only two cells that beat both floors
NUM_COLS = ["pearson_delta", "floor_cell_mean", "floor_linear_PCA", "floor_mean",
            "delta_vs_floor_mean", "delta_vs_cell_mean", "delta_vs_linear_PCA"]
KEY = ["cluster", "split", "model"]


def check():
    problems = []
    fresh, cons, n_bundles = build()

    # (2) prediction-layer size + no NaN
    if n_bundles != EXPECTED_BUNDLES:
        problems.append(f"bundle count {n_bundles} != expected {EXPECTED_BUNDLES}")
    if len(fresh) != EXPECTED_CELLS:
        problems.append(f"headline has {len(fresh)} cells != expected {EXPECTED_CELLS}")
    if fresh["pearson_delta"].isna().any():
        problems.append("NaN pearson_delta in freshly re-scored headline")

    # (1) committed census must equal the freshly re-scored census, cell by cell
    committed = pd.read_csv(os.path.join(OUT, "cross_cluster_headline.csv"))
    m = committed.merge(fresh, on=KEY, suffixes=("_committed", "_fresh"), how="outer", indicator=True)
    miss = m[m["_merge"] != "both"]
    for _, r in miss.iterrows():
        problems.append(f"roster mismatch: {r.cluster}/{r.split}/{r.model} only in {r._merge}")
    both = m[m["_merge"] == "both"]
    for col in NUM_COLS:
        d = (both[f"{col}_committed"] - both[f"{col}_fresh"]).abs()
        for i in both.index[d > 1e-9]:
            r = both.loc[i]
            problems.append(f"{r.cluster}/{r.split}/{r.model} {col}: "
                            f"committed {r[f'{col}_committed']} != re-scored {r[f'{col}_fresh']}")
    bad_verdict = both[both["beats_both_floor_members_committed"] != both["beats_both_floor_members_fresh"]]
    for _, r in bad_verdict.iterrows():
        problems.append(f"{r.cluster}/{r.split}/{r.model} floor verdict drifted")

    # (3) the two headline floor-clearers are intact and unique
    clearers = {(r.cluster, r.model) for _, r in fresh[fresh.beats_both_floor_members].iterrows()}
    if clearers != FLOOR_CLEARERS:
        problems.append(f"floor-clearing cells {sorted(clearers)} != expected {sorted(FLOOR_CLEARERS)}")

    # (4) the per-cluster results_raw.csv the figures read must agree with the bundles at display
    # precision (scripts/sync_results_raw.py enforces this) — guards the figures from drifting.
    from sync_results_raw import drift
    for rel, model, key, old, new in drift():
        problems.append(f"results_raw drift: {rel} {model} {key} = {old} != bundle {new:.4f}")

    return problems, n_bundles, len(fresh)


def main():
    problems, n_bundles, n_cells = check()
    if problems:
        print("DEPOSIT CONSISTENCY: FAIL")
        for p in problems:
            print("  -", p)
        print(f"\n{len(problems)} problem(s). The committed census no longer equals the bundle re-score.")
        return 1
    print(f"DEPOSIT CONSISTENCY: PASS  ({n_bundles} bundles re-scored -> {n_cells} census cells, "
          "every number reproduces, floor verdicts intact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
