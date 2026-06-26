#!/usr/bin/env python3
"""Sync the per-cluster results_raw.csv `pearson_delta` to the deposited prediction bundles.

Single source of truth = the bundles. The per-cluster `results/C*/results_raw.csv` tables carry the
rich per-run detail the figures need (on-target recovery, energy distance, AUCell programs, per-unit
rows), but their response-direction `pearson_delta` must agree, cell for cell, with the bundle that a
reviewer re-scores. They can drift when the bundle freeze and the results_raw write come from different
stochastic runs; this script re-derives `pearson_delta` from the bundles wherever a bundle exists for
that (split, model[, dataset, modality]) cell, so every figure reads the same numbers as the headline
census. Cells with no matching bundle (the C4 protein arm, the C3 25/50 percent folds, the C5 fine-6
lineages) keep their deposited value. `scripts/check_consistency.py` verifies the sync held.

    python scripts/sync_results_raw.py        # prints the cells it re-derives
"""
from __future__ import annotations
import os

import pandas as pd

from assemble_cross_cluster import ROOT, score_all

# results_raw file -> the bundle clusters that back it, plus the extra match key columns (besides
# split + model) needed to land each bundle on a unique results_raw row.
SPECS = [
    ("results/C1/results_raw.csv", ["C1_LOCT"], []),               # C1 LOCT: (split, baseline)
    ("results/C3/results_raw.csv", ["C3_LO_gene"], ["dataset"]),   # C3: (split, baseline, dataset)
    ("results/C4/results_raw.csv", ["C4", "C4_Axis2"], ["RNA"]),   # C4: RNA arm only (bundles are RNA)
    ("results/C5/results_raw.csv", ["C1_LOCT", "C5", "C5_unseen_cpd"], []),  # C5 LOCT + unseen-compound
]


def _bundle_map(bdf, clusters, extra):
    m = {}
    for _, r in bdf[bdf["cluster"].isin(clusters)].iterrows():
        key = [r["split"], r["model"]]
        if "dataset" in extra:
            key.append(str(r.get("dataset", "")))
        m[tuple(key)] = float(r["pearson_delta"])
    return m


def _sync_results_raw(bdf, write=True):
    changed = []
    for rel, clusters, extra in SPECS:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        bm = _bundle_map(bdf, clusters, extra)
        rna_only = "RNA" in extra
        for i, r in d.iterrows():
            if rna_only and str(r.get("modality", "")).lower() != "rna":
                continue
            key = [r.get("split"), r.get("baseline")]
            if "dataset" in extra:
                key.append(str(r.get("dataset", "")))
            want = bm.get(tuple(key))
            # only re-derive cells that disagree at 4-decimal display precision (a real stochastic
            # drift), not the sub-display float32 round-trip on deterministic comparators.
            if want is not None and round(float(r["pearson_delta"]), 4) != round(want, 4):
                changed.append((rel, r.get("baseline"), r.get("split"), float(r["pearson_delta"]), want))
                d.at[i, "pearson_delta"] = want
        if write:
            d.to_csv(path, index=False)
    return changed


def _sync_soskic(bdf, write=True):
    """C2 donor axis: per-donor pearson_delta for the models the bundles re-score (scheme C2_lodo)."""
    path = os.path.join(ROOT, "results", "C2", "soskic_donor_axis.csv")
    if not os.path.exists(path):
        return []
    sb = bdf[(bdf["cluster"] == "C2_LODO") & bdf["split"].str.startswith("C2_lodo")].copy()
    sb["donor"] = sb["split"].str.replace("C2_lodo_", "", regex=False)
    bm = {(r["model"], r["donor"]): float(r["pearson_delta"]) for _, r in sb.iterrows()}
    d = pd.read_csv(path)
    changed = []
    for i, r in d.iterrows():
        want = bm.get((r.get("model"), str(r.get("donor"))))
        if want is not None and round(float(r["pearson_delta"]), 4) != round(want, 4):
            changed.append(("soskic_donor_axis.csv", r.get("model"), str(r.get("donor")),
                            float(r["pearson_delta"]), want))
            d.at[i, "pearson_delta"] = want
    if write:
        d.to_csv(path, index=False)
    return changed


def drift(bdf=None):
    """List the results_raw cells that disagree with the bundles at 4dp (without writing). For the guard."""
    if bdf is None:
        bdf = score_all()
    return _sync_results_raw(bdf, write=False) + _sync_soskic(bdf, write=False)


def main():
    bdf = score_all()
    changed = _sync_results_raw(bdf) + _sync_soskic(bdf)
    if not changed:
        print("results_raw already in sync with the bundles (0 cells changed).")
        return
    from collections import Counter
    by_model = Counter(f"{c[0].split('/')[0] if '/' in c[0] else c[0]}:{c[1]}" for c in changed)
    print(f"re-derived {len(changed)} pearson_delta cells from the bundles:")
    for k, v in sorted(by_model.items()):
        print(f"  {k}: {v} cell(s)")


if __name__ == "__main__":
    main()
