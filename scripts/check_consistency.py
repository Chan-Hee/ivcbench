#!/usr/bin/env python3
"""Read-only consistency gate for the bundle-derived census and its provenance.

Rebuilds headline values, execution/status metadata, within-family summaries,
the exact input manifest (including hashes), and canonical counts. Historical
non-census files are distinguished from actual method/floor inputs. This checks
reproducibility and internal consistency, not the scientific validity of a runner.
"""
from __future__ import annotations

from io import StringIO
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assemble_cross_cluster import (
    OUT,
    EXPECTED_CENSUS_CELLS,
    build,
    score_all,
    census_bundle_manifest,
    canonical_numbers,
)

KEY = ["cluster", "split", "model"]
NUM_COLS = [
    "pearson_delta",
    "floor_cell_mean",
    "floor_linear_PCA",
    "floor_mean",
    "delta_vs_floor_mean",
    "delta_vs_cell_mean",
    "delta_vs_linear_PCA",
]


def compare_frame(committed, fresh, key, label):
    """Compare the complete CSV schema/content, including nonnumeric provenance."""
    problems = []
    if set(committed.columns) != set(fresh.columns):
        return [
            f"{label}: column mismatch {set(committed.columns) ^ set(fresh.columns)}"
        ]
    for name, frame in (("stored", committed), ("rebuilt", fresh)):
        if frame.duplicated(key).any():
            problems.append(f"{label}: duplicate {name} keys {key}")
    if problems:
        return problems
    # Lists and missing string values must have the same representation as the
    # actual deposited CSV, including the within-family delta vector column.
    expected = pd.read_csv(StringIO(fresh.to_csv(index=False)))
    actual = committed[expected.columns]
    try:
        pd.testing.assert_frame_equal(
            actual.sort_values(key).reset_index(drop=True),
            expected.sort_values(key).reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-9,
        )
    except AssertionError as exc:
        problems.append(f"{label}: {exc}")
    return problems


def check():
    problems = []
    scored = score_all()
    fresh, cons, n_bundles = build(scored)
    manifest = census_bundle_manifest(scored)
    canon = canonical_numbers(fresh, scored, manifest)
    from audit_census_targets import check_targets
    from census_units import build_unit_scores, uncertainty_table
    from assemble_fit_matrix import build_fit_matrix

    try:
        check_targets(manifest)
    except ValueError as exc:
        problems.append(f"Paired target/reference audit: {exc}")
    units = build_unit_scores(scored)
    if len(fresh) != EXPECTED_CENSUS_CELLS:
        problems.append(
            f"headline has {len(fresh)} cells, expected {EXPECTED_CENSUS_CELLS}"
        )
    if not np.isfinite(fresh[NUM_COLS].to_numpy(dtype=float)).all():
        problems.append("Non-finite headline score or floor value")

    tables = [
        ("cross_cluster_headline.csv", fresh, KEY),
        ("within_family_consistency.csv", cons, ["cluster", "split", "family"]),
        ("census_bundle_manifest.csv", manifest, ["task", "model", "unit"]),
        ("census_unit_scores.csv", units, ["task_key", "model", "unit"]),
        ("census_uncertainty.csv", uncertainty_table(units), ["task_key", "model"]),
        (
            "descriptive_fit_matrix.csv",
            build_fit_matrix(units),
            ["task_key", "family", "role"],
        ),
    ]
    for filename, rebuilt, key in tables:
        table_path = os.path.join(OUT, filename)
        if not os.path.isfile(table_path):
            problems.append(f"Missing {filename}")
            continue
        problems.extend(compare_frame(pd.read_csv(table_path), rebuilt, key, filename))

    canonical_path = os.path.join(OUT, "CANONICAL_NUMBERS.json")
    if not os.path.isfile(canonical_path):
        problems.append("Missing CANONICAL_NUMBERS.json")
    else:
        with open(canonical_path) as handle:
            stored = json.load(handle)
        for key in stored.keys() | canon.keys():
            if stored.get(key) != canon.get(key):
                problems.append(
                    f"CANONICAL_NUMBERS.json/{key}: stored {stored.get(key)!r} "
                    f"!= rebuilt {canon.get(key)!r}"
                )

    # Raw result tables retain execution identities, not the CPA/chemCPA group
    # label. The sync checker must not conflate these different experiments.
    from sync_results_raw import drift

    for rel, model, key, old, new in drift(scored):
        problems.append(
            f"results_raw drift: {rel} {model} {key} = {old} != bundle {new:.4f}"
        )
    return problems, n_bundles, len(fresh)


def main():
    problems, n_bundles, n_cells = check()
    if problems:
        print("DEPOSIT CONSISTENCY: FAIL")
        for problem in problems:
            print("  -", problem)
        return 1
    print(
        f"DEPOSIT CONSISTENCY: PASS ({n_bundles} stored bundles re-scored; "
        f"{n_cells} census cells; scores, metadata, manifest hashes and counts agree)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
