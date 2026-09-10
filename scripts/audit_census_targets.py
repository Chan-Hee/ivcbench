#!/usr/bin/env python3
"""Verify paired held-out observations, reference controls and gene masks in the census."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def check_targets(manifest):
    checked = 0
    max_difference = {"obs_means": 0.0, "control_mean": 0.0}
    for (task, unit), rows in manifest.groupby(["task", "unit"]):
        reference = rows[rows.model == "cell-mean"]
        if len(reference) != 1:
            raise ValueError(f"{task}/{unit}: ambiguous floor reference")
        with np.load(ROOT / reference.iloc[0].bundle_path, allow_pickle=True) as bundle:
            expected = {
                key: bundle[key]
                for key in ("genes", "strata", "obs_means", "control_mean")
            }
            mask = (
                sorted(bundle["exclude_gene_idx"].tolist())
                if "exclude_gene_idx" in bundle.files
                else []
            )
        for row in rows.itertuples():
            with np.load(ROOT / row.bundle_path, allow_pickle=True) as bundle:
                for key in ("genes", "strata"):
                    if not np.array_equal(expected[key], bundle[key]):
                        raise ValueError(f"{task}/{unit}/{row.model}: different {key}")
                for key in max_difference:
                    a, b = expected[key], bundle[key]
                    if a.shape != b.shape or not np.allclose(
                        a, b, atol=1e-4, rtol=1e-5
                    ):
                        raise ValueError(f"{task}/{unit}/{row.model}: different {key}")
                    max_difference[key] = max(
                        max_difference[key], float(np.max(np.abs(a - b)))
                    )
                observed_mask = (
                    sorted(bundle["exclude_gene_idx"].tolist())
                    if "exclude_gene_idx" in bundle.files
                    else []
                )
                if observed_mask != mask:
                    raise ValueError(
                        f"{task}/{unit}/{row.model}: different evaluation gene mask"
                    )
            checked += 1
    return {
        "bundles_checked": checked,
        "paired_targets_controls_genes_masks": True,
        "maximum_float32_difference": max_difference,
        "tolerance": {"absolute": 1e-4, "relative": 1e-5},
        "scope": (
            "Evaluation inputs only; does not validate model fitting or biological"
            " generalizability"
        ),
    }


def main():
    manifest = pd.read_csv(
        ROOT / "results/_paper/census_bundle_manifest.csv", dtype={"unit": str}
    )
    result = check_targets(manifest)
    (ROOT / "results/_paper/census_target_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
