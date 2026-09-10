#!/usr/bin/env python3
"""Multiplicity over all current census comparisons with at least eight units.

This is a post-review analysis of the final executed roster, not a public
preregistration or a selective family of positive results. All eligible
two-sided paired contrasts enter one BH/Holm family, including negative ones.
The smaller-sample entries remain descriptive (ranges in Table S5). The six-lineage OP3
sensitivity is separate from the four-lineage census and does not supply its p.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from census_units import ROOT, MIN_BOOT, build_unit_scores, binding_floor, task_matrix


def adjust(pvalues):
    """BH and Holm, including monotonicity, with stable ordering."""
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid multiplicity p-values")
    m = len(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    bh_order = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    holm_order = np.maximum.accumulate(ranked * np.arange(m, 0, -1))
    bh, holm = np.empty(m), np.empty(m)
    bh[order], holm[order] = np.minimum(bh_order, 1), np.minimum(holm_order, 1)
    return bh, holm


def main():
    units = build_unit_scores()
    rows = []
    for task in sorted(units.task_key.unique()):
        matrix = task_matrix(units, task)
        floor = binding_floor(matrix)
        for model in sorted(set(matrix.columns) - {"cell-mean", "linear-PCA"}):
            gaps = (matrix[model] - matrix[floor]).to_numpy(float)
            n = len(gaps)
            p = (
                float(wilcoxon(gaps, alternative="two-sided").pvalue)
                if n >= MIN_BOOT and np.any(gaps != 0)
                else 1.0 if n >= MIN_BOOT else np.nan
            )
            # Retain this identifier for downstream figure/data compatibility.
            key = (
                "C2_donor_CellOT_vs_floor"
                if task == "T2" and model == "CellOT"
                else f"{task}_{model}_vs_floor"
            )
            rows.append(
                dict(
                    contrast=key,
                    task_key=task,
                    model=model,
                    n_units=n,
                    binding_floor=floor,
                    mean_gap=float(gaps.mean()),
                    units_positive=int((gaps > 0).sum()),
                    raw_p=p,
                    family=(
                        "current_census_n_ge_8"
                        if n >= MIN_BOOT
                        else "descriptive_low_n"
                    ),
                    inference=(
                        "paired two-sided Wilcoxon; fixed floor; post-review"
                        if n >= MIN_BOOT
                        else "no inferential test; unit range in Table S5"
                    ),
                )
            )
    result = pd.DataFrame(rows)
    valid = result.raw_p.notna()
    result["BH_p"], result["Holm_p"] = np.nan, np.nan
    bh, holm = adjust(result.loc[valid, "raw_p"])
    result.loc[valid, "BH_p"], result.loc[valid, "Holm_p"] = bh, holm
    result["survives_FDR05"] = result.BH_p < 0.05
    result["positive_supported"] = (
        (result.mean_gap > 0) & (result.BH_p < 0.05) & (result.Holm_p < 0.05)
    )
    result["source"] = (
        "results/_paper/census_unit_scores.csv; exact current model/floor units"
    )
    paper = Path(ROOT) / "results/_paper"
    result.to_csv(paper / "headline_multiplicity_adjusted.csv", index=False)
    report = dict(
        family_size=int(valid.sum()),
        descriptive_cells=int((~valid).sum()),
        positive_supported=result.loc[result.positive_supported, "contrast"].tolist(),
        caveats=[
            "post-review family, not publicly preregistered",
            "floor selection and shared training folds are conditioned upon",
            (
                "T3 dataset-arms and T4 overlapping fractions are not independent"
                " replicates"
            ),
            (
                "historical four-/three-test families do not adjust the full current"
                " census"
            ),
            "no Fisher combination of dependent OP3 lineage permutations",
        ],
    )
    (paper / "multiplicity_provenance.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    print(result.loc[result.positive_supported].to_string(index=False))


if __name__ == "__main__":
    main()
