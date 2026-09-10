#!/usr/bin/env python3
"""Final-roster dataset and chemical-distance summaries from existing predictions.

No new compound model fitting. T3 uses the census 10% holdout, not a legacy
50% slice with a different roster. Tanimoto distance is the preserved distance
to the training compounds; regression uses current per-compound predictions.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from census_units import ROOT, binding_floor, task_matrix

PAPER = Path(ROOT) / "results/_paper"


def main():
    units = pd.read_csv(PAPER / "census_unit_scores.csv")
    t3 = units[units.task_key == "T3"]
    matrix = task_matrix(units, "T3")
    models = sorted(t3[t3.status.isin(["native", "adapted"])].model.unique())
    floor = binding_floor(matrix)
    rows = []
    for dataset, record in matrix.iterrows():
        best = record[models].idxmax()
        rows.append(
            dict(
                dataset=dataset,
                holdout_pct=10,
                binding_floor=floor,
                floor_score=record[floor],
                best_model=best,
                best_score=record[best],
                best_margin=record[best] - record[floor],
                selection="retrospective per-dataset best; descriptive only",
            )
        )
    pd.DataFrame(rows).to_csv(PAPER / "t3_by_dataset.csv", index=False)
    dist = pd.read_csv(Path(ROOT) / "results/C5/tanimoto_percompound.csv")
    if not (dist.groupby("compound").tanimoto_dist.nunique() == 1).all():
        raise ValueError("Inconsistent preserved compound distance")
    dist = dist.drop_duplicates("compound").set_index("compound").tanimoto_dist
    t5 = units[units.task_key == "T5u"].copy()
    t5["compound"] = t5.unit.str.removeprefix("perturbation=")
    if set(t5.compound) != set(dist.index):
        raise ValueError("Census compounds differ from the Tanimoto analysis")
    t5["tanimoto_distance"] = t5.compound.map(dist)
    rows = []
    for model, frame in t5.groupby("model"):
        x = (
            frame.tanimoto_distance - frame.tanimoto_distance.mean()
        ) / frame.tanimoto_distance.std(ddof=1)
        y = 1 - frame.pearson_delta
        reg = stats.linregress(x, y)
        n = len(x)
        df = n - 2
        lo, hi = reg.slope + np.array([-1, 1]) * stats.t.ppf(0.95, df) * reg.stderr
        equivalence = max(
            stats.t.sf((reg.slope + 0.05) / reg.stderr, df),
            stats.t.cdf((reg.slope - 0.05) / reg.stderr, df),
        )
        rows.append(
            dict(
                model=model,
                execution_model=frame.execution_model.iloc[0],
                n_compounds=n,
                slope_per_distance_sd=reg.slope,
                ci90_lo=lo,
                ci90_hi=hi,
                slope_p=reg.pvalue,
                tost_p=equivalence,
                r_squared=reg.rvalue**2,
                equivalence_bound=0.05,
                inference="exploratory; unadjusted; observed distance range only",
            )
        )
    result = pd.DataFrame(rows)
    result.to_csv(PAPER / "op3_tanimoto_sensitivity.csv", index=False)
    t5.to_csv(PAPER / "current_tanimoto_percompound.csv", index=False)
    (PAPER / "secondary_check_provenance.json").write_text(
        json.dumps(
            dict(
                t3=(
                    "five dataset-arms from four studies; 10% held targets;"
                    f" {len(models)} conditioned methods"
                ),
                t5=(
                    "current 28-compound scores paired with preserved distances to"
                    " training compounds"
                ),
                distance_scope=(
                    "linear association within one test set and fingerprint"
                    " representation; not proof that chemical similarity is irrelevant"
                ),
                seeds="same fitted prediction policy as census; no retraining",
            ),
            indent=2,
        )
        + "\n"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
