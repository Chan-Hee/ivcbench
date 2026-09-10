#!/usr/bin/env python3
"""Summarize preserved donor validation and auxiliary metrics without model fitting.

External cohorts have per-seed scalar scores, not retained cell predictions.
The two floor members are compared at cohort-macro level, then the selected
member is paired with every donor, as in the census. Auxiliary Soskic metrics
retain their original per-donor context reference and three-seed averaging.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT.parent / "revision_BIB-26-1553/03_etc/04_new_analysis"
if (ROOT / "results/provenance_inputs/donor").is_dir():
    ANALYSIS = ROOT / "results/provenance_inputs/donor"
PAPER = ROOT / "results/_paper"


def describe(values):
    values = np.asarray(values, float)
    n = len(values)
    row = dict(
        n=n,
        mean_margin=float(values.mean()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        n_positive=int((values > 0).sum()),
        ci_lo=np.nan,
        ci_hi=np.nan,
        p_value=np.nan,
        inference="observed range only; fewer than eight donors",
    )
    if n >= 8:
        rng = np.random.default_rng(0)
        means = values[rng.integers(0, n, size=(20000, n))].mean(axis=1)
        row.update(
            ci_lo=float(np.quantile(means, 0.025)),
            ci_hi=float(np.quantile(means, 0.975)),
            p_value=float(wilcoxon(values, alternative="two-sided").pvalue),
            inference=(
                "conditional donor bootstrap; two-sided Wilcoxon, unadjusted"
                " sensitivity"
            ),
        )
    return row


def main():
    sources, external, result = [], [], []
    for cohort in ["kang", "cano_gamez"]:
        base = ANALYSIS / "raw/foundation"
        paths = [base / f"{cohort}_lodo_per_seed.csv", base / f"{cohort}_lodo.csv"]
        seeds, agg = [pd.read_csv(p, dtype={"held_donor": str}) for p in paths]
        if seeds.duplicated(["held_donor", "seed"]).any():
            raise ValueError("Duplicate donor-seed validation record")
        if set(seeds.seed) != {0, 1} or set(seeds.held_donor) != set(agg.held_donor):
            raise ValueError("Incomplete donor-seed coverage")
        if not (seeds.groupby("held_donor").size() == 2).all():
            raise ValueError("Expected both training seeds for every donor")
        frame = agg.set_index("held_donor").copy()
        mean = seeds.groupby("held_donor").cellot.mean()
        if float((frame.cellot - mean).abs().max()) > 0.000051:
            raise ValueError("Seed means disagree with preserved donor aggregates")
        frame["cellot"] = mean
        floor = frame[["cell_mean", "linear_pca"]].mean().idxmax()
        frame["floor_member"] = floor
        frame["floor_score"] = frame[floor]
        frame["margin"] = frame.cellot - frame.floor_score
        frame["cohort"] = cohort
        frame["seeds"] = "0,1; scalar scores averaged within donor"
        external.append(frame.reset_index())
        result.append(
            dict(
                analysis=cohort,
                metric="Pearson-delta",
                model_score=frame.cellot.mean(),
                reference_score=frame.floor_score.mean(),
                reference=floor,
                seeds="0,1",
                **describe(frame.margin),
            )
        )
        sources.extend(paths)
    foldpath = ANALYSIS / "foldlocal_sensitivity.csv"
    fold = pd.read_csv(foldpath)
    floor = fold[["cell_mean", "linear_PCA"]].mean().idxmax()
    fold["fixed_floor_margin"] = fold.cellot - fold[floor]
    fold.to_csv(PAPER / "donor_foldlocal_standardization.csv", index=False)
    result.append(
        dict(
            analysis="soskic_fold_local_standardization",
            metric="Pearson-delta",
            model_score=fold.cellot.mean(),
            reference_score=fold[floor].mean(),
            reference=floor,
            seeds="0",
            **describe(fold.fixed_floor_margin),
        )
    )
    sources.append(foldpath)
    auxpath = PAPER / "cellot_soskic_raw.csv"
    auxiliary = pd.read_csv(auxpath)
    auxiliary = auxiliary[
        auxiliary.metric.isin(["e_distance", "aucell_delta_score"])
    ].copy()
    for metric, frame in auxiliary.groupby("metric"):
        if len(frame) != 106 or frame.donor.nunique() != 106:
            raise ValueError(f"Incomplete donor coverage for {metric}")
        mean = frame.seed_scores.map(lambda x: np.mean(json.loads(x)))
        if float((mean - frame.cellot_score).abs().max()) > 0.00011:
            raise ValueError("Auxiliary metric seed aggregation mismatch")
        margin = (
            frame.baseline_score - frame.cellot_score
            if metric == "e_distance"
            else frame.cellot_score - frame.baseline_score
        )
        if not np.allclose(margin, frame.delta_vs_primary, atol=0.00011):
            raise ValueError("Auxiliary metric direction mismatch")
        result.append(
            dict(
                analysis="soskic_auxiliary_three_seed",
                metric=metric,
                model_score=frame.cellot_score.mean(),
                reference_score=frame.baseline_score.mean(),
                reference="metric-specific better cell-mean/donor-shift per donor",
                seeds="0,1,2",
                **describe(margin),
            )
        )
    sources.append(auxpath)
    auxiliary.to_csv(PAPER / "donor_auxiliary_metrics.csv", index=False)
    pd.concat(external, ignore_index=True).to_csv(
        PAPER / "donor_external_validation.csv", index=False
    )
    summary = pd.DataFrame(result)
    summary.to_csv(PAPER / "donor_validation_summary.csv", index=False)
    provenance = dict(
        source_files=[
            dict(
                path=str(p.relative_to(ROOT.parent)),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
            for p in sources
        ],
        external_cohorts=(
            "two recorded scalar scores per donor; seed aggregation verified; cell"
            " outputs not retained"
        ),
        baseline_precision=(
            "external floor members retained to four decimal places; model seed scores"
            " to six"
        ),
        inference=(
            "conditional on fitted models; external and auxiliary checks outside the"
            " main census-test family"
        ),
        auxiliary=(
            "three-seed scalar metrics with metric-specific context references, not"
            " mean-bundle energy-distance replay"
        ),
        standardization=(
            "training-only scaling sensitivity; does not reselect upstream HVGs"
        ),
    )
    (PAPER / "donor_validation_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
