#!/usr/bin/env python3
"""Connect every panel comparison to precision and matched-target repeatability.

Replay uses retained partition correlations, not raw cells. Power-scale and
attenuation statements remain distinct: a conditional MDE does not identify a
noise-free target, and cell-sampling repeatability is not biological replication.
"""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from assemble_cross_cluster import EXPECTED_CENSUS_CELLS, ROOT

PAPER = Path(ROOT) / "results/_paper"
DATASETS = [
    "kang",
    "soskic",
    "shifrut",
    "schmidt",
    "mccutcheon_CRISPRi",
    "mccutcheon_CRISPRa",
    "chen",
    "frangieh",
    "op3",
]


def summarize_repeatability(
    strata: pd.DataFrame, draws: pd.DataFrame, units: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weight strata within units, then units within tasks, matching the census."""
    summaries = []
    unit_scores = strata.groupby(["task_key", "unit"])[
        ["independent_control_r", "shared_control_r"]
    ].mean()
    unit_draws = draws.groupby(["task_key", "unit", "partition"])[
        ["independent_control_r", "shared_control_r"]
    ].mean()
    for task, frame in strata.groupby("task_key"):
        expected = set(
            units[(units.task_key == task) & (units.model == "cell-mean")].unit
        )
        if set(frame.unit) != expected:
            raise ValueError(f"{task}: repeatability units differ from the panel")
        score = unit_scores.loc[task]
        partition_means = unit_draws.loc[task].groupby("partition").mean()
        row = dict(
            task_key=task,
            n_units=len(expected),
            n_strata=len(frame),
            n_estimable_strata=int((frame.status == "estimable").sum()),
            n_partitions=int(frame.partitions.min()),
            independent_control_r=float(score.independent_control_r.mean()),
            shared_control_r=float(score.shared_control_r.mean()),
            minimum_treated_cells=int(frame.n_treated.min()),
            minimum_control_cells=int(frame.n_controls.min()),
        )
        row["shared_minus_independent_control"] = (
            row["shared_control_r"] - row["independent_control_r"]
        )
        for metric in ["independent_control_r", "shared_control_r"]:
            row[metric + "_partition_lo"] = float(
                partition_means[metric].quantile(0.025)
            )
            row[metric + "_partition_hi"] = float(
                partition_means[metric].quantile(0.975)
            )
        summaries.append(row)
    summary = pd.DataFrame(summaries).sort_values("task_key")
    if set(summary.task_key) != set(units.task_key):
        raise ValueError("Target-repeatability coverage is incomplete")
    return summary, unit_scores.reset_index()


def describe_precision(row: pd.Series) -> tuple[str, str]:
    """Label the conditional estimate, not architectural ability or learnability."""
    if pd.isna(row.ci_lo):
        statement = (
            f"Only {row.n_units} task units; no inferential power/MDE is claimed. Even"
            " under independence, the minimum two-sided exact signed-rank P is"
            f" {2.0**(1-int(row.n_units)):.3f}; shared folds, related lineages/arms or"
            " overlapping fractions further limit inference."
        )
        return "descriptive only", statement
    statement = (
        f"Conditional 95% margin interval [{row.ci_lo:.6f}, {row.ci_hi:.6f}];"
        f" normal-approximation 80%-power margin scale {row.mde_80:.6f}, not"
        " prospective Wilcoxon power or a multiplicity-adjusted detection threshold."
        " Training-seed and shared-fold variation are not covered."
    )
    if row.ci_lo <= 0 <= row.ci_hi:
        category = "unresolved difference"
    elif row.ci_lo > 0:
        category = "positive conditional interval"
    else:
        category = "negative conditional interval"
    return category, statement


def describe_attenuation(row: pd.Series) -> str:
    """State what the matched-control comparison can and cannot identify."""
    return (
        "Matched-mask held-target split-half correlation"
        f" {row.independent_control_r:.6f} with disjoint controls versus"
        f" {row.shared_control_r:.6f} with shared controls;"
        f" {row.n_estimable_strata}/{row.n_strata} strata estimable. The pooled control"
        " has more cells than each disjoint half, so this compares control dependence"
        " and sampling precision. This diagnoses cell-sampling noise under the"
        " recorded feature space, not a prediction ceiling, a correction for"
        " attenuation, independent biological replication, or separation of model"
        " error from an unlearnable target."
    )


def attach_diagnostics(
    uncertainty: pd.DataFrame, repeatability: pd.DataFrame
) -> pd.DataFrame:
    """Attach a precision statement and target diagnostic to every panel row."""
    panel = uncertainty.merge(
        repeatability, on="task_key", validate="many_to_one", suffixes=("", "_target")
    )
    for index, row in panel.iterrows():
        category, statement = describe_precision(row)
        panel.loc[index, "precision_class"] = category
        panel.loc[index, "power_statement"] = statement
        panel.loc[index, "attenuation_statement"] = describe_attenuation(row)
    # One row per reported cell. The count was written out as 47 when the census was that size,
    # which then failed the whole summaries chain the moment the revision added cells. Take it
    # from the assembler, which is where the panel size is decided.
    assert len(panel) == EXPECTED_CENSUS_CELLS, (
        f"{len(panel)} panel rows against {EXPECTED_CENSUS_CELLS} census cells"
    )
    assert panel.power_statement.notna().all()
    assert panel.attenuation_statement.notna().all()
    return panel


def main() -> None:
    folder = PAPER / "target_repeatability"
    strata = pd.concat(
        [pd.read_csv(folder / f"{d}.csv", dtype={"unit": str}) for d in DATASETS],
        ignore_index=True,
    )
    draws = pd.concat(
        [
            pd.read_csv(folder / f"{d}_partitions.csv", dtype={"unit": str})
            for d in DATASETS
        ],
        ignore_index=True,
    )
    units = pd.read_csv(PAPER / "census_unit_scores.csv", dtype={"unit": str})
    summary, unit_scores = summarize_repeatability(strata, draws, units)
    summary.to_csv(PAPER / "target_repeatability_summary.csv", index=False)
    unit_scores.to_csv(PAPER / "target_repeatability_units.csv", index=False)
    panel = attach_diagnostics(pd.read_csv(PAPER / "census_uncertainty.csv"), summary)
    panel.to_csv(PAPER / "panel_precision_attenuation.csv", index=False)
    checks = [json.loads((folder / f"{d}.json").read_text()) for d in DATASETS]
    provenance = dict(
        scope=(
            "exact census held units and gene masks; independent treated/control"
            " half-samples; no model fitting"
        ),
        weighting=(
            "equal strata within each census unit, then equal units within task;"
            " finite-stratum coverage printed explicitly"
        ),
        partitions=(
            "100 disjoint partitions; 2.5–97.5% partition ranges are not biological"
            " confidence intervals"
        ),
        control_comparison=(
            "Independent half-controls versus the same pooled control: both sample size"
            " and control-error dependence differ; the contrast is not a pure"
            " covariance-effect estimate"
        ),
        matches=[s for d in checks for s in d["inputs"]],
        inference=(
            "Each of the 47 contrasts has a precision/power-scale and attenuation"
            " statement; no model-failure or unlearnability attribution"
        ),
    )
    (PAPER / "target_repeatability_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
