#!/usr/bin/env python3
"""Biological-unit scores and paired margins for the executed census.

The headline, S3 and S18 must describe the same selected predictions, units and
floor. T5u expands the preserved bundle into 28 compounds; no uncertainty is
borrowed from another model or from the separate six-lineage sensitivity split.
The binding floor member is selected once per task on its macro-average. All
intervals condition on that selection and on the recorded fitted predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from assemble_cross_cluster import CELLS, ROOT, build, cell_status, score_all
from ivcbench.metrics.response import pearson_delta

FLOORS = ["cell-mean", "linear-PCA"]
PAPER = Path(ROOT) / "results/_paper"
MIN_BOOT = 8
BOOTSTRAPS = 4000
BOOTSTRAP_SEED = 0


def task_key(cell):
    if cell["cl"] == "C5":
        return "T5u" if cell["split"] == "unseen-compound" else "T5c"
    return cell["task_id"]


def bundle_stratum_scores(path):
    """Re-score every stratum using the same metric and gene exclusions as the headline."""
    with np.load(path, allow_pickle=True) as data:
        excluded = (
            np.asarray(data["exclude_gene_idx"], int)
            if "exclude_gene_idx" in data.files
            else None
        )
        if "pred_means" in data.files:
            pred, obs, strata = data["pred_means"], data["obs_means"], data["strata"]
        else:
            pred, obs, strata = (
                data["pred_cells"],
                data["test_cells"],
                data["cell_strata"],
            )
        result = pearson_delta(
            np.asarray(pred, np.float64),
            np.asarray(obs, np.float64),
            np.asarray(data["control_mean"], np.float64),
            np.asarray(strata),
            exclude_genes=excluded,
        )
    return {str(key): float(value) for key, value in result["per_stratum"].items()}


def validate_units(frame):
    """Require one finite score per model/unit and identical coverage with both floors."""
    if frame.duplicated(["task_key", "model", "unit"]).any():
        raise ValueError("Duplicate census model/unit scores")
    if not np.isfinite(frame.pearson_delta.to_numpy(float)).all():
        raise ValueError("Non-finite census unit score")
    for cell in CELLS:
        key = task_key(cell)
        rows = frame[frame.task_key == key]
        expected_models = set(cell["roster"] + FLOORS)
        if set(rows.model) != expected_models:
            raise ValueError(f"{key}: model roster differs from census")
        units = {
            model: set(group.unit.astype(str)) for model, group in rows.groupby("model")
        }
        reference = units[FLOORS[0]]
        if len(reference) != cell["n_unit"]:
            raise ValueError(
                f"{key}: {len(reference)} units, expected {cell['n_unit']}"
            )
        for model, observed in units.items():
            if observed != reference:
                raise ValueError(
                    f"{key}/{model}: unpaired units {observed ^ reference}"
                )


def build_unit_scores(scored=None):
    scored = score_all() if scored is None else scored
    records = []
    for cell in CELLS:
        key = task_key(cell)
        selected = scored[
            scored.cluster.isin(cell["clusters"])
            & scored.split.map(cell["match"])
            & scored.model.isin(cell["roster"] + FLOORS)
        ]
        for _, row in selected.iterrows():
            if key == "T5u":
                values = bundle_stratum_scores(Path(ROOT) / row.bundle_path)
                grouped = {}
                for stratum, value in values.items():
                    grouped.setdefault(stratum.split("|")[0], []).append(value)
                sizes = {len(values) for values in grouped.values()}
                if sizes != {4}:
                    raise ValueError(
                        f"{key}/{row.model}: expected four lineages per compound, got"
                        f" {sizes}"
                    )
                values = {
                    unit: float(np.mean(values)) for unit, values in grouped.items()
                }
            else:
                values = {str(cell["unit_of"](row)): float(row.pearson_delta)}
            for unit, value in values.items():
                records.append(
                    dict(
                        task_key=key,
                        task_id=cell["task_id"],
                        cluster=cell["cl"],
                        task=cell["task"],
                        split=cell["split"],
                        unit_kind=cell["unit"],
                        unit=unit,
                        model=row.model,
                        execution_model=row.execution_model,
                        status=(
                            "floor"
                            if row.model in FLOORS
                            else cell_status(cell["cl"], cell["split"], row.model)
                        ),
                        pearson_delta=value,
                        bundle_path=row.bundle_path,
                    )
                )
    frame = pd.DataFrame(records)
    validate_units(frame)
    # T5u has a different file-to-unit mapping. Assert that its within-compound
    # macro-average (and every other task) reproduces the headline, not just its sign.
    headline, _, _ = build(scored)
    for row in headline.itertuples():
        values = frame[
            (frame.cluster == row.cluster)
            & (frame.split == row.split)
            & (frame.model == row.model)
        ].pearson_delta
        if not np.isclose(values.mean(), row.pearson_delta, atol=0.000051, rtol=0):
            raise ValueError(
                f"Unit/headline mismatch: {row.cluster}/{row.split}/{row.model}"
            )
    return frame.sort_values(["task_key", "model", "unit"], ignore_index=True)


def task_matrix(frame, key):
    rows = frame[frame.task_key == key]
    return rows.pivot(
        index="unit", columns="model", values="pearson_delta"
    ).sort_index()


def binding_floor(matrix):
    """One fixed member, selected on the task macro-average (ties prefer cell-mean)."""
    if matrix[FLOORS].isna().any().any():
        raise ValueError("Incomplete paired floor")
    return max(FLOORS, key=lambda model: float(matrix[model].mean()))


def paired_summary(model_scores, floor_scores):
    model_scores = np.asarray(model_scores, float)
    floor_scores = np.asarray(floor_scores, float)
    if model_scores.shape != floor_scores.shape or model_scores.ndim != 1:
        raise ValueError("Scores must be paired one-dimensional vectors")
    margins = model_scores - floor_scores
    if len(margins) < 2 or not np.isfinite(margins).all():
        raise ValueError("At least two finite paired units are required")
    n = len(margins)
    record = dict(
        n_units=n,
        model_score=float(model_scores.mean()),
        binding_floor_score=float(floor_scores.mean()),
        margin=float(margins.mean()),
        margin_min=float(margins.min()),
        margin_max=float(margins.max()),
        units_below_floor=f"{int((margins < 0).sum())}/{n}",
        units_above_floor=f"{int((margins > 0).sum())}/{n}",
    )
    if n >= MIN_BOOT:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        means = margins[rng.integers(0, n, size=(BOOTSTRAPS, n))].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        mde = (
            (1.959963984540054 + 0.8416212335729143)
            * (hi - lo)
            / (2 * 1.959963984540054)
        )
        verdict = (
            "positive margin CI (unadjusted)"
            if lo > 0
            else (
                "negative margin CI (unadjusted)"
                if hi < 0
                else "inconclusive (interval spans zero)"
            )
        )
        record.update(
            inference="bootstrap 95% CI",
            ci_lo=float(lo),
            ci_hi=float(hi),
            mde_80=float(mde),
            verdict=verdict,
        )
    else:
        verdict = (
            "positive in all units (descriptive)"
            if margins.min() > 0
            else (
                "negative in all units (descriptive)"
                if margins.max() < 0
                else "mixed unit margins (descriptive)"
            )
        )
        record.update(
            inference=f"per-unit range (n = {n}; CI not reported)",
            ci_lo=np.nan,
            ci_hi=np.nan,
            mde_80=np.nan,
            verdict=verdict,
        )
    return record


def uncertainty_table(frame):
    validate_units(frame)
    records = []
    for cell in CELLS:
        key = task_key(cell)
        matrix = task_matrix(frame, key)
        floor = binding_floor(matrix)
        for model in cell["roster"]:
            record = dict(
                task_key=key,
                task_id=cell["task_id"],
                task=cell["task"],
                cluster=cell["cl"],
                split=cell["split"],
                unit=cell["unit"],
                model=model,
                status=cell_status(cell["cl"], cell["split"], model),
                binding_floor=floor,
            )
            record.update(paired_summary(matrix[model], matrix[floor]))
            records.append(record)
    return pd.DataFrame(records)


def main():
    units = build_unit_scores()
    summary = uncertainty_table(units)
    units.to_csv(PAPER / "census_unit_scores.csv", index=False)
    summary.to_csv(PAPER / "census_uncertainty.csv", index=False)
    print(
        json.dumps(
            {
                "unit_scores": len(units),
                "census_cells": len(summary),
                "verdicts": summary.verdict.value_counts().to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
