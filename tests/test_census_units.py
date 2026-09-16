"""Regression tests for the common paired-unit estimand, not training validity."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from census_units import (
    binding_floor,
    paired_summary,
    uncertainty_table,
    validate_units,
)
from assemble_fit_matrix import build_fit_matrix


def test_floor_member_is_fixed_at_task_level():
    matrix = pd.DataFrame(
        {"cell-mean": [0.8, 0.2], "linear-PCA": [0.1, 0.7], "model": [0.7, 0.3]}
    )
    floor = binding_floor(matrix)
    assert floor == "cell-mean"
    summary = paired_summary(matrix.model, matrix[floor])
    assert summary["margin"] == pytest.approx(0)
    assert np.isnan(summary["ci_lo"])
    assert summary["verdict"] == "mixed unit margins (descriptive)"


def test_paired_summary_rejects_nan_or_different_units():
    with pytest.raises(ValueError):
        paired_summary([1, 2], [1])
    with pytest.raises(ValueError):
        paired_summary([1, np.nan], [0, 0])


def test_current_census_units_and_s3_preserve_member_uncertainty():
    from assemble_cross_cluster import EXPECTED_CENSUS_CELLS

    units = pd.read_csv(
        ROOT / "results/_paper/census_unit_scores.csv", dtype={"unit": str}
    )
    validate_units(units)
    summary = uncertainty_table(units)
    # Derived, not frozen. These read 47 and 1605 through the census expansion to 58 cells, so
    # the test that was supposed to catch a census-shape regression was itself red and caught
    # nothing. The assembler's constant is the census's own declaration of its size.
    assert len(summary) == EXPECTED_CENSUS_CELLS
    assert len(units) == len(units.drop_duplicates(["task_key", "model", "unit"]))
    assert set(summary.task_key) == set(units.task_key)
    matrix = build_fit_matrix(units)
    for row in matrix.itertuples():
        own = summary[
            (summary.task_key == row.task_key) & (summary.model == row.scored_member)
        ].iloc[0]
        assert row.best_model_gap == pytest.approx(own.margin)
        assert row.floor + row.best_model_gap == pytest.approx(row.model_score)
        assert row.n_unit == own.n_units
        assert row.ci_low == pytest.approx(own.ci_lo, nan_ok=True)
        assert row.ci_high == pytest.approx(own.ci_hi, nan_ok=True)
    fp = matrix[(matrix.task_key == "T5c") & (matrix.scored_member == "FP-ridge")].iloc[
        0
    ]
    assert fp.n_unit == 4  # not the six-lineage sensitivity analysis
    assert np.isnan(fp.ci_low)
    assert fp.role.startswith("diagnostic")
    assert "apriori_expect_beats_floor" not in matrix.columns


def test_incomplete_coverage_fails_loudly():
    units = pd.read_csv(
        ROOT / "results/_paper/census_unit_scores.csv", dtype={"unit": str}
    )
    with pytest.raises(ValueError):
        validate_units(units.iloc[1:])
