"""Independent numerical and integration controls for target-matched diagnostics."""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_target_repeatability import disjoint_halves, held_label, correlation
from assemble_target_diagnostics import describe_precision


@pytest.mark.parametrize(
    "lo,hi,expected",
    [
        (np.nan, np.nan, "descriptive only"),
        (-0.1, 0.1, "unresolved difference"),
        (0.0, 0.1, "unresolved difference"),
        (0.1, 0.2, "positive conditional interval"),
        (-0.2, -0.1, "negative conditional interval"),
    ],
)
def test_precision_labels_distinguish_estimation_from_inference(lo, hi, expected):
    row = pd.Series(
        dict(ci_lo=lo, ci_hi=hi, n_units=4 if pd.isna(lo) else 8, mde_80=0.1)
    )
    category, statement = describe_precision(row)
    assert category == expected
    if pd.isna(lo):
        assert "0.125" in statement and "no inferential power/MDE" in statement
    else:
        assert "not prospective Wilcoxon power" in statement


@pytest.mark.parametrize("n", [6, 7, 100])
def test_partitions_are_balanced_disjoint_and_deterministic(n):
    a, b = disjoint_halves(n, np.random.default_rng(17))
    assert len(a) == len(b) == n // 2
    assert not set(a) & set(b)
    c, d = disjoint_halves(n, np.random.default_rng(17))
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(b, d)


def test_undersampled_and_unlabelled_targets_are_rejected():
    with pytest.raises(ValueError):
        disjoint_halves(5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        held_label("unlabelled_target")


def test_compound_stratum_decoding_preserves_name():
    assert (
        held_label("perturbation=ABT-199 (GDC-0199)|cell_type_coarse=T cells")
        == "ABT-199 (GDC-0199)"
    )
    assert held_label("perturbation=x=y") == "x=y"


def test_shared_control_can_create_spurious_repeatability():
    # Orthogonal half-sample noise without a true response. Reusing a noisy
    # baseline induces a positive correlation despite no repeatable target.
    treated_a = np.array([1.0, -1.0, 0.0, 0.0])
    treated_b = np.array([0.0, 0.0, 1.0, -1.0])
    control = np.array([2.0, 2.0, -2.0, -2.0])
    assert correlation(treated_a, treated_b) == pytest.approx(0)
    assert correlation(treated_a - control, treated_b - control) > 0.8


def test_recorded_program_targets_pass_identity_control():
    target = pd.read_csv(ROOT / "results/_paper/immune_readout_target_validation.csv")
    assert (
        len(target) == 37
        and not target.duplicated(["task_key", "unit", "program"]).any()
    )
    assert (target.identity_mae == 0).all()
    variable = target.identity_corr.notna()
    assert variable.sum() == 12
    np.testing.assert_allclose(target.loc[variable, "identity_corr"], 1, atol=1e-12)
    assert target.loc[target.task_key == "T3", "identity_corr"].isna().sum() == 22


def test_every_contrast_has_matched_target_and_precision_statement():
    paper = ROOT / "results/_paper"
    panel = pd.read_csv(paper / "panel_precision_attenuation.csv")
    # The production code carried this literal too and was de-hardcoded to EXPECTED_CENSUS_CELLS;
    # the test kept the 47 and went red, so it stopped guarding the 1:1 panel-to-census mapping.
    from assemble_cross_cluster import EXPECTED_CENSUS_CELLS

    assert len(panel) == EXPECTED_CENSUS_CELLS
    assert not panel.duplicated(["task_key", "model"]).any()
    census = pd.read_csv(paper / "census_uncertainty.csv")
    assert set(zip(panel.task_key, panel.model)) == set(zip(census.task_key, census.model))
    for column in ["power_statement", "attenuation_statement"]:
        assert panel[column].str.len().gt(100).all()
    assert (panel.n_units == panel.n_units_target).all()
    assert (panel.n_estimable_strata <= panel.n_strata).all()
    provenance = json.loads(
        (paper / "target_repeatability_provenance.json").read_text()
    )
    assert all(p["independent_control_halves"] for p in provenance["matches"])
    assert all(
        max(p["observed_max_difference"], p["control_max_difference"]) <= 1e-4
        for p in provenance["matches"]
    )


def test_soskic_supplied_coordinates_are_not_raw_expression():
    evidence = pd.read_csv(ROOT / "results/_paper/soskic_input_space.csv")
    assert len(evidence) == 2
    assert not evidence.raw_or_count_layer_present.any()
    assert (evidence.maximum == 10).all()
    assert (evidence.minimum < 0).all()
