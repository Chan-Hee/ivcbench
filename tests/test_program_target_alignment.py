"""Regression controls for the estimand, not just reproduction of old numbers."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "readout_alignment_audit", SCRIPTS / "immune_readout_audit.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_identity_mean_prediction_has_perfect_estimable_score():
    observed = np.zeros((3, 40))
    observed[:, :3] = [[9, 5, 1], [5, 9, 1], [1, 9, 5]]
    _, _, corr, status, identity_corr, _, error = audit.matched_profile_scores(
        observed.copy(), observed, [0]
    )
    assert status == "estimable (descriptive)"
    assert corr == pytest.approx(1)
    assert identity_corr == pytest.approx(1)
    assert error == 0


def test_constant_observed_target_is_not_model_failure():
    observed = np.zeros((3, 40))
    observed[:, 1] = [3, 4, 5]
    prediction = observed.copy()
    prediction[:, 0] = [1, 5, 9]
    _, _, corr, status, identity_corr, _, error = audit.matched_profile_scores(
        prediction, observed, [0]
    )
    assert np.isnan(corr) and np.isnan(identity_corr)
    assert status == "constant observed rank-program score"
    assert error == 0


def test_prediction_constant_is_separate_from_target_constant():
    observed = np.zeros((3, 40))
    observed[:, :3] = [[9, 5, 1], [5, 9, 1], [1, 9, 5]]
    prediction = np.tile(observed[0], (3, 1))
    assert (
        audit.matched_profile_scores(prediction, observed, [0])[3]
        == "constant predicted rank-program score"
    )


def test_average_cell_rank_score_is_not_rank_score_of_average():
    cells = np.zeros((2, 20))
    cells[:, :2] = [[9, 0], [0, 20]]
    mean_cell_score = audit.aucell(cells, [0]).mean()
    mean_profile_score = audit.aucell(cells.mean(axis=0, keepdims=True), [0])[0]
    assert mean_cell_score == 0.5
    assert mean_profile_score == 0


def test_unpaired_profiles_rejected():
    with pytest.raises(ValueError, match="same shape"):
        audit.matched_profile_scores(np.zeros((2, 40)), np.zeros((3, 40)), [0])


def test_complete_lineage_macro_does_not_average_available_subsets():
    frame = pd.DataFrame(
        dict(
            task_key=["T5c"] * 4,
            model=["example"] * 4,
            program=["IFN"] * 4,
            unit=["B", "Mono", "NK", "T_cells"],
            correlation_status=[audit.ESTIMABLE] * 3 + [audit.CONSTANT_OBSERVED],
            program_corr=[0.2, 0.4, 0.6, np.nan],
            pearson_delta=[0.1] * 4,
        )
    )
    macro = audit.complete_lineage_summary(frame).iloc[0]
    assert macro.n_estimable_lineages == 3 and np.isnan(macro.program_corr)
    with pytest.raises(ValueError, match="four distinct"):
        audit.complete_lineage_summary(frame.iloc[:3])
