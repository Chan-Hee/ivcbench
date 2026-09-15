"""Undefined biological readouts must not become numerical evidence of failure."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from headline_multiplicity import adjust
from immune_readout_audit import program_corr
from ivcbench.metrics.program import aucell_delta_corr


def test_constant_prediction_is_missing():
    value, status = program_corr([0.1, 0.1, 0.1], [0.1, 0.2, 0.3], 4)
    assert np.isnan(value)
    assert "constant predicted" in status


def test_constant_observation_is_missing():
    value, status = program_corr([0.1, 0.2, 0.3], [0.1, 0.1, 0.1], 4)
    assert np.isnan(value)
    assert "constant observed" in status


def test_two_target_correlation_not_used_as_comparative_evidence():
    value, status = program_corr([0.1, 0.2], [0.4, 0.3], 4)
    assert np.isclose(value, -1)
    assert "two perturbations" in status


def test_metric_does_not_return_false_zero():
    cells = np.array([[4, 3, 2, 1], [1, 2, 3, 4]], float)
    result = aucell_delta_corr(
        cells, cells, cells, np.array([], int), np.array(["a", "b"])
    )
    assert np.isnan(result["corr"])
    assert result["status"] == "no measured program genes"


def test_multiplicity_known_values_and_order():
    bh, holm = adjust([0.109375, 0.0625, 0.03125, 0.0029326])
    np.testing.assert_allclose(bh, [0.109375, 0.0833333333333, 0.0625, 0.0117304])
    np.testing.assert_allclose(holm, [0.125, 0.125, 0.09375, 0.0117304])


def test_multiplicity_zeros_ties_and_one():
    bh, holm = adjust([0, 0.01, 0.01, 1])
    np.testing.assert_allclose(bh, [0, 0.0133333333333, 0.0133333333333, 1])
    np.testing.assert_allclose(holm, [0, 0.03, 0.03, 1])


def test_energy_distance_requires_training_transform(tmp_path):
    from ivcbench.eval.bundle import save_bundle, score_bundle

    cells = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1], [2, 3, 2]], float)
    path = tmp_path / "no_training_basis.npz"
    save_bundle(
        path,
        pred_cells=cells,
        test_cells=cells,
        cell_strata=["a", "a", "b", "b"],
        control_mean=[0, 0, 0],
        genes=["G1", "G2", "G3"],
    )
    result = score_bundle(path)
    assert np.isclose(result["pearson_delta"], 1)
    assert np.isnan(result["e_distance"])


def test_energy_distance_with_stored_training_transform(tmp_path):
    from ivcbench.eval.bundle import save_bundle, score_bundle

    cells = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1], [2, 3, 2]], float)
    path = tmp_path / "with_training_basis.npz"
    save_bundle(
        path,
        pred_cells=cells,
        test_cells=cells,
        cell_strata=["a", "a", "b", "b"],
        control_mean=[0, 0, 0],
        genes=["G1", "G2", "G3"],
        fit_on=cells + 5,
        n_pca=2,
    )
    result = score_bundle(path)
    assert np.isclose(result["e_distance"], 0)
