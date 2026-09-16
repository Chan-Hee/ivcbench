"""Artifact-selection regression without importing STATE or loading cell data."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_runners"))
from state_output import prediction_path


def test_real_file_sorts_after_prediction_but_is_never_selected(tmp_path):
    pred, real = tmp_path / "adata_pred.h5ad", tmp_path / "adata_real.h5ad"
    pred.touch()
    real.touch()
    assert sorted(tmp_path.glob("*.h5ad"))[-1] == real  # the historical defect
    assert prediction_path(tmp_path) == pred


def test_missing_prediction_never_falls_back_to_query(tmp_path):
    (tmp_path / "adata_real.h5ad").touch()
    with pytest.raises(RuntimeError, match="exactly one"):
        prediction_path(tmp_path)


def test_multiple_checkpoints_require_explicit_selection(tmp_path):
    for label in ["epoch0", "epoch1"]:
        folder = tmp_path / label
        folder.mkdir()
        (folder / "adata_pred.h5ad").touch()
    with pytest.raises(RuntimeError, match="ambiguous"):
        prediction_path(tmp_path)


def test_invalid_state_executions_are_not_census_entries():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from assemble_cross_cluster import CELLS

    # "STATE remains evaluated on T1/T2 only" was the 9-September roster. STATE was re-run on
    # T3, T4, T5c and T5u through the corrected path and admitted. What this test guards is that
    # an INVALID execution never becomes a census entry, so check that against the withdrawal
    # registry rather than freezing the roster.
    from assemble_cross_cluster import census_metadata_rows

    state_cells = {(c["task_id"], c["model"]) for c in census_metadata_rows()
                   if c["model"] == "STATE"}
    assert state_cells, "STATE reports no census cell at all"
    manifest = pd.read_csv(
        Path(__file__).resolve().parents[1] / "results/_paper/census_bundle_manifest.csv"
    )
    from assemble_cross_cluster import eligible_bundle

    for path in manifest.loc[manifest.model == "STATE", "bundle_path"].unique():
        assert eligible_bundle(Path(path)), f"a withdrawn bundle reached the census: {path}"
