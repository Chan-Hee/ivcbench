"""Regression checks for the distinction between native chemCPA and CPA adaptations."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from assemble_cross_cluster import (
    CELLS,
    EXECUTION_MODEL,
    canonicalise_bundle_models,
    cell_long,
    eligible_bundle,
)
from ivcbench.eval.bundle import score_bundle


def test_native_chemcpa_is_the_compound_execution():
    native = ROOT / "predictions/C5/C5__chemCPA__C5_global_compound_holdout.npz"
    adapted = ROOT / "predictions/C5/C5_unseen_cpd__CPA__C5_global_compound_holdout.npz"
    if not native.exists() or not adapted.exists():
        pytest.skip("Historical comparison bundles are not present")
    native_score, adapted_score = score_bundle(native), score_bundle(adapted)
    assert not np.isclose(
        native_score["pearson_delta"], adapted_score["pearson_delta"], atol=1e-5
    )
    # Assert the ACTUAL gate, not one of the mechanisms behind it: the historical adaptation is
    # kept out by exact path+sha in results/_paper/withdrawn_bundles.csv, the native execution is
    # admitted. Asserting a filename pattern instead let the gate change out from under the test.
    assert not eligible_bundle(adapted)
    assert eligible_bundle(native)
    rows = canonicalise_bundle_models(pd.DataFrame([native_score]))
    cell = next(
        cell
        for cell in CELLS
        if cell["cl"] == "C5" and cell["split"] == "unseen-compound"
    )
    result = cell_long(rows, cell).set_index("model")
    assert result.loc["CPA", "pearson_delta"] == native_score["pearson_delta"]
    assert (
        rows.loc[0, "execution_model"]
        == EXECUTION_MODEL[("C5", "unseen-compound", "CPA")]
        == "chemCPA"
    )


def test_distinct_cpa_implementations_cannot_be_averaged():
    rows = pd.DataFrame(
        {"model": ["CPA", "chemCPA"], "split": ["C5_global_compound_holdout"] * 2}
    )
    with pytest.raises(ValueError, match="non-native CPA"):
        canonicalise_bundle_models(rows)


def test_historical_seen_compound_bundles_stay_withdrawn():
    """The OLD C5-LOCT CPA/scGen bundles must never re-enter the census.

    They were withdrawn as non-native. The native executions the plan calls for are a DIFFERENT
    artifact produced by a different runner, so the gate has to key on the artifact, not on the
    model name -- otherwise admitting the native scGen T5c run would drag the withdrawn one back in
    with it. That is exactly why the filename denylist became a path+sha registry.
    """
    withdrawn = ROOT / "results/_paper/withdrawn_bundles.csv"
    if not withdrawn.exists():
        pytest.skip("withdrawal registry is not present")
    listed = [
        r["bundle_path"]
        for r in pd.read_csv(withdrawn).to_dict("records")
        if "C5_LOCT__" in r["bundle_path"] and ("__CPA__" in r["bundle_path"] or "__scGen__" in r["bundle_path"])
    ]
    assert listed, "the historical C5-LOCT CPA/scGen bundles are no longer registered as withdrawn"
    for rel in listed:
        assert not eligible_bundle(ROOT / rel)


def test_context_cpa_adaptation_is_excluded_but_native_scgen_remains():
    for cell in CELLS:
        if cell["cl"] in {"C1", "C2"}:
            assert "CPA" not in cell["roster"]
            assert "scGen" in cell["roster"]


def test_raw_execution_names_are_not_relabelled_by_census_grouping():
    from sync_results_raw import _bundle_map

    rows = pd.DataFrame(
        {
            "model": ["CPA"],
            "execution_model": ["chemCPA"],
            "cluster": ["C5"],
            "split": ["C5_global_compound_holdout"],
            "pearson_delta": [0.11158957805986694],
        }
    )
    scores = _bundle_map(rows, ["C5"], [])
    assert ("C5_global_compound_holdout", "CPA") not in scores
    assert (
        scores[("C5_global_compound_holdout", "chemCPA")] == rows.pearson_delta.iloc[0]
    )


def test_pooled_cellot_map_is_not_a_conditioned_unseen_ko_entry():
    cell = next(cell for cell in CELLS if cell["cl"] == "C4")
    assert "CellOT" not in cell["roster"]
