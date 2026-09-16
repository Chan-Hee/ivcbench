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
    # The native path was hardcoded to predictions/C5/..., which has itself been withdrawn since
    # 2026-09-14 as superseded by the v2_native re-run -- so the test asserted that a bundle the
    # census refuses is admitted. Read the path the census actually uses.
    manifest = pd.read_csv(ROOT / "results/_paper/census_bundle_manifest.csv")
    paths = set(manifest.loc[(manifest.task == "T5u") & (manifest.model == "CPA"), "bundle_path"])
    assert len(paths) == 1, f"expected one T5u/CPA bundle in the manifest, found {sorted(paths)}"
    native = ROOT / paths.pop()
    adapted = ROOT / "predictions/C5_unseen_cpd__CPA__C5_global_compound_holdout.npz"
    if not adapted.exists():
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
    """The historical latent-mean-shift CPA adaptation must not be what the C1/C2 cells report.

    'CPA is absent from the roster' was how that was checked when the only context CPA execution
    WAS the adaptation. The runner was since rewritten to the public CPA.predict, so CPA is on the
    roster legitimately and the old assertion inverted. What has to stay true is that the
    adaptation's own bundles are withdrawn and none of them reaches the census.
    """
    withdrawn = pd.read_csv(ROOT / "results/_paper/withdrawn_bundles.csv")
    historical = [r for r in withdrawn.to_dict("records")
                  if "__CPA__" in r["bundle_path"]
                  and ("C1_LOCT__" in r["bundle_path"] or "C2" in r["bundle_path"])]
    assert historical, "the historical context-CPA bundles are no longer registered as withdrawn"
    manifest = set(pd.read_csv(ROOT / "results/_paper/census_bundle_manifest.csv").bundle_path)
    for row in historical:
        assert row["bundle_path"] not in manifest, f"withdrawn bundle in census: {row}"
    for cell in CELLS:
        if cell["cl"] in {"C1", "C2"}:
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
