"""Deposit consistency gate: the committed census must equal the bundle re-score.

This re-derives the headline census from the deposited prediction bundles and fails if the
committed results/_paper/cross_cluster_headline.csv has drifted from it. Keeps the paper's
numbers and what a reviewer reproduces GPU-free permanently in lockstep.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

_PRED = os.path.join(os.path.dirname(__file__), "..", "predictions")


@pytest.mark.skipif(
    not os.path.isdir(_PRED), reason="deposited prediction bundles not present"
)
def test_census_matches_bundle_rescore():
    from check_consistency import check

    problems, n_bundles, n_cells = check()
    assert not problems, "deposit drift:\n" + "\n".join(problems)
    # 기대값은 committed census 에서 읽는다 (하드코딩하면 census 확장 때마다 어긋난다)
    import csv

    committed = os.path.join(
        os.path.dirname(__file__),
        "..",
        "results",
        "_paper",
        "cross_cluster_headline.csv",
    )
    expected = sum(1 for _ in csv.DictReader(open(committed)))
    assert (
        n_cells == expected
    ), f"re-score gave {n_cells} cells, committed census has {expected}"
