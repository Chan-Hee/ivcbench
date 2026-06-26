"""Deposit consistency gate: the committed census must equal the bundle re-score.

This re-derives the 35-cell headline from the deposited prediction bundles and fails if the
committed results/_paper/cross_cluster_headline.csv has drifted from it. Keeps the paper's
numbers and what a reviewer reproduces GPU-free permanently in lockstep.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

_PRED = os.path.join(os.path.dirname(__file__), "..", "predictions")


@pytest.mark.skipif(not os.path.isdir(_PRED), reason="deposited prediction bundles not present")
def test_census_matches_bundle_rescore():
    from check_consistency import check
    problems, n_bundles, n_cells = check()
    assert not problems, "deposit drift:\n" + "\n".join(problems)
    assert n_cells == 35
