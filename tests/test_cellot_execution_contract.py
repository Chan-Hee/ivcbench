"""Validate the CellOT execution route without importing or training the GPU model."""

from types import SimpleNamespace

import pandas as pd
import pytest

from ivcbench.baselines.heavy import CellOT, CellOTC1, SubprocessAdapter


def test_cellot_uses_payload_runner_not_helper_module():
    for cls in (CellOT, CellOTC1):
        assert cls.runner == "cellot_c1_runner.py"
        assert cls.pred_key_is_group
        assert not cls.requires_gene_side


def test_cellot_refuses_unseen_intervention_before_training():
    cs = SimpleNamespace(
        obs=pd.DataFrame({"perturbation": ["control", "KO_A", "KO_B"]})
    )
    split = SimpleNamespace(train_idx=[0, 1], test_idx=[2])
    with pytest.raises(NotImplementedError, match="pooled map"):
        CellOTC1().fit(cs, split)


def test_cellot_keeps_seen_response_transfer(monkeypatch):
    cs = SimpleNamespace(
        obs=pd.DataFrame({"perturbation": ["control", "stim", "stim"]})
    )
    split = SimpleNamespace(train_idx=[0, 1], test_idx=[2])
    called = []
    monkeypatch.setattr(
        SubprocessAdapter, "fit", lambda self, *args: called.append(args)
    )
    CellOT().fit(cs, split)
    assert called == [(cs, split, None)]
