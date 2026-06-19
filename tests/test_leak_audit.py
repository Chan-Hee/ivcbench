"""The leak auditor is the v5 guardrail — these tests pin its contract."""
import numpy as np
import pytest

from ivcbench.clusters import c5
from ivcbench.data.synth import make_op3_like
from ivcbench.splits.audit import LeakError, audit_split
from ivcbench.splits.builder import build_split


def test_loct_split_is_leak_free():
    cs = make_op3_like(seed=1)
    split = build_split(cs, c5.cross_celltype_loct("NK"))
    report = audit_split(cs, split)
    assert report["leak_free"] is True
    assert report["n_test"] > 0 and report["n_train"] > 0


def test_compound_holdout_is_leak_free():
    cs = make_op3_like(seed=1)
    held = cs.uns["compounds"][:3]
    split = build_split(cs, c5.global_compound_holdout(held))
    report = audit_split(cs, split)
    assert report["leak_free"] is True
    # held compounds must be entirely absent from train
    train_perts = set(cs.obs.iloc[split.train_idx]["perturbation"].unique())
    assert train_perts.isdisjoint(set(held))


def test_auditor_catches_injected_leak():
    cs = make_op3_like(seed=1)
    split = build_split(cs, c5.cross_celltype_loct("NK"))
    # inject leakage: a test (held-lineage treated) cell sneaks into train
    split.train_idx = np.append(split.train_idx, split.test_idx[0])
    with pytest.raises(LeakError):
        audit_split(cs, split)


def test_auditor_catches_treated_in_inference_input():
    cs = make_op3_like(seed=1)
    split = build_split(cs, c5.cross_celltype_loct("NK"))
    # control_inference_only violated: a treated cell in the inference input
    split.inference_input_idx = np.append(split.inference_input_idx, split.test_idx[0])
    with pytest.raises(LeakError):
        audit_split(cs, split)
