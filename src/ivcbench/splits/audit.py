"""Validate held-response membership before model fitting and scoring.

Checks disjoint train/test cells, absent held labels in training, treated-only
test cells and control-only inference inputs. The returned ``leak_free`` flag
is retained for file compatibility and certifies only these checks. Source
preprocessing, model internals and model selection require separate audits.
"""

from __future__ import annotations

import numpy as np

from ..data.schema import CellSet
from .builder import Split


class LeakError(AssertionError):
    """Raised when cell membership violates the declared held-response contract."""


def audit_split(cs: CellSet, split: Split) -> dict:
    obs = cs.obs
    spec = split.spec
    held = set(spec.held_values)
    train, test, inf = split.train_idx, split.test_idx, split.inference_input_idx

    def _check(cond, msg):
        if not cond:
            raise LeakError(f"[{spec.name}] {msg}")

    # 1. train and test are disjoint
    _check(len(np.intersect1d(train, test)) == 0, "train ∩ test is non-empty")
    # 2. the held-out key values are absent from training cells
    train_keys = set(obs.iloc[train][spec.key_col].unique())
    _check(
        train_keys.isdisjoint(held),
        f"held value(s) {held & train_keys} present in train",
    )
    # 3. test cells are treated (we predict the perturbation response, not control)
    _check(not obs.iloc[test]["is_control"].any(), "test set contains control cells")
    # 4. inference input is disjoint from test
    _check(len(np.intersect1d(inf, test)) == 0, "inference-input ∩ test is non-empty")

    if spec.control_inference_only:
        # 5. inference input must be controls of the held-out group only
        _check(
            obs.iloc[inf]["is_control"].all(),
            "control_inference_only but inference-input has treated cells",
        )
        _check(
            set(obs.iloc[inf][spec.key_col].unique()).issubset(held),
            "control_inference_only inference-input leaks outside the held-out group",
        )
        _check(
            len(np.intersect1d(inf, train)) == 0,
            "control_inference_only inference-input overlaps train",
        )
    else:
        # inference baseline = shared control context -> may overlap train (controls are not the
        # held-out label's response), but must itself contain no held-out treated cells
        _check(
            obs.iloc[inf]["is_control"].all(),
            "inference-input baseline must be control cells",
        )

    return {
        "split": spec.name,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_inference_input": int(len(inf)),
        "n_test_strata": int(len(set(split.test_strata.tolist()))),
        "held_values": sorted(held),
        "leak_free": True,
    }
