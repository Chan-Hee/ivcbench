"""Build train / test / inference-input index sets from a SplitSpec."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.schema import CellSet
from .spec import SplitSpec


@dataclass
class Split:
    spec: SplitSpec
    train_idx: np.ndarray
    test_idx: np.ndarray            # held-out group's TREATED cells (the prediction target)
    inference_input_idx: np.ndarray  # control/context cells the model is allowed to see at inference
    test_strata: np.ndarray         # stratum label per test cell (for macro-averaging)


def build_split(cs: CellSet, spec: SplitSpec) -> Split:
    obs = cs.obs
    held = set(spec.held_values)
    in_group = obs[spec.key_col].isin(held).to_numpy()
    is_ctrl = obs["is_control"].to_numpy()

    # train: everything NOT in the held-out group (the held value is fully removed)
    train_idx = np.where(~in_group)[0]
    # test: the held-out group's TREATED cells
    test_idx = np.where(in_group & ~is_ctrl)[0]

    if spec.control_inference_only:
        # inference input = the held-out group's CONTROL cells only
        inference_input_idx = np.where(in_group & is_ctrl)[0]
    else:
        # unseen-label (e.g. compound) has no controls of its own -> use control cells from
        # matched contexts. v0: all control cells (shared baseline state); Phase-1 refines to
        # per-context matching via spec.inference_context_cols.
        inference_input_idx = np.where(is_ctrl & ~in_group)[0]

    test_strata = np.array(
        [spec.stratum_key(obs.iloc[i]) for i in test_idx], dtype=object
    )
    return Split(spec, train_idx, test_idx, inference_input_idx, test_strata)
