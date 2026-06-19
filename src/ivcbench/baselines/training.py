"""Shared training / fine-tuning conventions — identical across clusters and baseline families.

Keeping the fine-tuning protocol uniform (gene space, normalization, model-selection split, seeds)
is what lets the paper state one Methods protocol for all of C1–C5. Every trainable adapter consumes
this TrainConfig and selects models with leak_safe_val_split (validation carved from TRAIN ONLY —
never test/inference; the leak auditor already guarantees test is excluded from train).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    val_frac: float = 0.10        # model-selection split, taken from train only
    max_epochs: int = 100
    early_stop_patience: int = 10
    lr: float = 1e-3
    batch_size: int = 128
    device: str = "cuda"
    # feature space + normalization are FIXED by data.preprocess (HVG, library-log-norm); foundation
    # models map their vocabulary onto this shared HVG space so all families see identical inputs.


def leak_safe_val_split(train_idx: np.ndarray, val_frac: float = 0.10, seed: int = 0):
    """Carve (fit_idx, val_idx) from TRAIN ONLY for early stopping / hyperparameter selection.
    Never touches test or inference-input cells — those are out of train by construction (audited)."""
    train_idx = np.asarray(train_idx)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_idx)
    n_val = max(1, int(round(val_frac * len(train_idx))))
    return perm[n_val:], perm[:n_val]
