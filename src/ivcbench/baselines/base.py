"""BaselineAdapter — the single interface every one of the 13 baselines wraps.

Heavy baselines (scGen, CPA/chemCPA, scGPT, UCE, CellOT, CINEMA-OT, STATE, GEARS, AttentionPert)
each live behind this ABC, shelling out to their own pinned conda env when needed. The runner only
ever sees fit() / predict(); the applicability registry decides whether a given (baseline, split)
is even invoked.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..data.schema import CellSet
from ..splits.builder import Split


@dataclass
class PredResult:
    """Predicted expression for each test cell, plus the control baseline used for Δ metrics."""
    pred_cells: np.ndarray   # (n_test, n_genes) predicted expression aligned to split.test_idx
    control_mean: np.ndarray  # (n_genes,) control baseline state (for Pearson-Δ / E-dist deltas)


class BaselineAdapter(ABC):
    name: str = "base"
    family: str = "abstract"  # simple / latent / graph / foundation / ot / hybrid
    gpu: bool = False

    @abstractmethod
    def fit(self, cs: CellSet, split: Split, side_info: dict | None = None) -> None:
        """Learn from split.train_idx ONLY. Must never read split.test_idx."""

    @abstractmethod
    def predict(self, cs: CellSet, split: Split, side_info: dict | None = None) -> PredResult:
        """Predict the held-out group's treated expression from inference-input + side-info."""

    # shared helper: control baseline state from the inference input (Phase-1: per-context)
    @staticmethod
    def _control_mean(cs: CellSet, split: Split) -> np.ndarray:
        inf = split.inference_input_idx
        if len(inf) == 0:
            return cs.X[split.train_idx][cs.obs.iloc[split.train_idx]["is_control"].to_numpy()].mean(0)
        return cs.X[inf].mean(axis=0)
