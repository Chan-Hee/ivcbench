"""Legacy heavy-baseline placeholders.

The production heavy-baseline wrappers live in ``ivcbench.baselines.heavy`` and
``model_runners/``. This module remains for older imports and makes unavailable
adapters fail with a clear message instead of silently entering a benchmark run.
"""
from __future__ import annotations

from .base import BaselineAdapter, PredResult  # noqa: F401


class _Unimplemented(BaselineAdapter):
    repo = "external model repository"
    conda_env = "model-specific environment"

    def fit(self, cs, split, side_info=None):
        raise NotImplementedError(
            f"{self.name}: adapter is not available in this legacy stub module. "
            "Use ivcbench.baselines.heavy or the corresponding model_runners entry."
        )

    def predict(self, cs, split, side_info=None) -> PredResult:  # pragma: no cover
        raise NotImplementedError(self.name)


class ScGen(_Unimplemented):
    name, family, gpu = "scGen", "latent", True
    repo, conda_env = "theislab/scgen", "ivc-cpa"


class CPA(_Unimplemented):  # chemCPA on C5 (chemistry-aware variant)
    name, family, gpu = "CPA", "latent", True
    repo, conda_env = "theislab/cpa / chemCPA", "ivc-cpa"


class GEARS(_Unimplemented):
    name, family, gpu = "GEARS", "graph", True
    repo, conda_env = "snap-stanford/GEARS", "ivc-gears"


class AttentionPert(_Unimplemented):
    name, family, gpu = "AttentionPert", "graph", True
    repo, conda_env = "AttentionPert", "ivc-gears"


class ScGPT(_Unimplemented):
    name, family, gpu = "scGPT", "foundation", True
    repo, conda_env = "bowang-lab/scGPT", "ivc-scgpt"


class UCE(_Unimplemented):
    name, family, gpu = "UCE", "foundation", True
    repo, conda_env = "snap-stanford/UCE", "ivc-scgpt"


class CellOT(_Unimplemented):
    name, family, gpu = "CellOT", "ot", True
    repo, conda_env = "bunnech/cellot", "ivc-ot"


class CINEMAOT(_Unimplemented):
    name, family, gpu = "CINEMA-OT", "ot", True
    repo, conda_env = "vandijklab/CINEMA-OT", "ivc-ot"


class STATE(_Unimplemented):
    name, family, gpu = "STATE", "hybrid", True
    repo, conda_env = "STATE", "ivc-state"


HEAVY_BASELINES = [ScGen, CPA, GEARS, AttentionPert, ScGPT, UCE, CellOT, CINEMAOT, STATE]
