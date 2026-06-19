"""Heavy-baseline adapter stubs — one class per non-Simple baseline.

Each is owned by an `adapter@<family>` agent (PLAN.md §6) and wraps the native repo behind the
BaselineAdapter interface, running in its own pinned conda env. They raise until implemented; the
applicability registry + gating already route jobs to them correctly, and the Simple-4 baselines
prove the surrounding pipeline. `adapted=True` baselines must additionally implement the
side-info conditioning extension noted in the registry.
"""
from __future__ import annotations

from .base import BaselineAdapter, PredResult  # noqa: F401


class _Unimplemented(BaselineAdapter):
    repo = "TODO"
    conda_env = "TODO"

    def fit(self, cs, split, side_info=None):
        raise NotImplementedError(
            f"{self.name}: wrap {self.repo} (env: {self.conda_env}). See PLAN.md §6/§8."
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
