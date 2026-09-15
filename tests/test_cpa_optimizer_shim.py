"""CPA's chemical projection must end up in an optimizer — S2 regression.

cpa-tools 0.8.8 creates pert_network.pert_transformation (the nn.Linear that maps the frozen
molecular embedding into the latent space) but collects only encoder/decoder/pert_embedding/
covars/dosers/adversaries in configure_optimizers. The projection therefore receives gradients and
never a step, so chemistry reaches the decoder through a fixed RANDOM linear map. One deposited
cell came from that path: CPA x T5u (execution_model chemCPA).
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class _P:
    def __init__(self, n):
        self.numel_ = n
        self.requires_grad = True

    def numel(self):
        return self.numel_


class _Opt:
    def __init__(self, params):
        self.param_groups = [{"params": list(params)}]


def _fake_cpa_task(projection_params, already_in_optimizer):
    """A stand-in for CPATrainingPlan with cpa-tools' exact optimizer membership."""
    other = [_P(10)]
    inc = list(projection_params) if already_in_optimizer else []

    class Plan:
        def __init__(self):
            net = types.SimpleNamespace(pert_transformation=None)
            if projection_params is not None:
                net.pert_transformation = types.SimpleNamespace(
                    parameters=lambda: iter(projection_params)
                )
            self.module = types.SimpleNamespace(pert_network=net)

        def configure_optimizers(self):
            return [_Opt(other + inc)]

    return Plan


def _install_on(Plan):
    import cpa_optimizer_shim as shim

    mod = types.ModuleType("cpa._task")
    mod.CPATrainingPlan = Plan
    sys.modules["cpa._task"] = mod
    sys.modules.setdefault("cpa", types.ModuleType("cpa"))
    Plan._ivcbench_optimizer_shim = False
    return shim.install()


def test_the_projection_is_added_when_upstream_omits_it():
    proj = [_P(128), _P(8)]
    Plan = _fake_cpa_task(proj, already_in_optimizer=False)
    _install_on(Plan)
    opts = Plan().configure_optimizers()
    got = {id(p) for g in opts[0].param_groups for p in g["params"]}
    assert all(id(p) in got for p in proj), "the chemical projection must reach an optimizer"


def test_nothing_is_added_when_it_is_already_there():
    proj = [_P(128)]
    Plan = _fake_cpa_task(proj, already_in_optimizer=True)
    _install_on(Plan)
    opts = Plan().configure_optimizers()
    n = sum(len(g["params"]) for g in opts[0].param_groups)
    assert n == 2, "a future release that fixes this upstream must not get duplicate parameters"


def test_categorical_cpa_is_untouched():
    Plan = _fake_cpa_task(None, already_in_optimizer=False)   # no pert_transformation at all
    _install_on(Plan)
    opts = Plan().configure_optimizers()
    assert sum(len(g["params"]) for g in opts[0].param_groups) == 1
