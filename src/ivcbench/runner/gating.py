"""Translate the applicability matrix into a run-time action for each (baseline, split)."""
from __future__ import annotations

from enum import Enum

from ..baselines.registry import Status, status_for


class Action(str, Enum):
    RUN_HEADLINE = "run_headline"   # applicable: run + eligible for ranking
    RUN_ADAPTED = "run_adapted"     # adapted: run only if extension implemented; reported separately
    RUN_FLOOR = "run_floor"         # not_defined: run as floor reference, excluded from ranking
    SKIP = "skip"                   # inapplicable: do not run


def decide(baseline: str, registry_task: str, adapted_implemented: bool = False) -> Action:
    st = status_for(baseline, registry_task)
    if st is Status.APPLICABLE:
        return Action.RUN_HEADLINE
    if st is Status.ADAPTED:
        return Action.RUN_ADAPTED if adapted_implemented else Action.SKIP
    if st is Status.NOT_DEFINED:
        return Action.RUN_FLOOR
    return Action.SKIP
