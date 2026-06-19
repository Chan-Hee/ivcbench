"""End-to-end smoke test — the GPU-free proof that the C5 "1패스" pipeline is sound."""
import numpy as np

from ivcbench.baselines.simple import SIMPLE_BASELINES
from ivcbench.clusters import c5
from ivcbench.data.schema import validate_cellset
from ivcbench.data.synth import make_op3_like
from ivcbench.runner.run import run_job


def _run_all(spec):
    cs = make_op3_like(seed=0)
    validate_cellset(cs)
    program = cs.uns["immune_program"]["immunomod_moa"]
    return {B().name: run_job(cs, spec, B(), seed=0, immune_program_genes=program)
            for B in SIMPLE_BASELINES}


def test_loct_pipeline_runs_and_audits():
    res = _run_all(c5.cross_celltype_loct("NK"))
    for r in res.values():
        assert r["ran"] and r["leak_free"]
        assert np.isfinite(r["pearson_delta"]) and np.isfinite(r["e_distance"])
    # ctrl-pred predicts no effect -> Δ≡0 -> Pearson-Δ ≈ 0 (the floor)
    assert abs(res["ctrl-pred"]["pearson_delta"]) < 0.05
    # a shift-aware baseline must beat the no-effect floor (shared immunomod axis is recoverable)
    assert res["cell-mean"]["pearson_delta"] > res["ctrl-pred"]["pearson_delta"]


def test_gating_floors_simple_baselines_on_unseen_compound():
    res = _run_all(c5.global_compound_holdout(["cpd01", "cpd02", "cpd03"]))
    # Simple family has no compound-side representation -> not_defined -> floor, excluded from ranking
    for r in res.values():
        assert r["action"] == "run_floor"
        assert r["headline_eligible"] is False
        assert r["leak_free"]


def test_gating_keeps_simple_baselines_headline_on_loct():
    res = _run_all(c5.cross_celltype_loct("NK"))
    # cross-cell-type LOCT uses a seen compound -> simple baselines are applicable (headline-eligible)
    for r in res.values():
        assert r["action"] == "run_headline"
        assert r["headline_eligible"] is True
