"""End-to-end execution of one (split, baseline) evaluation.

Flow:  build split -> AUDIT (hard gate) -> applicability gating -> fit (train only) -> predict ->
4-axis metrics. Returns a flat result row ready to concatenate into the per-cluster results table.
"""
from __future__ import annotations

import os

import numpy as np

from ..baselines.base import BaselineAdapter
from ..data.schema import CellSet
from ..metrics.distribution import e_distance
from ..metrics.program import aucell_delta_corr
from ..metrics.response import pearson_delta
from ..metrics.stats import bootstrap_ci
from ..splits.audit import audit_split
from ..splits.builder import build_split
from ..splits.spec import SplitSpec
from .gating import Action, decide


def run_job(
    cs: CellSet,
    spec: SplitSpec,
    adapter: BaselineAdapter,
    *,
    seed: int = 0,
    immune_program_genes: list[str] | None = None,
    immune_programs: dict[str, list[str]] | None = None,
    exclude_genes: list[str] | None = None,
    response_gene_fn=None,
    adapted_implemented: bool = False,
    dataset: str | None = None,
) -> dict:
    registry_task = spec.registry_task or spec.name
    action = decide(adapter.name, registry_task, adapted_implemented)
    if action is Action.SKIP:
        return {"baseline": adapter.name, "split": spec.name, "action": action.value, "ran": False}

    np.random.seed(seed)

    split = build_split(cs, spec)
    audit = audit_split(cs, split)  # raises LeakError on any violation (leaks must NEVER be swallowed)

    # A heavy baseline (own conda env / GPU) may fail at runtime; record it as `failed` per the
    # 4-status taxonomy instead of crashing the whole sweep. LeakError above is intentionally NOT
    # caught — a leak is a hard stop.
    try:
        adapter.fit(cs, split, side_info=cs.side_info)
        pred = adapter.predict(cs, split, side_info=cs.side_info)
    except Exception as e:  # noqa: BLE001
        return {"baseline": adapter.name, "family": getattr(adapter, "family", "?"),
                "split": spec.name, "registry_task": registry_task, "action": "failed",
                "headline_eligible": False, "seed": seed, "ran": False,
                "error": f"{type(e).__name__}: {str(e)[:200]}"}

    test_X = cs.X[split.test_idx]
    excl = cs.gene_index(exclude_genes) if exclude_genes else None
    # response_gene_fn (C2 donor-LODO): leak-safe TRAINING-only response-gene panel. The Pearson-Δ is
    # computed on the genes OUTSIDE that panel — i.e. the panel is EXCLUDED (the exact bespoke Soskic
    # rule: pearson_delta(..., exclude_genes=response_genes)), so the strongly stimulation-driven
    # response genes don't dominate the direction-recovery score. Selected from the train fold only
    # (never seen by any model), so it is a metric choice, not a leak. n_response_genes is recorded.
    n_response_genes = None
    if response_gene_fn is not None:
        rg = np.asarray(response_gene_fn(cs, split), dtype=int)
        n_response_genes = int(len(rg))
        excl = rg if excl is None else np.union1d(excl, rg)
    resp = pearson_delta(pred.pred_cells, test_X, pred.control_mean, split.test_strata, excl)
    # Secondary, on-target-inclusive Pearson-Δ (perturbed gene NOT excluded) — Supp Table S3.
    # Equals the main score when no genes are excluded (non-downstream-only clusters).
    resp_incl = (resp if excl is None
                 else pearson_delta(pred.pred_cells, test_X, pred.control_mean, split.test_strata, None))
    dist = e_distance(pred.pred_cells, test_X, split.test_strata, fit_on=cs.X[split.train_idx])

    # Deposit this evaluation's PREDICTION BUNDLE if IVCBENCH_PRED_DUMP=<dir> is set, so a cluster re-run
    # materialises the model-output layer in the GPU-free reproduce_eval format (predictions -> metrics).
    # dump_bundle stores the EXACT scoring inputs + the train-cloud PCA basis and never raises.
    from ..eval.bundle import dump_bundle
    dump_bundle(os.environ.get("IVCBENCH_PRED_DUMP"), cluster=registry_task, model=adapter.name, split=spec.name,
                dataset=dataset,  # key the bundle filename per-dataset (C3 reuses one split across datasets)
                pred_cells=pred.pred_cells, test_cells=test_X, cell_strata=split.test_strata,
                control_mean=pred.control_mean, genes=cs.var_names, exclude_gene_idx=excl,
                fit_on=cs.X[split.train_idx])

    # Immune-program axis (Axis 3): dataset-aware, one AUCell-Δ correlation per program. The headline
    # aucell_program_corr is the mean over programs; per-program values populate panel (b)/Supp S3.
    ctrl_cells = cs.X[split.inference_input_idx] if len(split.inference_input_idx) else test_X
    progs = dict(immune_programs or {})
    if not progs and immune_program_genes:
        progs = {"program": immune_program_genes}
    prog_corrs: dict[str, float] = {}
    for pname, pgenes in progs.items():
        gs = cs.gene_index(pgenes)
        prog_corrs[pname] = aucell_delta_corr(pred.pred_cells, test_X, ctrl_cells, gs,
                                              split.test_strata)["corr"]
    headline_prog = float(np.nanmean(list(prog_corrs.values()))) if prog_corrs else float("nan")

    # Runner-level 95% bootstrap CI for THIS result row, resampling the per-stratum macro scores.
    # These are per-row descriptive CIs, NOT the final paper inferential CIs: the headline donor /
    # lineage / dataset / compound claims are re-bootstrapped over their biological unit (with seeds
    # collapsed within a unit) by the bespoke assembly scripts. Meaningful even for deterministic
    # baselines where the model seeds are identical.
    resp_ci = bootstrap_ci(list(resp["per_stratum"].values()), seed=seed)
    dist_ci = bootstrap_ci(list(dist["per_stratum"].values()), seed=seed)

    row = {
        "baseline": adapter.name,
        "family": adapter.family,
        "split": spec.name,
        "registry_task": registry_task,
        "action": action.value,
        "headline_eligible": action is Action.RUN_HEADLINE,
        "seed": seed,
        "ran": True,
        "leak_free": audit["leak_free"],
        "n_train": audit["n_train"],
        "n_test": audit["n_test"],
        "n_test_strata": audit["n_test_strata"],
        "pearson_delta": resp["macro"],          # Axis 1, main (downstream-only) (↑)
        "pearson_delta_lo": resp_ci["lo"],
        "pearson_delta_hi": resp_ci["hi"],
        "pearson_delta_ontarget": resp_incl["macro"],  # Axis 1, secondary (on-target-inclusive)
        "e_distance": dist["macro"],             # Axis 2 (↓)
        "e_distance_lo": dist_ci["lo"],
        "e_distance_hi": dist_ci["hi"],
        "aucell_program_corr": headline_prog,    # Axis 3, mean over dataset-aware programs (↑)
    }
    if n_response_genes is not None:
        row["n_response_genes"] = n_response_genes   # C2: size of the training-only response panel
    row.update({f"aucell::{p}": v for p, v in prog_corrs.items()})  # per-program (panel b / Supp S3)
    return row
