"""End-to-end execution of one (split, baseline) evaluation.

Flow: applicability gate -> build and audit split -> fit -> predict -> score.
Returns one raw-run row. Runtime eligibility is not final-panel admission, and
the membership audit does not certify upstream preprocessing independence.
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


def _panel_masked(spec, dataset) -> bool:
    """Do the genes outside the released checkpoint vocabulary come out of this cell's metric?

    Only where a checkpoint-bound model is on the roster: the unseen-gene and unseen-knockout
    splits. The Frangieh protein readout shares the C4 split name but carries a 20-marker panel
    that no such model is scored on, so masking it would drop genes for no reason.
    """
    if dataset == "frangieh_protein":
        return False
    name = str(getattr(spec, "name", "") or "")
    return name.startswith("C3_true_lo_gene") or name.startswith("C4_modality_lo_ko")

def run_job(
    cs: CellSet,
    spec: SplitSpec,
    adapter: BaselineAdapter,
    *,
    seed: int = 0,
    # Compatibility argument; the authoritative side input is always cs.side_info.
    side_info: dict | None = None,
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
        return {
            "baseline": adapter.name,
            "split": spec.name,
            "action": action.value,
            "ran": False,
        }

    np.random.seed(seed)

    split = build_split(cs, spec)
    audit = audit_split(
        cs, split
    )  # raises LeakError on any violation (leaks must NEVER be swallowed)

    # A heavy baseline (own conda env / GPU) may fail at runtime; record it as `failed` per the
    # 4-status taxonomy instead of crashing the whole sweep. LeakError above is intentionally NOT
    # caught — a leak is a hard stop.
    try:
        adapter.fit(cs, split, side_info=cs.side_info)
        pred = adapter.predict(cs, split, side_info=cs.side_info)
    except Exception as e:  # noqa: BLE001
        return {
            "baseline": adapter.name,
            "family": getattr(adapter, "family", "?"),
            "split": spec.name,
            "registry_task": registry_task,
            "action": "failed",
            "headline_eligible": False,
            "seed": seed,
            "ran": False,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }

    test_X = cs.X[split.test_idx]
    excl = cs.gene_index(exclude_genes) if exclude_genes else None
    # The C2 response panel is selected from training cells, then excluded from
    # Pearson-Δ. This metric mask does not remove those genes from model inputs.
    n_response_genes = None
    if response_gene_fn is not None:
        rg = np.asarray(response_gene_fn(cs, split), dtype=int)
        n_response_genes = int(len(rg))
        excl = rg if excl is None else np.union1d(excl, rg)
    # The panel mask belongs to the CELL, so it is applied here rather than in a driver.
    # NATIVE_102_FINAL_REVIEW.md section 247 rules that padding output genes a model did not
    # predict with the control mean has to go, and offers "a common evaluation mask over the genes
    # actually output" in its place. Common is the load-bearing word: if one model on a cell is
    # scored on 2,000 genes and another on 1,946, the comparison is gone. Twelve scripts call
    # run_job -- run_cluster, run_c4_conditioned, foundation_c4_modality, graph_frangieh,
    # state_frangieh, cpa_frangieh, cinemaot_frangieh and the rest -- and run_obligation.py, where
    # this first went, cannot even produce the two floor members or four of the ten T3 roster
    # models. A mask enforced in one writer of twelve is a property of that writer, not of the
    # panel. Here every caller inherits it.
    if _panel_masked(spec, dataset):
        from ivcbench.eval.panel_mask import unrepresentable

        blind = unrepresentable(cs.var_names)
        if blind:
            idx = cs.gene_index(blind)
            excl = idx if excl is None else np.union1d(excl, idx)
    resp = pearson_delta(
        pred.pred_cells, test_X, pred.control_mean, split.test_strata, excl
    )
    # Secondary Pearson-Δ without the evaluation gene mask.
    # Equals the main score when no genes are excluded (non-downstream-only clusters).
    resp_incl = (
        resp
        if excl is None
        else pearson_delta(
            pred.pred_cells, test_X, pred.control_mean, split.test_strata, None
        )
    )
    dist = e_distance(
        pred.pred_cells, test_X, split.test_strata, fit_on=cs.X[split.train_idx]
    )

    # Deposit this evaluation's PREDICTION BUNDLE if IVCBENCH_PRED_DUMP=<dir> is set, so a cluster re-run
    # materialises the model-output layer in the GPU-free reproduce_eval format (predictions -> metrics).
    # dump_bundle stores the EXACT scoring inputs + the train-cloud PCA basis and never raises.
    from ..eval.bundle import dump_bundle

    _bundle_path = dump_bundle(
        os.environ.get("IVCBENCH_PRED_DUMP"),
        cluster=registry_task,
        model=adapter.name,
        split=spec.name,
        dataset=dataset,  # key the bundle filename per-dataset (C3 reuses one split across datasets)
        pred_cells=pred.pred_cells,
        test_cells=test_X,
        cell_strata=split.test_strata,
        control_mean=pred.control_mean,
        genes=cs.var_names,
        exclude_gene_idx=excl,
        fit_on=cs.X[split.train_idx],
        declined=getattr(pred, "declined", None),
    )

    # Raw-run diagnostic: average per-cell rank scores within strata. The final
    # T3/T5c analysis instead scores both population means symmetrically in
    # scripts/immune_readout_audit.py; it does not reuse this auxiliary macro.
    ctrl_idx = (
        split.inference_input_idx
        if len(split.inference_input_idx)
        else np.asarray(split.train_idx)[
            cs.obs.iloc[split.train_idx]["is_control"].to_numpy(bool)
        ]
    )
    if not len(ctrl_idx):
        raise ValueError(
            "Immune-program scoring requires an inference or training control pool"
        )
    ctrl_cells = cs.X[ctrl_idx]
    progs = dict(immune_programs or {})
    if not progs and immune_program_genes:
        progs = {"program": immune_program_genes}
    prog_corrs: dict[str, float] = {}
    for pname, pgenes in progs.items():
        gs = cs.gene_index(pgenes)
        prog_corrs[pname] = aucell_delta_corr(
            pred.pred_cells, test_X, ctrl_cells, gs, split.test_strata
        )["corr"]
    finite_programs = [v for v in prog_corrs.values() if np.isfinite(v)]
    headline_prog = float(np.mean(finite_programs)) if finite_programs else float("nan")

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
        "pearson_delta": resp["macro"],  # Axis 1, main (downstream-only) (↑)
        "pearson_delta_lo": resp_ci["lo"],
        "pearson_delta_hi": resp_ci["hi"],
        "pearson_delta_ontarget": resp_incl[
            "macro"
        ],  # Axis 1, secondary (on-target-inclusive)
        "e_distance": dist["macro"],  # Axis 2 (↓)
        "e_distance_lo": dist_ci["lo"],
        "e_distance_hi": dist_ci["hi"],
        "aucell_program_corr": (
            headline_prog
        ),  # Axis 3, mean over dataset-aware programs (↑)
    }
    if n_response_genes is not None:
        row["n_response_genes"] = (
            n_response_genes  # C2: size of the training-only response panel
        )
    row.update({f"aucell::{p}": v for p, v in prog_corrs.items()})
    # Carry the deposited bundle path so a caller can audit what was actually written -- the
    # control-mean-collapse guard in scripts/run_obligation.py reads it.
    row["pred_bundle"] = _bundle_path
    # A decline is a REFUSAL to predict, not a prediction of "this perturbation does nothing".
    # It has to survive in the RESULT ROW, not only in the bundle. The C4 protein-CITE arm dumps
    # no bundle at all -- scripts/run_c4_conditioned.py drops IVCBENCH_PRED_DUMP there so the
    # protein run cannot overwrite the RNA bundle the census reads -- so on that arm the bundle
    # was the only record of a decline and there was no bundle. Measured case: linear-shift-KOemb
    # declines 6984 of 7044 protein rows, i.e. 99.1% of its reported protein number is the control
    # mean, and nothing downstream could see that.
    _dec = getattr(pred, "declined", None)
    _n_dec = int(np.asarray(_dec, dtype=bool).sum()) if _dec is not None else 0
    row["n_declined_cells"] = _n_dec
    row["frac_declined"] = round(_n_dec / max(1, int(len(split.test_strata))), 6)
    return row
