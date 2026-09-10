#!/usr/bin/env python3
"""Re-score final-roster T3/T5c mean-profile program readouts without fitting.

Both sides are scored at the SAME population-mean level. Cached means of
per-cell rank scores are used only to diagnose aggregation, never as the target
for ranking mean-profile predictors. An observed-mean identity control must
have zero score error and correlation one whenever the target is estimable.
This readout does not recover within-cell heterogeneity a model may generate.
Undefined correlations stay missing, never zero. Two-stratum correlations are
reported in source data but excluded from comparative summaries.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from assemble_cross_cluster import ROOT
from ivcbench.metrics.program import aucell

PAPER = Path(ROOT) / "results/_paper"
ESTIMABLE = "estimable (descriptive)"
CONSTANT_OBSERVED = "constant observed rank-program score"
CONSTANT_PREDICTED = "constant predicted rank-program score"
SCORE_TOLERANCE = 1e-12
PROFILE_TOLERANCE = 1e-4


class MatchedScores(NamedTuple):
    """Symmetric scores and identity checks, with named rather than positional access."""

    predicted: np.ndarray
    observed: np.ndarray
    correlation: float
    status: str
    identity_correlation: float
    identity_status: str
    identity_mae: float


def program_corr(
    pred: np.ndarray,
    obs: np.ndarray,
    n_measured: int,
    constant_profiles: bool = False,
    constant_observed: bool = False,
) -> tuple[float, str]:
    """Distinguish an unidentifiable target from an unvarying prediction.

    Two-point correlations are retained for provenance but not counted as
    comparative estimates. Constant observed targets take precedence, including
    for an identity predictor.
    """
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    if (
        len(pred) != len(obs)
        or not np.isfinite(pred).all()
        or not np.isfinite(obs).all()
    ):
        raise ValueError("Unpaired or nonfinite program scores")
    if n_measured == 0:
        return np.nan, "no measured program genes"
    if len(obs) < 2:
        return np.nan, "fewer than two perturbations"
    if constant_observed or np.std(obs) < SCORE_TOLERANCE:
        return np.nan, CONSTANT_OBSERVED
    if constant_profiles or np.std(pred) < SCORE_TOLERANCE:
        return np.nan, CONSTANT_PREDICTED
    corr = float(np.corrcoef(pred, obs)[0, 1])
    status = (
        "two perturbations only; not a comparative estimate"
        if len(obs) == 2
        else ESTIMABLE
    )
    return corr, status


def matched_profile_scores(
    pred: np.ndarray, observed: np.ndarray, idx: np.ndarray
) -> MatchedScores:
    """Symmetric scoring and an explicit identity control on the actual target."""
    pred, observed = np.asarray(pred), np.asarray(observed)
    if pred.shape != observed.shape or pred.ndim != 2:
        raise ValueError(
            "Predicted and observed mean profiles must have the same shape"
        )
    predicted_scores, target_scores = aucell(pred, idx), aucell(observed, idx)
    predicted_constant = bool(np.ptp(pred, axis=0).max() < PROFILE_TOLERANCE)
    observed_constant = bool(np.ptp(observed, axis=0).max() < PROFILE_TOLERANCE)
    corr, status = program_corr(
        predicted_scores, target_scores, len(idx), predicted_constant, observed_constant
    )
    identity = aucell(observed.copy(), idx)
    identity_corr, identity_status = program_corr(
        identity, target_scores, len(idx), observed_constant, observed_constant
    )
    identity_mae = float(np.mean(np.abs(identity - target_scores)))
    if identity_mae != 0 or (
        np.isfinite(identity_corr) and not np.isclose(identity_corr, 1)
    ):
        raise AssertionError("Observed-mean identity control failed")
    return MatchedScores(
        predicted_scores,
        target_scores,
        corr,
        status,
        identity_corr,
        identity_status,
        identity_mae,
    )


def complete_lineage_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Average OP3 correlations only when every recorded lineage is estimable.

    Never compare models by averages over different subsets of available
    lineages. The census supplies the complete four-lineage panel.
    """
    rows = []
    op3 = summary.loc[summary.task_key == "T5c"]
    for (model, program), frame in op3.groupby(["model", "program"]):
        if len(frame) != 4 or frame.unit.nunique() != 4:
            raise ValueError(f"{model}/{program}: expected four distinct OP3 lineages")
        valid = frame.correlation_status == ESTIMABLE
        rows.append(
            dict(
                model=model,
                program=program,
                n_estimable_lineages=int(valid.sum()),
                n_lineages=len(frame),
                program_corr=(
                    float(frame.program_corr.mean()) if valid.all() else np.nan
                ),
                pearson_delta=float(frame.pearson_delta.mean()),
            )
        )
    return pd.DataFrame(rows)


def main():
    units = pd.read_csv(PAPER / "census_unit_scores.csv")
    # CINEMA-OT's distributional transformation is not reconstructed from means.
    chosen = units[units.task_key.isin(["T3", "T5c"]) & (units.model != "CINEMA-OT")]
    summaries, strata_rows, manifest, targets = [], [], [], {}
    for row in chosen.itertuples():
        cache = PAPER / "program_observations" / f"{row.task_key}__{row.unit}.npz"
        bundle = Path(ROOT) / row.bundle_path
        with (
            np.load(cache, allow_pickle=False) as obs,
            np.load(bundle, allow_pickle=True) as data,
        ):
            if not np.array_equal(obs["genes"], data["genes"].astype(str)):
                raise ValueError(f"{row.model}/{row.unit}: gene mismatch")
            lookup = {str(s): i for i, s in enumerate(data["strata"])}
            order = [lookup[str(s)] for s in obs["strata"]]
            pred = data["pred_means"][order]
            observed = data["obs_means"][order]
            for j, program in enumerate(obs["programs"]):
                members = set(str(obs["measured_genes"][j]).split(";"))
                idx = np.array(
                    [k for k, gene in enumerate(obs["genes"]) if gene in members], int
                )
                matched = matched_profile_scores(pred, observed, idx)
                pred_scores, obs_scores = matched.predicted, matched.observed
                per_cell_scores = obs["obs_program_means"][:, j]
                aggregation_corr, aggregation_status = program_corr(
                    obs_scores, per_cell_scores, len(idx)
                )
                target_key = (row.task_key, str(row.unit), str(program))
                target = dict(
                    task_key=row.task_key,
                    unit=row.unit,
                    program=str(program),
                    n_strata=len(observed),
                    n_measured=len(idx),
                    mean_profile_score_sd=float(np.std(obs_scores)),
                    per_cell_score_sd=float(np.std(per_cell_scores)),
                    identity_corr=matched.identity_correlation,
                    identity_status=matched.identity_status,
                    identity_mae=matched.identity_mae,
                    aggregation_corr=aggregation_corr,
                    aggregation_status=aggregation_status,
                    aggregation_mae=float(
                        np.mean(np.abs(obs_scores - per_cell_scores))
                    ),
                )
                if target_key in targets:
                    previous = targets[target_key]
                    for field in ("mean_profile_score_sd", "aggregation_mae"):
                        if not np.isclose(previous[field], target[field], atol=1e-10):
                            raise ValueError(
                                f"Unpaired program target: {target_key}/{field}"
                            )
                targets[target_key] = target
                summaries.append(
                    dict(
                        task_key=row.task_key,
                        unit=row.unit,
                        model=row.model,
                        execution_status=row.status,
                        program=str(program),
                        n_strata=len(pred),
                        n_measured=len(idx),
                        measured_genes=";".join(obs["genes"][idx]),
                        program_corr=matched.correlation,
                        correlation_status=matched.status,
                        pearson_delta=row.pearson_delta,
                        obs_program_sd=float(np.std(obs_scores)),
                        pred_program_sd=float(np.std(pred_scores)),
                        identity_corr=matched.identity_correlation,
                        identity_mae=matched.identity_mae,
                        target_level="rank score of observed mean expression",
                        observed_zero_fraction=float(
                            np.average(
                                obs["obs_zero_fraction"][:, j], weights=obs["n_cells"]
                            )
                        ),
                        bundle_path=row.bundle_path,
                        observation_cache=str(cache.relative_to(ROOT)),
                    )
                )
                for i, s in enumerate(obs["strata"]):
                    strata_rows.append(
                        dict(
                            task_key=row.task_key,
                            unit=row.unit,
                            model=row.model,
                            program=str(program),
                            stratum=str(s),
                            n_cells=int(obs["n_cells"][i]),
                            predicted_program_score=float(pred_scores[i]),
                            observed_program_score=float(obs_scores[i]),
                            observed_per_cell_score_mean=float(per_cell_scores[i]),
                            control_program_score=float(
                                aucell(data["control_mean"][None, :], idx)[0]
                            ),
                        )
                    )
        manifest.append(
            dict(
                bundle=row.bundle_path,
                bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
                cache=str(cache.relative_to(ROOT)),
                cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
            )
        )
    summary = pd.DataFrame(summaries)
    scores = pd.DataFrame(strata_rows)
    summary.to_csv(PAPER / "immune_readout_summary.csv", index=False)
    scores.to_csv(PAPER / "immune_readout_per_stratum.csv", index=False)
    target_frame = pd.DataFrame(targets.values()).sort_values(
        ["task_key", "unit", "program"]
    )
    target_frame.to_csv(PAPER / "immune_readout_target_validation.csv", index=False)
    macro = complete_lineage_summary(summary)
    macro.to_csv(PAPER / "immune_readout_op3_macro.csv", index=False)
    (PAPER / "immune_readout_provenance.json").write_text(
        json.dumps(
            dict(
                inputs=manifest,
                scope=(
                    "final T3 and T5c census, excluding CINEMA-OT; CPA/scGen non-native"
                    " entries absent"
                ),
                metric=(
                    "same custom AUCell-like top-5% rank score on predicted and"
                    " observed MEAN profiles; Pearson across perturbations"
                ),
                target_validation=(
                    "observed-mean identity: zero error and correlation one where"
                    " estimable; per-cell means only an aggregation diagnostic"
                ),
                caveat=(
                    "post-review estimand correction, not a new model fit;"
                    " target-constant and prediction-constant results distinguished; no"
                    " cell-heterogeneity or magnitude claim"
                ),
                identity_controls=len(target_frame),
                max_identity_mae=float(target_frame.identity_mae.max()),
                target_status=target_frame.identity_status.value_counts().to_dict(),
                counts=summary.correlation_status.value_counts().to_dict(),
            ),
            indent=2,
        )
        + "\n"
    )
    print(summary.groupby(["task_key", "correlation_status"]).size().to_string())
    print(macro.to_string(index=False))
    print(target_frame.groupby(["task_key", "identity_status"]).size().to_string())


if __name__ == "__main__":
    main()
