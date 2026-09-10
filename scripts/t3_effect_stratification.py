#!/usr/bin/env python3
"""Exploratory effect-size stratification of the final T3 census.

Global quartiles give genes equal weight and hold each dataset's gene count fixed
in the bootstrap. A within-dataset weak/strong split gives datasets equal weight
and is a separate descriptive sensitivity (five datasets, no inferential CI).
Effect bins and the best member are chosen from the evaluated data; intervals
condition on these choices. Neither analysis identifies a noise-free target.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from assemble_cross_cluster import CELLS, DIAGNOSTIC_MODELS, ROOT
from census_units import (
    FLOORS,
    BOOTSTRAPS,
    BOOTSTRAP_SEED,
    build_unit_scores,
    bundle_stratum_scores,
    binding_floor,
    task_matrix,
)

PAPER = Path(ROOT) / "results/_paper"


def gene_scores(units):
    cell = next(cell for cell in CELLS if cell["cl"] == "C3")
    models = [model for model in cell["roster"] if model not in DIAGNOSTIC_MODELS]
    selected = units[(units.task_key == "T3") & units.model.isin(models + FLOORS)]
    rows = []
    for row in selected.itertuples():
        for stratum, value in bundle_stratum_scores(
            Path(ROOT) / row.bundle_path
        ).items():
            gene = stratum.split("=", 1)[1] if "=" in stratum else stratum
            rows.append(dict(dataset=row.unit, gene=gene, model=row.model, score=value))
    long = pd.DataFrame(rows)
    if long.duplicated(["dataset", "gene", "model"]).any():
        raise ValueError("Duplicate T3 gene score")
    frame = long.pivot(index=["dataset", "gene"], columns="model", values="score")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("T3 model/floor gene coverage is not paired")
    probes = pd.read_csv(Path(ROOT) / "results/C3/predictability_probe_pergene.csv")
    probes = probes[probes.hold == 10].rename(columns={"held_gene": "gene"})
    if probes.duplicated(["dataset", "gene"]).any():
        raise ValueError("Duplicate T3 effect-size record")
    probes = probes.set_index(["dataset", "gene"])
    if set(frame.index) != set(probes.index):
        raise ValueError("Effect probes and scored genes differ")
    frame = frame.join(probes[["effect_l2", "snr_raw", "n_test_cells"]]).reset_index()
    if not np.isfinite(frame[["effect_l2", "snr_raw"]].to_numpy()).all():
        raise ValueError("Invalid effect-size probe")
    matrix = task_matrix(units, "T3")
    floor = binding_floor(matrix)
    fixed_model = max(models, key=lambda model: matrix[model].mean())
    return frame, models, floor, fixed_model


def conditional_interval(frame, model, floor):
    """Resample genes within each represented dataset; dataset composition is fixed."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    total = np.zeros(BOOTSTRAPS)
    for _, group in frame.groupby("dataset", sort=True):
        margins = (group[model] - group[floor]).to_numpy(float)
        total += margins[
            rng.integers(0, len(margins), size=(BOOTSTRAPS, len(margins)))
        ].sum(axis=1)
    return np.percentile(total / len(frame), [2.5, 97.5])


def summarize(frame, models, floor, fixed_model, scheme, stratum):
    dataset_weighted = scheme == "within-dataset halves"
    score_table = (
        frame.groupby("dataset")[models + FLOORS].mean() if dataset_weighted else frame
    )
    best = max(models, key=lambda model: score_table[model].mean())
    result = dict(
        scheme=scheme,
        stratum=stratum,
        n_genes=len(frame),
        n_datasets=frame.dataset.nunique(),
        dataset_composition=";".join(
            f"{dataset}:{len(group)}" for dataset, group in frame.groupby("dataset")
        ),
        weighting="equal dataset" if dataset_weighted else "equal gene",
        effect_l2_median=float(frame.effect_l2.median()),
        snr_median=float(frame.snr_raw.median()),
        binding_floor=floor,
        floor_mean=float(score_table[floor].mean()),
        best_model=best,
        best_model_mean=float(score_table[best].mean()),
        best_gap_vs_floor=float((score_table[best] - score_table[floor]).mean()),
        fixed_model=fixed_model,
        fixed_model_mean=float(score_table[fixed_model].mean()),
        fixed_gap_vs_floor=float(
            (score_table[fixed_model] - score_table[floor]).mean()
        ),
        genes_below_floor=f"{int((frame[best] < frame[floor]).sum())}/{len(frame)}",
        fixed_genes_below_floor=(
            f"{int((frame[fixed_model] < frame[floor]).sum())}/{len(frame)}"
        ),
    )
    if dataset_weighted:
        result.update(
            inference="five-dataset range; descriptive",
            gap_ci_lo=np.nan,
            gap_ci_hi=np.nan,
            fixed_ci_lo=np.nan,
            fixed_ci_hi=np.nan,
        )
    else:
        lo, hi = conditional_interval(frame, best, floor)
        flo, fhi = conditional_interval(frame, fixed_model, floor)
        result.update(
            inference="conditional within-dataset gene bootstrap; unadjusted",
            gap_ci_lo=float(lo),
            gap_ci_hi=float(hi),
            fixed_ci_lo=float(flo),
            fixed_ci_hi=float(fhi),
        )
    margins = score_table[best] - score_table[floor]
    result.update(margin_min=float(margins.min()), margin_max=float(margins.max()))
    for model in models:
        result[model + "_mean"] = float(score_table[model].mean())
        result[model + "_gap"] = float((score_table[model] - score_table[floor]).mean())
    return result


def main():
    units = build_unit_scores()
    frame, models, floor, fixed = gene_scores(units)
    cuts = np.quantile(frame.effect_l2, [0.25, 0.5, 0.75])
    frame["global_quartile"] = np.searchsorted(cuts, frame.effect_l2, side="right") + 1
    # Each dataset contributes to both halves, including datasets with only two
    # held genes. Stable gene-name tie breaking makes the partition reproducible.
    frame["within_dataset_half"] = ""
    for _, group in frame.groupby("dataset"):
        order = group.sort_values(["effect_l2", "gene"]).index
        frame.loc[order[: len(order) // 2], "within_dataset_half"] = "weaker half"
        frame.loc[order[len(order) // 2 :], "within_dataset_half"] = "stronger half"
    records = []
    for quartile in range(1, 5):
        subset = frame[frame.global_quartile == quartile]
        records.append(
            summarize(
                subset, models, floor, fixed, "global effect quartiles", f"Q{quartile}"
            )
        )
    for half in ("weaker half", "stronger half"):
        records.append(
            summarize(
                frame[frame.within_dataset_half == half],
                models,
                floor,
                fixed,
                "within-dataset halves",
                half,
            )
        )
    result = pd.DataFrame(records)
    result.to_csv(PAPER / "t3_effect_stratification.csv", index=False)
    frame.to_csv(PAPER / "t3_effect_gene_scores.csv", index=False)
    counts = dict(
        n_genes=len(frame),
        n_datasets=frame.dataset.nunique(),
        models=models,
        binding_floor=floor,
        fixed_model=fixed,
        fixed_model_genes_below=int((frame[fixed] < frame[floor]).sum()),
        no_conditioned_model_above=int(
            (frame[models].max(axis=1) <= frame[floor]).sum()
        ),
        fixed_floor_selection="highest five-dataset census macro-average",
        bootstrap=(
            "4,000 within-dataset gene resamples, seed 0; conditional on observed"
            " effect bins and selected member"
        ),
        limitation=(
            "No dataset-level replication inference; observed-effect selection is"
            " noisy; neither statistic proves target learnability"
        ),
    )
    (PAPER / "t3_effect_stratification_provenance.json").write_text(
        json.dumps(counts, indent=2) + "\n"
    )
    print(json.dumps(counts, indent=2))
    print(
        result[
            [
                "scheme",
                "stratum",
                "n_genes",
                "n_datasets",
                "floor_mean",
                "best_model",
                "best_gap_vs_floor",
                "gap_ci_lo",
                "gap_ci_hi",
                "fixed_gap_vs_floor",
                "genes_below_floor",
                "margin_min",
                "margin_max",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
