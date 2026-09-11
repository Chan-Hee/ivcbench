"""Execution notes derived from the current run's manifest and split objects.

These operational records deliberately avoid a second manuscript template.
Dataset methods, execution status and scientific inference belong to the
curated submission sources, not to a raw runner's applicability flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from ..splits.spec import SplitSpec


def framework_methods(manifest: dict) -> str:
    """Describe the runtime contract without promising upstream independence."""
    versions = ", ".join(
        f"{name} {version}"
        for name, version in manifest.get("packages", {}).items()
        if version not in (None, "absent")
    )
    return f"""# Shared execution notes

These notes describe the generic runner, not the final manuscript Methods.
Seeds requested: {manifest.get('seeds', [])}.
Software: {versions or 'see manifest.json'}.

## Split and fitting contract

The runner constructs a split, checks train/test membership and control-only
inference inputs, then fits and predicts through the selected adapter.
The auditor's `leak_free` field certifies these membership checks only; it does
not inspect source-study preprocessing, model internals or model selection.
Loader-specific QC and feature selection may precede splitting. Soskic inputs
are supplied in condition-specific covariate-regressed, scaled and clipped
coordinates; further standardization cannot recover unprocessed activation.

## Scores and aggregation

Pearson-Δ compares predicted and observed mean shifts relative to the same
control. Any excluded genes are recorded in the prediction bundle. Energy
distance uses cell clouds in a PCA space fitted to training cells. The runner's
program metric averages per-cell rank scores within strata; the final T3/T5c
analysis instead scores both population means with the same rank function.
Undefined program correlations remain missing in both workflows.

Runner-level intervals resample strata within a result row. They are not the
submission's paired model–reference intervals, and no final-panel multiplicity
correction is inferred here. Raw applicability fields are execution controls,
not native/adapted/diagnostic classifications.

## Reproduction boundaries

`results_raw.csv` and `manifest.json` record completed, skipped and failed jobs.
Mean-profile bundles support Pearson-Δ replay, but cannot reconstruct cell
clouds. The current submission uses the curated census and analytical summaries
under `results/_paper/`; its document source is in `submission/`. Raw fitting
requires the corresponding data access and model environments.
"""


def _split_sizes(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Take recorded sizes from completed rows, including an all-failed run."""
    columns = [
        column for column in ("n_train", "n_test", "n_test_strata") if column in raw_df
    ]
    if "ran" not in raw_df or "split" not in raw_df:
        return pd.DataFrame(columns=["split", *columns])
    return (
        raw_df.loc[raw_df["ran"].eq(True)]
        .groupby("split", as_index=False)[columns]
        .first()
    )


def cluster_methods(
    cluster: str,
    raw_df: pd.DataFrame,
    specs: Sequence[SplitSpec],
    manifest: dict,
) -> str:
    """Report supplied split metadata, without inventing data or compute costs."""
    sizes = _split_sizes(raw_df).set_index("split")
    blocks = []
    for spec in specs:
        recorded = "no completed size record"
        if spec.name in sizes.index:
            recorded = ", ".join(
                f"{key} = {int(value)}"
                for key, value in sizes.loc[spec.name].items()
                if pd.notna(value)
            )
        regime = (
            "held-context controls"
            if spec.control_inference_only
            else "pooled controls and the required intervention-side input"
        )
        blocks.append(
            f"- **{spec.name}**: hold `{spec.key_col}` in {spec.held_values}; "
            f"inference: {regime}; registry task: `{spec.registry_task}`; "
            f"{recorded}.\n  {spec.note}"
        )
    source = manifest.get("data_source", "unspecified")
    fixture_notice = (
        "Synthetic fixture; not a biological result."
        if "synthetic" in source.lower()
        else "Run-level provenance; not submission-ready Methods."
    )
    splits = "\n".join(blocks) or "No split specifications supplied."
    return f"""# {cluster} execution notes

> {fixture_notice}

Data source: {source}.

## Recorded splits

{splits}

## Data handling and compute

Consult the selected loader, adapter and manifest for this run's preprocessing,
side inputs, training configuration and recorded timing. No dataset inventory,
GPU-hour estimate or peak-memory guarantee is inferred from the cluster label.
The shared execution notes define the membership audit and metric scope.
"""


def write_methods(
    cluster: str,
    raw_df: pd.DataFrame,
    specs: Sequence[SplitSpec],
    manifest: dict,
    cluster_out: Path,
    shared_out: Path,
) -> tuple[Path, Path]:
    """Write run notes to the existing CLI's compatibility filenames."""
    shared_out, cluster_out = Path(shared_out), Path(cluster_out)
    shared_out.parent.mkdir(parents=True, exist_ok=True)
    cluster_out.parent.mkdir(parents=True, exist_ok=True)
    shared_out.write_text(framework_methods(manifest), encoding="utf-8")
    cluster_out.write_text(
        cluster_methods(cluster, raw_df, specs, manifest), encoding="utf-8"
    )
    return cluster_out, shared_out
