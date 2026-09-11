"""Run-level diagnostic reports, not publication-ready scientific conclusions.

The raw runner can execute configurations outside the submitted panel. Its
``action`` and ``headline_eligible`` fields describe runtime gating, not the
native/adapted/diagnostic census. Submission prose is maintained separately in
``submission/`` and is generated only from the curated analytical sources.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

METRICS = ("pearson_delta", "e_distance", "aucell_program_corr")
IDENTIFIERS = ("dataset", "split", "baseline", "action")


def _format_value(value: object) -> str:
    """Keep missing scores visible and escape Markdown table delimiters."""
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, int)):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _results_table(frame: pd.DataFrame) -> str:
    columns = [column for column in (*IDENTIFIERS, *METRICS) if column in frame]
    if frame.empty or not columns:
        return "No completed result rows were supplied."
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(_format_value, row)) + " |")
    return "\n".join(lines)


def run_report(cluster: str, frame: pd.DataFrame, data_source: str) -> str:
    """Describe only the rows supplied; do not infer model validity or support."""
    completed = frame
    if "ran" in frame:
        completed = frame.loc[frame["ran"].eq(True)]
    synthetic = "synthetic" in (data_source or "").lower()
    notice = (
        "Synthetic fixture: these scores test the pipeline, not biological prediction."
        if synthetic
        else "Run diagnostics only; this report is not the curated submission panel."
    )
    return f"""# {cluster} run report

> {notice}

Data source: {data_source or 'unspecified'}.
Completed result rows supplied: {len(completed)} of {len(frame)}.
Rows retain the caller's aggregation level; raw runs and failures are recorded
in `results_raw.csv` and `manifest.json`.

## Recorded scores

{_results_table(completed)}

## Interpretation and scope

Pearson-Δ measures response-pattern agreement, not amplitude calibration.
Energy distance requires predicted and observed cell clouds and a training-fitted
PCA basis. The runner's program metric averages per-cell rank scores within each
stratum; it is distinct from the submission's matched mean-profile readout.
Missing program correlations remain NA.

Runtime applicability flags do not certify a native prediction operation or
admission to the final panel. The split auditor checks cell membership, not
upstream normalization or feature selection. In particular, supplied Soskic
inputs use condition-specific residualized coordinates.

For the submitted comparisons, use `make census`, `make summaries` and the
document builders in `submission/`. Those sources specify the final roster,
paired reference, analysis units and conditional uncertainty. This run report
does not assign manuscript figure numbers or generate claims of model success.
"""


def c1_draft(df: pd.DataFrame, data_source: str) -> str:
    """Compatibility entry point for a C1 diagnostic report."""
    return run_report("C1", df, data_source)


def c3_draft(df: pd.DataFrame, data_source: str) -> str:
    """Compatibility entry point for a C3 diagnostic report."""
    return run_report("C3", df, data_source)


def c5_draft(df: pd.DataFrame, data_source: str) -> str:
    """Compatibility entry point for a C5 diagnostic report."""
    return run_report("C5", df, data_source)


def write_draft(
    cluster: str, df: pd.DataFrame, data_source: str, path: str | Path
) -> Path:
    """Retain historical output filenames while clearly labelling their scope."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run_report(cluster, df, data_source), encoding="utf-8")
    return path
