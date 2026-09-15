# Shared execution notes

These notes describe the generic runner, not the final manuscript Methods.
Seeds requested: [0].
Software: numpy 2.2.6, scipy 1.15.3, sklearn 1.7.2, pandas 2.3.3, matplotlib 3.10.9.

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
