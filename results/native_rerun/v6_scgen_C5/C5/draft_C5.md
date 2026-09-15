# C5 run report

> Run diagnostics only; this report is not the curated submission panel.

Data source: op3_GSE279945.
Completed result rows supplied: 5 of 5.
Rows retain the caller's aggregation level; raw runs and failures are recorded
in `results_raw.csv` and `manifest.json`.

## Recorded scores

| dataset | split | baseline | action | pearson_delta | e_distance | aucell_program_corr |
| --- | --- | --- | --- | --- | --- | --- |
| op3_GSE279945 | C5_global_compound_holdout | scGen | run_floor | 0.079 | 19.996 | 0.115 |
| op3_GSE279945 | C5_loct_B | scGen | run_headline | 0.127 | 9.069 | 0.498 |
| op3_GSE279945 | C5_loct_Mono | scGen | run_headline | 0.159 | 7.889 | 0.752 |
| op3_GSE279945 | C5_loct_NK | scGen | run_headline | 0.241 | 7.315 | 0.307 |
| op3_GSE279945 | C5_loct_T_cells | scGen | run_headline | 0.194 | 7.732 | 0.619 |

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
