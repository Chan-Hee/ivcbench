# Frozen-bundle program and response reanalysis

Run from the project root:

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ivcbench/.venv/bin/python -B revision_claude/figure3_complete_20260918_1556/scripts/program_analysis.py
```

The script reads only the isolated input snapshot and writes only this directory. It does not import models, load raw cells, train, or use a GPU. `protocol.json` was frozen before this final computation. The metric change was chosen during manuscript revision; this is not a claim of study preregistration. An attempted rerun with a different input/protocol raises an error.

## Estimands and support

The primary program score is the equal-weight mean of expression changes from control over a fixed member set: the original published program list intersected with measured features and the existing benchmark feature mask. The same fixed member set, control and observations are used for every model and stratum of a dataset/program. It is called **benchmark-mask mean-expression delta**, not direct-target-only exclusion. Per-stratum target removal is deliberately absent because it would change program composition across perturbations.

Per-unit concordance is Pearson correlation between predicted and observed program scores across shared supported perturbations. Eligibility depends only on observations: at least three strata, at least one measured member, scalar score SD greater than 1e-12, and retained observed-profile maximum range at least 1e-4. The symmetric predicted-profile numerical-constant guard also uses 1e-4. These numerical flags do not establish biological absence. Undefined correlations remain missing, with reasons. All raw deltas and errors remain available.

The shared support excludes the union of explicit declines from the complete final model roster, identically for every model. T3 Chen retains 27/30 perturbations (CYLD, FHIP1A and PTEN excluded); Schmidt retains 7/7. McCutcheon CRISPRa, McCutcheon CRISPRi and Shifrut each retain two strata and are not comparative correlation units. All five program rows therefore use the same Chen/Schmidt pair. T5c retains B 140/141, Mono 139/140, NK 140/141 and T cells 139/141, totaling 558/563 compound-lineage strata. Navitoclax is removed in every lineage and Alvocidib additionally in T cells. `support_by_unit.csv` identifies each declining model.

Program task summaries are unweighted means over common observed-defined units, requiring every selected unit to have an estimable correlation for a model. There are no per-model available-case averages. Descriptive maxima consider only native/adapted final-census candidates; diagnostic and floor rows remain separate. There are no new confidence intervals, selection-adjusted significance claims, matched-gene null bands, or permutation bands.

The mandatory all-measured-original-members mean sensitivity produces identical primary comparative values because no scored member is removed from eligible Chen/Schmidt program sets, and T5c has no benchmark feature exclusion. The only program-member mask difference in any T3 dataset is NR4A1 in the two-stratum McCutcheon arms. Rank sensitivities separately retain the benchmark-mask ranking universe and the original full measured ranking universe. Both rank mean profiles symmetrically using the original top-5% recovery operator and NumPy tie convention.

## Main plotting tables

Filter `figure3b_best.csv` and `model_program_summary.csv` to `scope=shared_supported_strata`, `gene_mask=benchmark_mask`, `method=mean_expression_delta`. The former contains each descriptive maximum and unit range; the latter contains every candidate, diagnostic and floor. Unit ranges are descriptive ranges, not confidence intervals.

| Task | Program | Descriptive maximum | Mean correlation | Unit range |
|---|---|---|---:|---:|
| T3 | IL2/STAT5 | PerturbNet | 0.449700 | 0.430488–0.468912 |
| T3 | TCR activation | PerturbNet | 0.640634 | 0.522512–0.758755 |
| T3 | Treg/exhaustion | PerturbNet | 0.388691 | 0.256378–0.521003 |
| T3 | Effector cytokine | PertAdapt | 0.477166 | 0.096877–0.857455 |
| T3 | Proliferation | AttentionPert | 0.454629 | 0.305557–0.603702 |
| T5c | Effector lymphocyte | PRnet | 0.408298 | 0.233883–0.708588 |
| T5c | Inflammatory/NF-kB | scGen | 0.615641 | 0.387046–0.773675 |
| T5c | Type-I IFN | scGPT | 0.906068 | 0.810884–0.941468 |

Primary rows are physical CSV rows 32–36 and 60–62, counting the header as row 1. All nine conditioned T3 candidates are comparable on both selected datasets; all ten conditioned T5c candidates are comparable across all four lineages for mean-expression delta. This supports graded program-dependent association; it does not establish that each positive maximum beats a predictive baseline or passes a null test. Every T3 program includes at least one candidate with a negative mean correlation.

`panel_c.csv` primary rows are 15–27 (`scope=shared_supported_strata`). The x coordinate (`correlation`) is the mean of four per-lineage IFN correlations over 20 measured members of the original 25-gene list. The y coordinate (`response_norm_ratio`) uses all 2,000 benchmark expression features and the same shared perturbation strata. Its estimator is the median per-stratum ratio `norm(pred-control)/norm(obs-control)` within each lineage, followed by the median over four lineages. It is not an IFN-only norm, a ratio of pooled norms, or a macro mean. `calibration_slope` is the analogous nested median of the through-origin slope mapping predicted to observed changes, `dot(pred-control,obs-control)/dot(pred-control,pred-control)`. `macro_mean_*` columns are separate mean sensitivities.

All eleven nonfloor models have an estimable primary IFN x coordinate in all four lineages. The two adapted models and fingerprint diagnostic have response norm ratios below one (scFoundation 0.459569, FP-ridge 0.595893, scGPT 0.762347); the eight native candidates range from 1.756706 to 3.015341. Constant-output floors have no estimable x coordinate and must not be plotted at zero. Other measured program coverage is 7/22 inflammatory members and 11/17 effector members in each lineage.

`legacy_full_support_magnitude.csv` reconstructs the existing S22 estimator exactly on original per-bundle target/control arrays and full support: nanmedian within lineage, then median across lineages. It retains explicit control placeholders for declined strata and reports counts of defined slopes. The legacy ratios for scFoundation/FP-ridge/scGPT are 0.458579/0.595859/0.761295. The small differences in primary Figure 3c come from shared-support restriction and common-target alignment, not an undisclosed change from median to mean. `response_model_summary.csv` also gives common-target full-support results; strict slope aggregates stay missing if any required stratum is undefined.

## Sensitivity and error interpretation

The original full-universe rank score has substantial constant-score degeneracy. `legacy_rank_degeneracy_counts.csv` checks constant observed, then constant predicted, before applying the two-stratum comparison policy: T3 has 264 observed-constant, 8 predicted-constant and 28 estimable model/unit/program combinations; T5c has 39, 53 and 64, respectively. `rank_degeneracy_counts.csv` instead retains the new policy-first status used for comparative selection. Independent scalar/profile and fewer-than-three-strata flags in `program_unit_metrics.csv` make both classifications auditable. Numerical constancy is not evidence that the underlying biology is absent.

`program_unit_metrics.csv` includes raw RMSE/MAE/sign agreement and an explicitly auxiliary error comparison. Its reference is the smaller RMSE of cell-mean and linear-PCA separately in each unit/program; this is different from the task-level census floor. High correlation is insufficient for accurate effect size: the selected T5c effector candidate has worse program RMSE than this reference in all four lineages, whereas the selected IFN candidate improves it in all four. Do not use these descriptive, selected comparisons as a new inferential claim.

## Provenance and checks

All 200 copied bundle hashes, roster completeness, gene and stratum identities, fixed masks, controls and observations are checked before scoring. Every candidate is then evaluated against canonical cell-mean target/control arrays. Maximum original observed/control differences are 6.198883e-6/4.768372e-7, consistent with float32 accumulation differences; they are retained in `source_alignment.csv`. Fresh full-support response scores agree with the frozen census unit scores to a maximum absolute difference of 1.447524e-6 after target unification. Identity scoring is exactly zero-error. All 3,648 program-unit rows are checked for fixed-unit aggregation and missing-value preservation. `validation.json` records these checks and hashes the script, frozen protocol and CSV outputs. The independent reviewer separately checks the panel coordinates directly from the frozen NPZ files.
