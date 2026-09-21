# Reproduction guide

Use the v1.2.5 revision snapshot accompanying BIB-26-1553. It preserves v1.2.2's prediction scores where the execution was unchanged, re-runs the executions the 9 September audit had excluded, admits scGPT and scFoundation on the two compound settings through a task interface written for this study, and corrects reference inventory and display rounding. The package has three distinct reproduction levels.

## 1. Mean-profile evaluation: CPU only

The independently verified environment uses Python 3.10.13. Core dependency versions are fixed in `requirements-core.txt` and recorded in `environment-verified.txt` in the submission archive. Other Python versions must be compatible with those dependencies.

```bash
make setup
make reproduce
make test
```

`make reproduce-eval` uses `results/_paper/census_bundle_manifest.csv`, not an unrestricted glob. It writes 1,401 rows to `reproduced_results.csv`. The manifest includes 1,149 selected model inputs and 252 simple-reference inputs. A bundle can contain multiple held-out strata; bundle count is not sample size.

`make check` independently re-scores eligible saved bundles, reconstructs the selected 58 evaluations and checks their scores, target alignment, source hashes, 1,698 analysis-unit rows and uncertainty summaries. It must print `DEPOSIT CONSISTENCY: PASS`. Historical bundles can remain in the archive without entering the selected panel; their existence does not imply valid native execution.

**The common panel mask on T3 and T4.** A reader re-scoring an unseen-gene or unseen-knockout bundle will find an `exclude_gene_idx` larger than the held targets alone, and should know why. Checkpoint-based models have no input embedding and no output column for genes outside the released `OS_scRNA_gene_index.19264.tsv`, so those genes cannot be predicted by them at all; padding them with the control mean would score a prediction no model made. They are therefore removed from the metric. The removal is a property of the 2,000-gene panel, not of a model: the same genes are excluded for every model AND for both universal-floor members on those cells, so no comparison is made across different gene sets. It affects 26 to 59 genes of 2,000 depending on the dataset arm. `results/_paper/panel_mask_migration.json` records every bundle it was applied to, with the score before and after: 243 bundles, moving Pearson-Δ between −0.009 and +0.046. `scripts/apply_panel_mask.py` is the migration and `src/ivcbench/eval/panel_mask.py` computes the gene set. The Frangieh 20-marker protein readout is not masked, since no checkpoint-based model is scored on it.

The panel has 46 native, eight adapted and four diagnostic entries. Native CPA/scGen only are included; native chemCPA supplies the held-compound CPA/chemCPA entry. STATE's T3/T4/T5 executions were excluded after an output-recovery error and have since been re-run through the corrected runner, so STATE is native on all six settings; see [EXECUTION_AUDIT.md](EXECUTION_AUDIT.md), whose update header records both. No new drug fitting is needed for replay.

Mean profiles reproduce response-direction Pearson-Δ, not per-cell distributions. Energy distance is unavailable from mean-only bundles. A per-cell bundle must also supply a training-only PCA basis; the scorer returns a missing energy-distance value when that information is absent rather than fitting a basis on held-out cells.

## 2. Analytical summaries, figures and documents

```bash
make census
make summaries
make figures
```

Run from the package root.

The document builders are **not in this repository** and the two commands below will not run from
a clone or from the Zenodo archive. They apply only to the submission archive, which carries the
`submission/` source directory:

```
python submission/build_submission.py --out rebuilt_documents
python submission/build_response.py --out rebuilt_documents
```

The common inference family has 27 entries with at least eight analysis units; the remaining 31 receive descriptive observed ranges, not confidence intervals or P values. Bootstrap intervals and Wilcoxon tests condition on the recorded fits, selected floor and fixed masks. They do not quantify optimization-seed or independent-dataset uncertainty. See [ANALYSIS_SCOPE.md](ANALYSIS_SCOPE.md).

Auxiliary reproducibility is explicitly bounded:

- Figure 3 scores immune programs with the equal-weight mean-expression shift, calculated identically from predicted and observed mean profiles, on the same supported conditions and eligible units for every model. `scripts/program_analysis.py` and `scripts/lineage_analysis.py` recompute the Figure 3b-d and Figure S6 analyses and the Supplementary Table S7, S12 and S22 sources on CPU from the deposited mean-profile bundles, with no model fitting and no access to cells; `make check` runs both and each compares its result with `results/_paper/immune_program_revision/`, which holds the full-precision unit values, program membership, support and exclusion reasons. The rank score the shift replaced is retained as a sensitivity analysis and recomputed alongside it, and the donor analysis keeps its own rank-based program-shift error. Identity controls for the rank readout are in `immune_readout_target_validation.csv`; NA-O means a constant observed target and NA-P a constant prediction against a variable target. Two things this does not cover: the surface-marker summaries behind Figure 3a, which come from the source RNA and protein data, and the observed per-cell caches, which serve only an aggregation diagnostic and need source cells to regenerate.
- `assemble_target_diagnostics.py` replays retained disjoint-control and paired shared-control partition correlations for the exact held targets and score masks. It emits six task summaries and precision/power/attenuation statements for all 58 contrasts. Regenerating partitions with `build_target_repeatability.py` requires the source cells and raw-data loader dependencies; no model is fitted. The 100-partition ranges are not biological confidence intervals or prediction ceilings.
- External donor validation retains seed-level scalar scores, not fitted cell predictions. The two training seeds are averaged within donor.
- The matched learning curve uses the same ten evaluation donors and seed-0 training subsets for CellOT/scGPT. scGPT uses a fixed 8,000-stimulated-cell cap. Available training cells are not the number actually consumed.
- Frangieh/Chen checkpoint figures can be regenerated from their supplied marker-level summaries. Recomputing those summaries requires the source RNA/protein objects and their documented normalizations.
- Timing records are partial and include a specific CPU replay measurement. Neither elapsed times nor a 48-GB L40 worker configuration establish full-panel GPU cost or a minimum-memory requirement.

`source_data/` in the submission archive contains one full-precision source per final table panel where numeric analytical sources are available, plus exact interface records. There is no second, rounded duplicate of every source table. The document builder uses the same source for clean and highlighted versions and asserts that their text differs only in editable run colors. Scientific artwork is identical.

## 3. Original-data fitting: separate model environments

Raw cell data and third-party pretrained weights are not redistributed here. Accessions and access limits are in [data/README.md](data/README.md). The executed configurations, trained components and budgets are recorded in Supplementary Table S15c and `results/_paper/supplementary_tables/Supplementary_Table_S15c.csv`.

The thin interfaces under `model_runners/` require their original model-family packages/environments, data preparation and checkpoints. Their native/adapted status is task-specific. The common CPU environment is not an environment for simultaneously fitting all neural methods. Historical fitting scripts are provenance, not a guarantee that a single command recreates every run on arbitrary hardware.

Shared upstream feature selection, published-object preprocessing and transductive diagnostic operations are disclosed in Supplementary Note S2; do not reinterpret the complete benchmark as a fully inductive end-to-end pipeline. No new raw-data fit is required to verify the submitted Pearson-Δ panel.

In particular, Soskic's supplied resting/activated matrices were separately covariate-regressed, scaled and capped at 10. They contain no raw-count layer. `audit_soskic_input_space.py` checks the original files and records hashes, feature statistics and the source-method reference; its outputs are retained for CPU inspection. The fold-local sensitivity changes additional scaling only, not source regression, clipping or feature selection. Raw RNA reads are controlled-access EGA EGAD00001008197; public processed inputs do not establish raw-cell inductive validation.

## Optional container

```bash
docker build -f Containerfile -t ivcbench .
docker run --rm ivcbench
```

The container performs the same CPU checks and does not download raw cells or fit neural models. A build requires network access to Python dependencies. The primary verified path is the local CPU environment described above.
