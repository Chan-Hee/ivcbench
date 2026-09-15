# Reproduction guide

Use the v1.2.3 revision snapshot accompanying BIB-26-1553. This interface/provenance correction preserves v1.2.2's prediction scores, reclassifies the PertAdapt-inspired T2 head as adapted, and corrects reference inventory and display rounding. The package has three distinct reproduction levels.

## 1. Mean-profile evaluation: CPU only

The independently verified environment uses Python 3.10.13. Core dependency versions are fixed in `requirements-core.txt` and recorded in `environment-verified.txt` in the submission archive. Other Python versions must be compatible with those dependencies.

```bash
make setup
make reproduce
make test
```

`make reproduce-eval` uses `results/_paper/census_bundle_manifest.csv`, not an unrestricted glob. It writes 1,362 rows to `reproduced_results.csv`. The manifest includes 1,110 selected model inputs and 252 simple-reference inputs. A bundle can contain multiple held-out strata; bundle count is not sample size.

`make check` independently re-scores eligible saved bundles, reconstructs the selected 47 evaluations and checks their scores, target alignment, source hashes, 1,605 analysis-unit rows and uncertainty summaries. It must print `DEPOSIT CONSISTENCY: PASS`. Historical bundles can remain in the archive without entering the selected panel; their existence does not imply valid native execution.

The panel has 34 native, seven adapted and six diagnostic entries. Native CPA/scGen only are included; native chemCPA supplies the held-compound CPA/chemCPA entry. STATE T1/T2 select genuine prediction files. STATE T3/T4/T5 results were excluded after an output-recovery error; see [EXECUTION_AUDIT.md](EXECUTION_AUDIT.md). No new drug fitting is needed for replay.

Mean profiles reproduce response-direction Pearson-Δ, not per-cell distributions. Energy distance is unavailable from mean-only bundles. A per-cell bundle must also supply a training-only PCA basis; the scorer returns a missing energy-distance value when that information is absent rather than fitting a basis on held-out cells.

## 2. Analytical summaries, figures and documents

```bash
make census
make summaries
make figures
.venv/bin/python submission/build_submission.py --out rebuilt_documents
.venv/bin/python submission/build_response.py --out rebuilt_documents
```

Run from the package root. The document commands apply to the submission archive,
which includes the `submission/` source directory.

The common inference family has 24 entries with at least eight analysis units; the remaining 23 receive descriptive observed ranges, not confidence intervals or P values. Bootstrap intervals and Wilcoxon tests condition on the recorded fits, selected floor and fixed masks. They do not quantify optimization-seed or independent-dataset uncertainty. See [ANALYSIS_SCOPE.md](ANALYSIS_SCOPE.md).

Auxiliary reproducibility is explicitly bounded:

- Immune-program comparisons score predicted and observed mean profiles with the same rank function. Identity controls are in `immune_readout_target_validation.csv`; NA-O means a constant observed target and NA-P a constant prediction against a variable target. Observed per-cell caches serve only an aggregation diagnostic. Regenerating those caches requires source cells.
- `assemble_target_diagnostics.py` replays retained disjoint-control and paired shared-control partition correlations for the exact held targets and score masks. It emits six task summaries and precision/power/attenuation statements for all 47 contrasts. Regenerating partitions with `build_target_repeatability.py` requires the source cells and raw-data loader dependencies; no model is fitted. The 100-partition ranges are not biological confidence intervals or prediction ceilings.
- External donor validation retains seed-level scalar scores, not fitted cell predictions. The two training seeds are averaged within donor.
- The matched learning curve uses the same ten evaluation donors and seed-0 training subsets for CellOT/scGPT. scGPT uses a fixed 8,000-stimulated-cell cap. Available training cells are not the number actually consumed.
- Frangieh/Chen checkpoint figures can be regenerated from their supplied marker-level summaries. Recomputing those summaries requires the source RNA/protein objects and their documented normalizations.
- Timing records are partial and include a specific CPU replay measurement. Neither elapsed times nor a 48-GB L40 worker configuration establish full-panel GPU cost or a minimum-memory requirement.

`source_data/` in the submission archive contains one full-precision source per final table panel where numeric analytical sources are available, plus exact interface records. There is no second, rounded duplicate of every source table. The document builder uses the same source for clean and highlighted versions and asserts that their text differs only in editable run colors. Scientific artwork is identical.

## 3. Original-data fitting: separate model environments

Raw cell data and third-party pretrained weights are not redistributed here. Accessions and access limits are in [data/README.md](data/README.md). The executed configurations, trained components and budgets are recorded in Supplementary Table S3 and `source_data/Table_S3.csv`.

The thin interfaces under `model_runners/` require their original model-family packages/environments, data preparation and checkpoints. Their native/adapted status is task-specific. The common CPU environment is not an environment for simultaneously fitting all neural methods. Historical fitting scripts are provenance, not a guarantee that a single command recreates every run on arbitrary hardware.

Shared upstream feature selection, published-object preprocessing and transductive diagnostic operations are disclosed in Supplementary Note S2; do not reinterpret the complete benchmark as a fully inductive end-to-end pipeline. No new raw-data fit is required to verify the submitted Pearson-Δ panel.

In particular, Soskic's supplied resting/activated matrices were separately covariate-regressed, scaled and capped at 10. They contain no raw-count layer. `audit_soskic_input_space.py` checks the original files and records hashes, feature statistics and the source-method reference; its outputs are retained for CPU inspection. The fold-local sensitivity changes additional scaling only, not source regression, clipping or feature selection. Raw RNA reads are controlled-access EGA EGAD00001008197; public processed inputs do not establish raw-cell inductive validation.

## Optional container

```bash
docker build -f Containerfile -t ivcbench .
docker run --rm ivcbench
```

The container performs the same CPU checks and does not download raw cells or fit neural models. A build requires network access to Python dependencies. The primary verified path is the local CPU environment described above.
