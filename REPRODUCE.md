# Reproducing the figures and tables

Every figure and Supplementary Table is rebuilt **from the deposited result files in `results/`**; no
raw data and no model re-runs are required. Run each script from the repository root with the `ivc`
environment active (`conda activate ivc` or the `.venv`), e.g.:

```bash
python scripts/make_figure2_landscape_verdict.py
```

Figure scripts read deposited CSV/JSON and write PNG+PDF into `results/_paper/`. Analysis scripts
(listed in the "computed by" column) recompute a derived table from the per-cluster `results_raw.csv`
and the real loaders; they are included for provenance but their outputs are already deposited.

**The headline census is bundle-sourced.** The 35-cell table `results/_paper/cross_cluster_headline.csv`
(Supplementary Table S2b), `within_family_consistency.csv` (S3) and `descriptive_fit_matrix.csv` (S4) are
re-scored directly from the deposited prediction bundles under `predictions/`, the same GPU-free path a
reviewer runs. `make census` rebuilds them (`scripts/assemble_cross_cluster.py`, `assemble_fit_matrix.py`),
and `make check` (also run by `make test`) re-derives the census from the bundles and fails the build if
the committed numbers have drifted from it, so the table a reader reproduces and the table the paper
reports cannot diverge. The C2 donor paired statistics (S5, S7) and their multiplicity adjustment (S10)
are likewise computed from the per-donor bundle scores. The per-cluster `results_raw.csv` tables remain the
provenance for the other derived tables and for the figure scripts; `scripts/sync_results_raw.py`
re-derives their response-direction `pearson_delta` from the bundles (to the printed precision) so the
figures read the same numbers as the census, and `make check` fails if any cell has drifted. In short, one
source (the bundles) feeds the census, the paired statistics, and the figures, and the gate keeps them
locked together.

**Final manuscript layout (binding).** The submission is **3 main figures + 8 supplementary figures
(S1–S8) + 12 Supplementary Tables (S1, S2a, S2b, S3–S12)**:

- **Fig 1**: benchmark framework (`figure_framework.py`)
- **Fig 2**: generalization map + CellOT donor CDF (`make_figure2_landscape_verdict.py`)
- **Fig 3**: immune blind-spot map (`figure_immune_blindspot.py`)

The full per-cluster figure plate (`figure_landscape / ranking / cellcontext / perturbation /
donor_decision / within_family_fit`) is retained for provenance and feeds several of the S-figures
below; the 8 submitted supplementary-figure PNGs (S1–S8) are the renders deposited in `results/_paper/`
(md5-identical to `BiB_submission/figures/SupplementaryFigure_S1..S8.png`). The additional analyses
listed further below (chemistry-channel collapse, program-dimensionality, and the three new-data
analyses) are deposited as release provenance; in the submitted supplement they appear as
Supplementary Notes/Tables, not as numbered supplementary figures.

---

## Main figures

| Main figure | Script (`scripts/`) | Output (`results/_paper/`) | Reads (deposited) |
|---|---|---|---|
| **Fig 1: Benchmark framework** | `figure_framework.py` | `figure_framework.{png,pdf}` | `results/{C1,C3,C4,C5}/results_raw.csv` (method / lineage / donor / split counts) |
| **Fig 2: Generalization map + donor CDF** | `make_figure2_landscape_verdict.py` | `figure2_landscape_verdict.{png,pdf,tiff}` | (a) 35-cell map (32 conditioned + 3 CINEMA-OT comparators) from `cross_cluster_headline.csv`; (b) CellOT donor CDF from `cellot_soskic_raw.csv` + `cellot_summary.csv` + `cellot_vs_floor_donor_paired.csv`, paired-Wilcoxon p in `headline_multiplicity_adjusted.csv` |
| **Fig 3: Immune blind-spot map** | `figure_immune_blindspot.py` | `figure_immune_blindspot.{png,pdf}` | `immune_novelty/T1_C4_per_marker_protein_recovery.csv`, `immune_novelty/T2_per_program_AUCell_map.csv`, `immune_novelty/T3_per_lineage_predictability.csv`, `results/C3/program_null.csv`, `results/C5/results_raw.csv`, `multiseed_scgen_summary.csv` |

---

## Supplementary figures (S1–S8, submitted) + additional release analyses (S9–S13)

**S1–S8 are the submitted supplementary figures** (md5-identical to `BiB_submission/figures/`).
**S9–S13 below are additional analyses deposited as release provenance; they are NOT part of the
submitted 8-figure supplement**; in the manuscript these analyses appear as Supplementary
Notes/Tables, not as numbered supplementary figures. Each row: **S-number → generating script →
backing data file**; every script and file named below is present in this release (verified).

| S# | Content | Figure script (`scripts/`) | Output (`results/_paper/`) | Computed by → backing data (deposited) |
|---|---|---|---|---|
| **S1** | Method × split ranking heatmap (raw Pearson-Δ) | `figure_ranking.py` | `figure_ranking.{png,pdf}` | `results/{C1,C3,C4,C5}/results_raw.csv` |
| **S2** | Within-family consistency + descriptive fit-matrix | `figure_within_family_fit.py` | `figure_within_family_fit.{png,pdf}` | `within_family_consistency.csv`, `descriptive_fit_matrix.csv`, `scpram_vs_cellot_donor_paired.csv`, `defensive_stats.json` |
| **S3** | Perturbation + modality axes (degenerate / structural zeros) | `figure_perturbation.py` | `figure_perturbation.{png,pdf}` | `results/C3/results_raw.csv`, `results/C5/results_raw.csv`, `results/C4/cite_marker_recovery.csv` |
| **S4** | PD-L1 assay-power + RNA-vs-surface decoupling | `figure_c4_pdl1_assay_power.py` | `figS_c4_pdl1_assay_power.{png,pdf}` | `c4_pdl1_assay_power.py` → `c4_pdl1_assay_power_summary.json`, `c4_surface_marker_CIs.csv`, `c4_rna_vs_surface_decoupling.csv` (from `results/C4/cite_marker_recovery.csv`) |
| **S5** | Per-lineage / per-compound cell-context recovery | `figure_cellcontext.py` | `figure_cellcontext.{png,pdf}` | `results/C5/results_raw.csv`, `immune_novelty/T3_per_lineage_predictability.csv`, `results/C5/ifn_shuffle_null.csv` |
| **S6** | Reliability / noise ceiling | `figure_reliability_ceiling.py` | `figure_reliability_ceiling.{png,pdf}` | `reliability_ceiling.py` → `immune_novelty/reliability_ceiling.csv` (+ `reliability_ceiling_perunit.json`); also reads `cross_cluster_headline.csv` |
| **S7** | C3 unseen-gene predictability probe | `c3_predictability_analysis.py` (computes ranking **and** renders) | `figS_c3_predictability_probe.{png,pdf}` | `c3_predictability_probe.py` → `results/C3/predictability_probe_pergene.csv` (+ `predictability_probe_stats.json`); ranking written to `predictability_factor_ranking.csv` |
| **S8** | C3 nearest-gene baseline | `figure_c3_nearest_gene.py` | `figS_c3_nearest_gene.{png,pdf}` | `c3_nearest_gene_baseline.py` → `results/C3/nearest_gene_baseline.csv`; summarized by `c3_nearest_gene_summary.py` → `c3_nearest_gene_summary.csv`; also reads `results/C3/results_raw.csv` |
| **S9** | chemCPA chemistry-channel collapse | `figure_chemcpa_collapse.py` | `figure_chemcpa_collapse.{png,pdf}` | `chemcpa_collapse_analysis.py` → `chemcpa_collapse_arrays.npz` + `chemcpa_collapse_stats.json` |
| **S10** | Program-recovery vs program-dimensionality | `figure_program_dimensionality.py` | `program_recovery_vs_dimensionality.{png,pdf}` | `program_recovery_vs_dimensionality.py` → `program_recovery_vs_dimensionality.csv` (+ `_corr.json`, `program_shift_signal.csv`); recovery from `immune_novelty/T2_per_program_AUCell_map.csv` |
| **S11** | Unseen-cytokine LOCO (Human Cytokine Dictionary; the two conditioning regimes) | `figure_newdata_cytokine_loco.py` | `figS_newdata_cytokine_loco.{png,pdf}` | `newdata_cytokine_loco.py` → `results/newdata/cytokine_loco_per_held.csv`, `cytokine_loco_per_celltype.csv`, `cytokine_loco_summary.json` (+ `cytokine_loco_unscoreable.csv`, `cytokine_loco_README.md`) |
| **S12** | Chen FOXP3 Perturb-icCITE-seq checkpoint replication (n = 2 surface readout) | `figure_chen_checkpoint_replication.py` | `figS_chen_checkpoint_replication.{png,pdf}` | `chen_checkpoint_replication.py` → `results/newdata/chen_cite_marker_recovery.csv`, `chen_checkpoint_replication_summary.json`; also reads `results/_paper/c4_surface_marker_CIs.csv` (Frangieh comparator) |
| **S13** | CellOT donor learning curve (Soskic, donor-count grid) | `figure_cellot_donor_learning_curve.py` | `figS_cellot_donor_learning_curve.{png,pdf}` | `cellot_donor_learning_curve.py` (GPU runner; provenance) → `results/newdata/cellot_donor_learning_curve_seed{0,1}.csv` (+ merged `cellot_donor_learning_curve.csv`, `cellot_donor_learning_curve_summary.csv`) |

**S13 GPU runner.** The per-seed learning-curve CSVs were produced by `scripts/cellot_donor_learning_curve.py`,
launched on two GPUs by `scripts/launch_cellot_donor_learning_curve.sh` (override the CellOT-env python with
`IVCBENCH_CELLOT_PY`). That runner needs a GPU, the CellOT conda env, and the raw Soskic data; it is included
for provenance only. The **figure** (`figure_cellot_donor_learning_curve.py`) rebuilds from the deposited
per-seed CSVs with no GPU.

---

## Supplementary Tables (S1, S2a, S2b, S3–S12)

Each Supplementary Table is published with the submission under the **deposited file name** shown
(in `BiB_submission/supplementary/`) and is backed by the listed result file in this release. S1 and
S2a are curated inventory tables (maintained in `scripts/datasets.csv` and `supp/results_supplementary.md`);
the rest are mechanically assembled from the deposited per-cluster leaderboards and immune-novelty tables.
*(The supplementary **Figure** series in the sections above is independent of the table numbering: the
submitted figures are S1–S8 (S9–S13 are additional release analyses, not submitted figures), while
Supplementary **Tables** are S1, S2a, S2b, S3–S12.)*

| Table | Content | Deposited file (`BiB_submission/supplementary/`) | Backing data / script (`scripts/`) |
|---|---|---|---|
| **S1** | 21-dataset immune inventory (12 anchors + curation criteria) | `Supplementary_TableS1_extended_dataset_inventory.docx` | `scripts/datasets.csv` (curated manifest) |
| **S2a** | Surveyed method + comparator inventory with benchmark applicability | `Supplementary_Table_S2a_method_survey.csv` | curated inventory (`datasets.csv` + `supp/results_supplementary.md`) |
| **S2b** | Per-(model, task) headline census (35-cell Pearson-Δ vs floor: 32 conditioned + 3 CINEMA-OT comparators) | `Supplementary_Table_S2b_cross_cluster_headline.csv` | `cross_cluster_headline.csv` via `assemble_cross_cluster.py` |
| **S3** | Within-family verdict agreement + per-unit Spearman ρ | `Supplementary_Table_S3_within_family_consistency.csv` | `within_family_consistency.csv` |
| **S4** | Descriptive fit-matrix (a-priori expectation vs observed beats-floor per family×task) | `Supplementary_Table_S4_descriptive_fit_matrix.csv` | `descriptive_fit_matrix.csv` via `assemble_fit_matrix.py` |
| **S5** | Donor axis: CellOT vs floor, paired per-donor (n = 106; cell-mean gap +0.109 **and** matched-baseline gap +0.102 [CI]) | `Supplementary_Table_S5_cellot_vs_floor_donor_paired.csv` | `cellot_vs_floor_donor_paired.csv` + `cellot_summary.csv` via `cellot_assemble.py` |
| **S6** | Per-immune-program AUCell-Δ recovery map | `Supplementary_Table_S6_per_program_AUCell_map.csv` | `immune_novelty/T2_per_program_AUCell_map.csv` |
| **S7** | Donor axis: scPRAM vs CellOT, paired per-donor | `Supplementary_Table_S7_scpram_vs_cellot_donor_paired.csv` | `scpram_vs_cellot_donor_paired.csv` via `cellot_assemble.py` / `soskic_donor_postprocess.py` |
| **S8** | Per-surface-marker protein recovery (PD-1/PD-L1; effect-size vs sign-match) | `Supplementary_Table_S8_per_marker_protein_recovery.csv` | `immune_novelty/T1_C4_per_marker_protein_recovery.csv` + `c4_surface_marker_CIs.csv` via `c4_per_marker.py` |
| **S9** | Per-lineage predictability (cell-context advantage localization) | `Supplementary_Table_S9_per_lineage_predictability.csv` | `immune_novelty/T3_per_lineage_predictability.csv` via `c5_loct_expand.py` |
| **S10** | Headline-survivor table after BH/Holm multiplicity correction (two pre-specified families) | `Supplementary_Table_S10_headline_multiplicity_adjusted.csv` | `headline_multiplicity.py` byte-reproduces `results/_paper/headline_multiplicity_adjusted.csv` (two-family BH/Holm; H4 FP-ridge BH=0.0625 does not survive). The deposited `BiB_submission/` S10 is the journal-formatted view of that table (reordered columns, rounded display, T-task labels). |
| **S11** | Per-program AUCell recovery vs observed-shift magnitude (the r = +0.87 law) + dimensionality proxies | `Supplementary_Table_S11_program_recovery_vs_magnitude.csv` | `program_recovery_vs_dimensionality.csv` via `program_recovery_vs_dimensionality.py` |
| **S12** | Distributional-fidelity axis: per-(task, split, model) energy distance + Pearson-Δ | `Supplementary_Table_S12_energy_distance.csv` | assembled from per-cluster `results/{C1,C3,C4,C5}/results_raw.csv` distributional columns |

---

## New-data corroboration analyses (S11–S13)

The three new-data supplementary figures extend the benchmark's central law
(*conditioning helps cell/donor context, not unseen biology*) onto three independent datasets:

- **S11: unseen-cytokine LOCO** (`newdata_cytokine_loco.py`): leave-one-cytokine-out over the Human
  Cytokine Dictionary summary table (`data/human_cytokine_dict/hcd_mini.csv`, raw). It separates the
  **two conditioning regimes**: annotation-only "truly novel" cytokines sit at/below the floor (the law
  holds), while a cytokine observed in *other* celltypes transfers above it. Outputs in `results/newdata/`.
- **S12: Chen checkpoint replication** (`chen_checkpoint_replication.py`): an n = 2 surface-protein
  readout replication on the Chen FOXP3 Perturb-icCITE-seq data (`data/C3/chen`, access-controlled),
  scored on the same surface-marker recovery metric as C4/Frangieh.
- **S13: CellOT donor learning curve** (`cellot_donor_learning_curve.py`): the donor-count sweep
  {8,16,32,64,96} on Soskic, two seeds on two GPUs, showing how the CellOT donor-axis win scales with the
  number of training donors. The merged/summary CSVs are deposited; the figure rebuilds with no GPU.

The analysis scripts (`newdata_cytokine_loco.py`, `chen_checkpoint_replication.py`,
`cellot_donor_learning_curve.py`) read **raw** data and are included for provenance; the three figure
scripts read only the deposited tables in `results/newdata/` (+ one Frangieh comparator in
`results/_paper/`) and rebuild without raw data or a GPU.

---

## Retraining from scratch

Everything above rebuilds from the deposited predictions with no GPU. To regenerate the predictions themselves
by retraining every model, you need the raw data and the per-family environments.

**Raw data in one command.** `make data` (which runs `scripts/download_all.sh`) fetches every public census
dataset into `data/<cluster>/<dataset>/` and prints a summary of what is present and what is still missing;
`bash scripts/download_all.sh --list` previews the plan without downloading. The two access-controlled deposits
(Chen 2025 via DDBJ/GEA login, Cano-Gamez via the EGA data-access committee) are named in that summary but must
be obtained manually; see `data/README.md` and `scripts/apply_ega_dac.md`.

**The bundled-environment training image.** As an alternative to building each environment by hand, the
per-family conda environments can be bundled into one large training image. `scripts/build_train_image.sh`
conda-packs each existing heavy env into `build/train_envs/<env>.tar.gz` and then builds `Containerfile.train`,
which unpacks them into `/opt/conda/envs/<env>` so the family interpreters are present at
`/opt/conda/envs/<env>/bin/python` (the build script also writes `build/train_envs/train_image.env` with the
matching `IVCBENCH_*_PY` exports). This is the scPerturBench-style artifact: it is large (the prior full build
was about 70 GB), so it is built once and hosted on Zenodo rather than in this repository, and it is separate
from the small, verified, GPU-free eval image (`Containerfile`). `build/` is git-ignored.

**Per-family environments (built by hand).** Each model family has its own environment because the upstream
implementations carry conflicting CUDA and PyTorch versions (they span CUDA 11.3 to 12.8); each environment
carries its own CUDA runtime, so only a recent NVIDIA driver is needed.

Build each heavy environment from its model's **upstream repository** (each upstream pins its own
CUDA/PyTorch); only the core `ivc` env ships here (`environment.yml`). The authoritative provenance for every
model is in the header of its runner script under `scripts/`.

| Environment | Models | Upstream implementation (build the env from this) |
|---|---|---|
| `ivc` (core, CPU) | run_cluster orchestration, floor baselines, FP-ridge, linear-shift-KOemb, metrics, assembly, figures | this repo (`environment.yml`) |
| `cellot` | CellOT | `bunnech/cellot` @ `ff28778` (Bunne et al. 2023); scored on the official `ae`/`data_space` path (`scripts/cellot_runner.py`) |
| `ivc-scpram` | scPRAM | `github.com/jiang-q19/scPRAM`, PyPI `scpram==0.0.3` (Jiang et al. 2024, Bioinformatics btae265) |
| `ivc-cpa` | CPA, chemCPA | `theislab/cpa` (+ chemCPA) |
| `scperturbench_eval` (+ `_jaxgpu` for the GPU path) | scGen, CINEMA-OT | `theislab/scgen`; CINEMA-OT via the scPerturBench eval harness |
| `ivc-state` | STATE | Arc Institute **State** state-transition model (`scripts/state_*.py`) |
| `scgpt` | scGPT, GEARS, AttentionPert | `bowang-lab/scGPT`, `snap-stanford/GEARS`, AttentionPert (`scripts/graph_frangieh.py`) |
| `scfoundation` | scFoundation, PertAdapt | `biomap-research/scFoundation`; PertAdapt (Bai et al. 2025); official artifacts per `data/pertadapt/official/PROVENANCE.md` (`scripts/pertadapt_validate.py`) |

Run each model family's runner script (`scripts/{cellot,scpram,state,cpa,chemcpa,graph,cinemaot,pertadapt}_*.py`,
plus `scripts/run_cluster.py` for the core-runner models) in its own environment from the table above; each
runner inlines its own per-model commands and hyperparameters (caps, epochs, steps, seeds). The donor-axis
models (STATE, scPRAM on Soskic) run one donor per GPU: each donor launches roughly two dozen internal workers,
so more than two in parallel oversubscribes the host. CPA uses three seeds, the others one.

**Runner environment variables (`$IVCBENCH_*`).** The per-family runners take their inputs and per-model
settings from environment variables, so the same script runs on any host with no path edits. The ones you set
when retraining:

| Variable(s) | Purpose |
|---|---|
| `IVCBENCH_KANG_PATH`, `IVCBENCH_SOSKIC_PATH`, `IVCBENCH_OP3_PATH`, `IVCBENCH_FRANGIEH_DIR` | raw dataset locations (download per [`data/README.md`](data/README.md)) |
| `IVCBENCH_SCPERTURBENCH_DATASET_DIR`, `IVCBENCH_GENE2GO`, `IVCBENCH_SCFOUNDATION_CKPT` | scPerturBench data dir, GO mapping (GEARS/PertAdapt), scFoundation checkpoint |
| `IVCBENCH_CELLOT_PY`, `IVCBENCH_IVC_SCPRAM_PYTHON`, `IVCBENCH_SCPERTURBENCH_EVAL_PYTHON` | per-family Python interpreter, pointing at that conda env's `bin/python` |
| `IVCBENCH_PRED_DUMP`, `IVCBENCH_PRED_DUMP_MEANS` | dump predictions to a directory; `=1` writes the **compact per-stratum mean bundles** (the deposited `predictions/` layer that `make reproduce-eval` re-scores) |
| `IVCBENCH_SEEDS`, `IVCBENCH_<MODEL>_EPOCHS` / `_MAXCELLS` / `_STEPS` (e.g. `IVCBENCH_CPA_EPOCHS`, `IVCBENCH_SCGEN_EPOCHS`, `IVCBENCH_STATE_STEPS`) | optional overrides; each runner has the paper defaults inlined, so these are only for re-tuning |
| `IVCBENCH_PA_REPO`, `IVCBENCH_PA_GO_MASK_NPZ`, `IVCBENCH_PA_ADAMSON_DIR` | PertAdapt upstream repo + its official artifacts |

So a from-scratch run of one model is: build that family's env from the upstream repo above → set its data-path
and `IVCBENCH_*_PY` variables → run its `scripts/` runner (which prints the exact command + hyperparameters it
used) → set `IVCBENCH_PRED_DUMP[_MEANS]` to refreeze bundles. `make reproduce-eval` then re-scores them GPU-free.

**Driven by the manifest (`make train` / `make train-all`).** Those per-model steps are now wrapped by a single
auditable manifest, [`scripts/train_manifest.csv`](scripts/train_manifest.csv), which lists one row per runnable
(model, cluster) unit with its conda env, the `$IVCBENCH_*` interpreter and data-path variables, and the exact
runner command. `make train MODEL=<name>` retrains and re-scores one model; `make train-all` runs the whole
pipeline (CPU `ivc` rows first, then the GPU families). Both preflight each unit against the environment and data
variables in the tables above, set `IVCBENCH_PRED_DUMP[_MEANS]` for you, run the manifest command, and finish with
`make reproduce-eval`; a unit whose env or data is missing on the host is reported as a clean skip naming the
variable to set, never a crash. Pass `--dry-run` to either to print the resolved plan without executing. The two
foundation models (scGPT, scFoundation) were run through the scPerturBench eval harness rather than an in-repo
runner, so the manifest leaves their command empty and the drivers report them as not-runnable from this
repository with a pointer back to this section.

**Stability.** We retrained every cell from scratch in an independent end-to-end run. The deterministic
baselines reproduced exactly; the trained models reproduced their reported Pearson-Δ to within run-to-run
variation (of order 0.003–0.01), and no verdict in the main figure changed.

---

## Notes

- All figure scripts resolve the repository root from their own location (`Path(__file__).resolve().parents[1]`),
  so they run from any checkout directory with no path edits.
- Heavy per-family model runners (`scripts/{cellot,scpram,state,cpa,chemcpa,graph,cinemaot,pertadapt}_*.py`
  and `scripts/cellot_donor_learning_curve.py`) are included for provenance. They require GPU, per-family
  conda environments, and the raw data, and read external paths from `$IVCBENCH_*` environment variables.
  They are **not** needed to rebuild any figure or Supplementary Table above; those are reproduced from the
  deposited tables only.
