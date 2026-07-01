# Reproducing the figures and tables

Every data-driven figure and Supplementary Table is rebuilt **from the deposited result files in `results/`** (Figure 1 is a static hand-drawn schematic); no
raw data and no model re-runs are required. Run each script from the repository root with the `ivc`
environment active (`conda activate ivc` or the `.venv`), e.g.:

```bash
python scripts/make_figure2_landscape_verdict.py
```

Figure scripts read deposited CSV/JSON and write PNG+PDF into `results/_paper/`. Analysis scripts
(listed in the "computed by" column) recompute a derived table from the per-cluster `results_raw.csv`
and the real loaders; they are included for provenance but their outputs are already deposited.

**The headline census is bundle-sourced.** The 35-cell table `results/_paper/cross_cluster_headline.csv`
(Supplementary Table S4) and `descriptive_fit_matrix.csv` (Supplementary Table S3) are
re-scored directly from the deposited prediction bundles under `predictions/`, the same GPU-free path a
reviewer runs. Those deposited bundles are compact per-stratum mean bundles, so this path reproduces
**Pearson-Δ exactly** but does **not** recompute energy distance, which requires per-cell prediction clouds
and the train-cloud PCA basis. `make census` rebuilds the bundle-sourced Pearson-Δ tables
(`scripts/assemble_cross_cluster.py`, `assemble_fit_matrix.py`), and `make check` (also run by `make test`)
re-derives the census from the bundles and fails the build if the committed numbers have drifted from it,
so the table a reader reproduces and the table the paper reports cannot diverge. The C2 donor paired
statistics (S9, S10) and their multiplicity adjustment (S11) are likewise computed from the per-donor bundle
scores. The per-cluster `results_raw.csv` tables remain the provenance for the distributional-fidelity axis
(including Supplementary Table S8) and for figure scripts; `scripts/sync_results_raw.py` re-derives their
response-direction `pearson_delta` from the bundles (to the printed precision) so the figures read the same
Pearson-Δ numbers as the census, and `make check` fails if any cell has drifted. Regenerating energy
distance from raw predictions requires re-running the model scripts with `IVCBENCH_PRED_DUMP_MEANS` unset,
which writes the larger per-cell bundles described in `predictions/COVERAGE.md`.

**Final manuscript layout (binding).** The submission is **3 main figures + 8 supplementary figures
(S1–S8) + 14 Supplementary Tables (S1–S14)**:

- **Fig 1**: benchmark workflow schematic (hand-drawn; a static figure, not script-generated)
- **Fig 2**: generalization map + CellOT donor CDF (`make_figure2_landscape_verdict.py`)
- **Fig 3**: immune blind-spot map (`figure_immune_blindspot.py`)

The full per-cluster figure plate (`figure_landscape / ranking / cellcontext / perturbation /
donor_decision / within_family_fit`) is retained for provenance and feeds several of the S-figures
below; the 8 submitted supplementary figures (S1–S8) are embedded in
`BiB_submission/supplementary/Supplementary_Material.docx`. They are journal-formatted renders of the
`results/_paper/` figures (margin-normalized and exported, so not byte-identical to every source PNG; only
S2 == `figure_cellcontext.png` happens to match). The additional analyses listed further below
(chemistry-channel collapse, program-dimensionality, and the three new-data analyses) are deposited as
release provenance; in the submitted supplement they appear as Supplementary Notes/Tables, not as numbered
supplementary figures.

---

## Main figures

| Main figure | Script (`scripts/`) | Output (`results/_paper/`) | Reads (deposited) |
|---|---|---|---|
| **Fig 1: Benchmark workflow** | hand-drawn (not script-generated) | static figure (`results/_paper/Figure1.png`; submitted as `BiB_submission/figures/Figure1.tiff`) | — |
| **Fig 2: Generalization map + donor CDF** | `make_figure2_landscape_verdict.py` | `figure2_landscape_verdict.{png,pdf,tiff}` | (a) 35-cell map (29 conditioned + 6 diagnostic comparators) from `cross_cluster_headline.csv`; (b) CellOT donor CDF from `cellot_soskic_raw.csv` + `cellot_summary.csv` + `cellot_vs_floor_donor_paired.csv`, paired-Wilcoxon p in `headline_multiplicity_adjusted.csv` |
| **Fig 3: Immune blind-spot map** | `figure_immune_blindspot.py` | `figure_immune_blindspot.{png,pdf}` | `immune_novelty/T1_C4_per_marker_protein_recovery.csv`, `immune_novelty/T2_per_program_AUCell_map.csv`, `immune_novelty/T3_per_lineage_predictability.csv`, `results/C3/program_null.csv`, `results/C5/results_raw.csv`, `multiseed_scgen_summary.csv` |

---

## Supplementary figures (submitted S1–S8) + additional release figures

*Note: `BiB_submission/…` paths name files in the journal submission package, which is separate from this
code archive; this release ships the regenerating scripts and the backing result files named below.*

**S1–S8 are the submitted supplementary figures** (embedded in the journal submission
`Supplementary_Material.docx`; this release regenerates their sources under `results/_paper/`,
margin-normalized so not byte-identical). Each row: submitted S-number → figure script → backing data (all
present in this release).

| S# | Content | Figure script (`scripts/`) | Computed by → backing data (deposited) |
|---|---|---|---|
| **S1** | Observed-effect reliability ceiling by task | `figure_reliability_ceiling.py` | `reliability_ceiling.py` → `immune_novelty/reliability_ceiling.csv`; also `cross_cluster_headline.csv` |
| **S2** | Cell-context transfer detail (FP-ridge / type-I IFN) | `figure_cellcontext.py` | `results/C5/results_raw.csv`, `immune_novelty/T3_per_lineage_predictability.csv`, `results/C5/ifn_shuffle_null.csv` |
| **S3** | Per-marker surface-protein recovery (Frangieh CITE; PD-1 vs PD-L1, RNA-vs-surface) | `figure_c4_pdl1_assay_power.py` | `c4_pdl1_assay_power.py` → `c4_surface_marker_CIs.csv`, `c4_rna_vs_surface_decoupling.csv` (from `results/C4/cite_marker_recovery.csv`) |
| **S4** | Chen Perturb-icCITE-seq checkpoint replication (n = 2) | `figure_chen_checkpoint_replication.py` | `chen_checkpoint_replication.py` → `results/newdata/chen_cite_marker_recovery.csv` (+ Frangieh comparator `results/_paper/c4_surface_marker_CIs.csv`) |
| **S5** | Program recovery vs program properties (magnitude / dimensionality) | `figure_program_dimensionality.py` | `program_recovery_vs_dimensionality.py` → `program_recovery_vs_dimensionality.csv`; recovery from `immune_novelty/T2_per_program_AUCell_map.csv` |
| **S6** | CellOT donor learning curve (Soskic donor-count grid) | `figure_cellot_donor_learning_curve.py` | `results/newdata/cellot_donor_learning_curve.csv` (GPU runner = provenance; figure rebuilds GPU-free) |
| **S7** | Nearest-training-gene prior on the unseen-gene axis | `figure_c3_nearest_gene.py` | `c3_nearest_gene_baseline.py` → `results/C3/nearest_gene_baseline.csv`; `c3_nearest_gene_summary.py` → `c3_nearest_gene_summary.csv`; also `results/C3/results_raw.csv` |
| **S8** | Unseen-cytokine LOCO (Human Cytokine Dictionary; two conditioning regimes) | `figure_newdata_cytokine_loco.py` | `newdata_cytokine_loco.py` → `results/newdata/cytokine_loco_per_held.csv`, `cytokine_loco_per_celltype.csv` |

**Additional release figures (provenance — NOT submitted supplementary figures).** Retained for
provenance; in the manuscript their content appears in Supplementary Notes/Tables rather than as numbered
figures: method×split ranking heatmap (`figure_ranking.py`), within-family consistency + fit-matrix
(`figure_within_family_fit.py`), perturbation/modality degenerate-axis panels (`figure_perturbation.py`),
C3 unseen-gene predictability probe (`c3_predictability_analysis.py`), and chemCPA chemistry-channel
collapse (`figure_chemcpa_collapse.py`).


## Supplementary Tables (S1–S14)

Each Supplementary Table is published with the submission and backed by the listed result file in this
release. Thirteen of the fourteen tables are embedded in `Supplementary_Material.docx`; the descriptive
fit-matrix (S3) is provided only as a machine-readable CSV, and the energy-distance table (S8) is
embedded and also deposited as a CSV. S1 and S2 are curated inventory tables exported verbatim from
the submitted supplement into `results/_paper/Supplementary_Table_S1_dataset_inventory.csv` and
`results/_paper/Supplementary_Table_S2_method_inventory.csv`; the scored-dataset download manifest
remains `scripts/datasets.csv`. The rest are mechanically assembled from the deposited per-cluster leaderboards
and immune-novelty tables. The per-dataset CRISPR breakdown (S13) and the Tanimoto-distance
negative-control (S14) tables — placed at the end of the supplement — are backed by the deposited
leave-one-gene-out leaderboard (`results/C3/results_raw.csv`), `results/C5/tanimoto_percompound.csv`,
and the chemCPA by-compound provenance table (`results/_paper/chemcpa_op3_unseen_compound_by_unit.csv`).
*(The supplementary **Figure** series in the sections above is independent of the table numbering: the
submitted figures are S1–S8 (additional release figures are provenance, not submitted), while
Supplementary **Tables** are S1–S14.)*

| Table | Content | Journal supplement | Backing data / script (`scripts/`) |
|---|---|---|---|
| **S1** | 21-dataset immune inventory (anchors + curation criteria) | embedded in `Supplementary_Material.docx` | `results/_paper/Supplementary_Table_S1_dataset_inventory.csv`; scored-dataset access manifest in `scripts/datasets.csv` |
| **S2** | Surveyed method + comparator inventory with benchmark applicability | embedded in `Supplementary_Material.docx` | `results/_paper/Supplementary_Table_S2_method_inventory.csv` |
| **S3** | Descriptive fit-matrix (a-priori expectation vs observed beats-floor per family×task) | `Supplementary_Table_S3_descriptive_fit_matrix.csv` | `descriptive_fit_matrix.csv` via `assemble_fit_matrix.py` |
| **S4** | Per-(model, split) headline census (35-cell Pearson-Δ vs floor: 29 conditioned + 6 diagnostic comparators) | embedded in `Supplementary_Material.docx` | `cross_cluster_headline.csv` via `assemble_cross_cluster.py` |
| **S5** | OP3 fine-lineage cell-context transfer (FP-ridge vs the two universal-floor members and binding maximum, six fine OP3 lineages) | embedded in `Supplementary_Material.docx` | `Supplementary_Table_S5_op3_fine_lineage.csv` (FP-ridge per-lineage scores) |
| **S6** | Per-surface-marker protein recovery (PD-1/PD-L1; effect-size vs sign-match) | embedded in `Supplementary_Material.docx` | `immune_novelty/T1_C4_per_marker_protein_recovery.csv` + `c4_surface_marker_CIs.csv` via `c4_per_marker.py` |
| **S7** | Per-immune-program AUCell-Δ recovery map | embedded in `Supplementary_Material.docx` | `immune_novelty/T2_per_program_AUCell_map.csv` |
| **S8** | Distributional-fidelity axis: per-(task, split, model, modality) energy distance + Pearson-Δ | `Supplementary_Table_S8_energy_distance.csv` | `assemble_s8_energy_distance.py` from `results/{C1,C3,C4,C5}/results_raw.csv` |
| **S9** | Donor axis: CellOT vs floor, paired per-donor (n = 106; cell-mean gap +0.107 **and** matched-baseline gap +0.100 [CI]) | embedded in `Supplementary_Material.docx` | `cellot_vs_floor_donor_paired.csv` via `scripts/c2_donor_paired.py` (per-donor bundle scores) |
| **S10** | Donor axis: scPRAM vs CellOT, paired per-donor | embedded in `Supplementary_Material.docx` | `scpram_vs_cellot_donor_paired.csv` via `scripts/c2_donor_paired.py` (per-donor bundle scores) |
| **S11** | Headline-survivor table after BH/Holm multiplicity correction (two pre-specified families) | embedded in `Supplementary_Material.docx` | `headline_multiplicity.py` byte-reproduces `results/_paper/headline_multiplicity_adjusted.csv` (two-family BH/Holm; H4 FP-ridge BH=0.0625 does not survive) |
| **S12** | Per-program AUCell recovery vs observed-shift magnitude (r = +0.87 relationship) + dimensionality proxies | embedded in `Supplementary_Material.docx` | `program_recovery_vs_dimensionality.csv` via `program_recovery_vs_dimensionality.py` |
| **S13** | Per-dataset CRISPR leave-one-gene-out breakdown | embedded in `Supplementary_Material.docx` | `results/C3/results_raw.csv` and `results/_paper/cross_cluster_headline.csv`; summary helper `c3_nearest_gene_summary.py` writes `results/_paper/c3_nearest_gene_summary.csv` |
| **S14** | OP3 Tanimoto-distance negative control | embedded in `Supplementary_Material.docx` | `results/C5/tanimoto_percompound.csv`; chemCPA per-compound values are in `results/_paper/chemcpa_op3_unseen_compound_by_unit.csv`; multiplicity row reproduced by `headline_multiplicity.py` |

---

## New-data corroboration analyses (submitted figures S4, S6, S8)

Three new-data analyses extend the benchmark's central law (*conditioning helps cell/donor context, not
unseen biology*) onto independent datasets; in the submission they are Supplementary Figures **S8, S4, S6**:

- **S8: unseen-cytokine LOCO** (`newdata_cytokine_loco.py`): leave-one-cytokine-out over the Human Cytokine
  Dictionary summary table (derived from the Oesinghaus et al. 2025 resource; obtained separately, see above).
  It separates the two conditioning regimes: annotation-only "truly novel" cytokines sit at/below the floor
  (the law holds), while a cytokine observed in *other* cell types transfers above it. Outputs in `results/newdata/`.
- **S4: Chen checkpoint replication** (`chen_checkpoint_replication.py`): an n = 2 surface-protein readout
  replication on the Chen FOXP3 Perturb-icCITE-seq data (`data/C3/chen`, login-gated DDBJ/GEA), scored on the
  same surface-marker recovery metric as C4/Frangieh.
- **S6: CellOT donor learning curve** (`cellot_donor_learning_curve.py`): the donor-count sweep {8,16,32,64,96}
  on Soskic, two seeds on two GPUs, showing how the CellOT donor-axis advantage scales with training-donor count.
  The merged/summary CSVs are deposited; the figure rebuilds with no GPU.

The analysis scripts read **raw** data and are provenance; the three figure scripts read only the deposited
`results/newdata/` tables (+ one Frangieh comparator in `results/_paper/`) and rebuild without raw data or a GPU.


## Regenerating the prediction bundles by retraining (provenance)

This section documents how the deposited prediction bundles were produced; it is provenance, not a
one-command reproduction path. The reproduction of record for the paper's numbers is the GPU-free bundle
re-score described above. Everything above rebuilds with no GPU. The bundle-sourced tables include the
descriptive fit matrix (S3), the headline census (S4), the OP3 fine-lineage check (S5), the donor paired
checks (S9, S10), and the multiplicity table (S11); the figures and the remaining tables (including the
S8 energy-distance table and the additional release analyses) are assembled from the deposited result CSVs,
not from the bundles. To regenerate the predictions themselves by retraining every model, you need the raw
data and the per-family environments.

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

**Release packaging.** Do not zip a live working tree for code deposition: local `data/`, `build/`,
`.venv/`, checkpoints, and training-image tarballs can be tens of gigabytes and are intentionally ignored.
Package the code release from tracked files only, for example with `git archive`, and keep raw data and
training-image artifacts as separate external deposits.

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
| `scfoundation` | scFoundation, PertAdapt | `biomap-research/scFoundation`; PertAdapt (Bai et al. 2025); artifact requirements and access notes in `data/pertadapt/official/PROVENANCE.md` (`scripts/pertadapt_validate.py`) |

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
variable to set, never a crash. To print the resolved preflight plan without executing, forward `--dry-run`
through `ARGS` (`make train MODEL=<name> ARGS=--dry-run`, `make train-all ARGS=--dry-run`) or call the scripts
directly (`scripts/train_one.sh <name> --dry-run`, `scripts/reproduce_all.sh --dry-run`); a bare
`make train-all --dry-run` is GNU make's own dry run, not the script preflight. The two
foundation models (scGPT, scFoundation) were run through the external scPerturBench C3 eval harness rather
than an in-repo retraining command, so the manifest intentionally leaves those commands empty. Their deposited
prediction bundles and result tables are still re-scored by `make reproduce-eval`; full retraining for those
two cells requires the external harness/provenance environment rather than `make train`.

**Determinism and run-to-run variation.** The GPU-free path is exact by construction: the census is
re-scored from the deposited bundles, so the headline numbers and floor verdicts reproduce bit-for-bit.
The deterministic components of the retraining path (the floor baselines, FP-ridge, and the linear-shift
diagnostic) likewise re-dump their bundles exactly. Trained models are stochastic and reproduce their
reported Pearson-Δ only to within run-to-run variation (of order 0.003–0.01), which is why the
bundle-level GPU-free path, not retraining, is the reproduction of record for the paper's numbers.

---

## Notes

- All figure scripts resolve the repository root from their own location (`Path(__file__).resolve().parents[1]`),
  so they run from any checkout directory with no path edits.
- Heavy per-family model runners (`scripts/{cellot,scpram,state,cpa,chemcpa,graph,cinemaot,pertadapt}_*.py`
  and `scripts/cellot_donor_learning_curve.py`) are included for provenance. They require GPU, per-family
  conda environments, and the raw data, and read external paths from `$IVCBENCH_*` environment variables.
  They are **not** needed to rebuild any figure or Supplementary Table above; those are reproduced from the
  deposited tables only.
