# ivcbench

Repository for the analyses in:

**Toward Immune Virtual Cells: An Immune-Aware Benchmark of Perturbation-Prediction Generalization**

Chanhee Lee and Jae Yong Ryu

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20756159.svg)](https://doi.org/10.5281/zenodo.20756159)

This repository contains the benchmark code, deposited result tables, and figure scripts used for the
manuscript. The study is an evaluation of perturbation-prediction models in immune single-cell settings;
it does not introduce a new model.

The benchmark asks whether models generalize across several axes that are common in immune perturbation
studies: cell context, unseen perturbations, unseen donors, and readout modality. Models are compared with
simple pre-specified baselines, including a two-member "floor" used for the main pass/fail interpretation.

![Benchmark overview](results/_paper/figure1_benchmark_process.png)

## Reproducing the paper

Pick the path that matches what you have. Both are a handful of commands; the graduated detail behind them is
in the [reproduction ladder](#reproduction-ladder) below. `docker` works in place of `podman` throughout.

### No GPU: confirm every number (one command)

The repository ships a `Containerfile` that builds a GPU-free image carrying the deposited prediction bundles
and the analysis environment. One build and one run recompute the 35-cell census, with no conda, no GPU, and
no raw single-cell data:

```bash
podman build -t ivcbench . && podman run --rm ivcbench
```

It writes `reproduced_results.csv` (per-(model, task) Pearson-Δ); the C2 donor CellOT macro comes back at
0.3666. To keep the file on the host, mount a directory:

```bash
podman run --rm -v "$PWD/out:/ivcbench/out" ivcbench \
  .venv/bin/python scripts/reproduce_eval.py 'predictions/**/*.npz' -o out/reproduced_results.csv
```

### A GPU: reproduce the whole paper from nothing (three steps)

This trains every model and rebuilds the numbers and figures. The model families need conflicting CUDA and
PyTorch versions, so they are bundled as separate conda environments inside one image, the same arrangement
scPerturBench uses; the author publishes that image to Zenodo so you do not build the environments yourself.

```bash
# 1. get the all-environments image (every model env inside)
podman pull <zenodo-image>        # author builds + hosts it with scripts/build_train_image.sh

# 2. download all public raw data (Chen and Cano-Gamez need a manual login/DAC; the script says which)
make data

# 3. train every model and rebuild the numbers and the figure, inside the image
podman run --gpus all -v "$PWD:/ivcbench" ivcbench-train make reproduce-all
```

Step 3 runs on the GPU and takes a while: `make reproduce-all` chains the data download, the full retraining,
the GPU-free re-score, and the final figure in order. A model whose environment or data is missing is reported
as a clean skip naming what to set, not a crash. If you would rather build the environments on the host than
pull the image, the [reproduction ladder](#reproduction-ladder) covers that route.

## Repository contents

| Path | Contents |
|------|----------|
| `src/ivcbench/` | Core benchmark package: schemas, loaders, split construction, leak audit, baseline registry, metrics, statistics, and runners. |
| `scripts/` | Figure scripts, table assembly scripts, model-family runners, and dataset download/preprocessing scripts. |
| `results/` | Deposited paper-level result tables and generated figure files. Raw data and model checkpoints are not included. |
| `data/README.md` | Dataset accessions, download notes, and the corresponding scripts. |
| `predictions/` | Prediction-bundle format and small examples used by the reproduction tests. |
| `REPRODUCE.md` | Mapping from figures/tables to scripts and input result files. |

The release is intended to support inspection of the reported analyses without requiring reviewers to
rerun the original GPU-heavy model training jobs. Raw single-cell objects, checkpoints, large prediction
arrays, and local virtual environments are excluded.

## Main results

The main benchmark contains 35 model-by-task cells across five task clusters. In this release, the headline
tables are:

- [`results/_paper/cross_cluster_headline.csv`](results/_paper/cross_cluster_headline.csv)
- [`results/_paper/cross_cluster_headline.md`](results/_paper/cross_cluster_headline.md)
- [`results/_paper/descriptive_fit_matrix.csv`](results/_paper/descriptive_fit_matrix.csv)

The short version of the result is that most conditioned models do not clear both simple floor baselines.
Two model-by-task cells do: CellOT on the Soskic donor-held-out task and an FP-ridge chemistry prior on the
OP3 cell-context task. The CellOT effect is largest on the donor axis; the FP-ridge result is reported as a
model-level observation and does not survive all multiplicity corrections.

For unseen perturbations, the benchmark is largely negative. No method clears the floor on the unseen-gene
CRISPR tasks, explicit chemistry conditioning does not rescue the unseen-compound setting, and most
multi-dimensional immune programs do not transfer cleanly across contexts. The type-I interferon result is
treated separately because it is close to a coarse mean-shift response.

The release also examines three external or partially independent settings: a leave-one-cytokine
analysis on the Human Cytokine Dictionary summary table, the Chen FOXP3 Perturb-icCITE-seq checkpoint
replication, and a CellOT donor learning curve on the Soskic dataset. These analyses are included for
corroboration and provenance rather than as additional submitted supplementary figures.

## Installation

The core analysis environment is GPU-free and is used for split auditing, metric recomputation, table
assembly, and figure generation.

```bash
conda env create -f environment.yml
conda activate ivc
pip install -e .
```

or:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Several model families require their own environments because their upstream implementations have conflicting
Python, PyTorch, and CUDA requirements. The corresponding runner scripts are kept under `scripts/` to document
how the submitted result tables were generated. The core `environment.yml` and `requirements.txt` are not
meant to reproduce every heavy training environment.

## Quick check

```bash
make setup
make test
make pilot
```

`make test` runs the lightweight leak-audit and reproduction tests. `make pilot` runs a small synthetic
OP3-shaped example and writes a results table.

## Reproduction ladder

The release is built so you can reproduce as much or as little as you want, easiest first. Each rung goes one
step deeper, and you can stop at whichever one answers your question.

**Rung 1: one command, no local setup.** Build and run the container as shown in
[Reproduce in one command](#reproduce-in-one-command) above. This recomputes the 35-cell census from the
deposited bundles with no conda, no GPU, and no raw data, and writes `reproduced_results.csv`. For most
readers this is enough to confirm the headline numbers.

**Rung 2: a local environment.** If you would rather not use a container, install the core environment with
`environment.yml` or `requirements.txt` followed by `pip install -e .` (see [Installation](#installation)),
then run `make reproduce-eval`. This is the same recomputation as rung 1, run directly on your machine instead
of inside an image. The deterministic baselines and the deterministic heavy comparators (CellOT, scPRAM,
STATE, CPA on the donor axis) come back exactly, including the headline C2 donor CellOT macro of 0.3666, and
the stochastic models come back to within their seed variation. The distributional axis (energy distance) and
CellOT's bespoke asymmetric scorer are reproduced by other routes. `predictions/COVERAGE.md` is the
cell-by-cell account of what reproduces and how.

**Rung 3: figures and tables.** To rebuild the manuscript figures and the deposited derived tables from the
result files, run the scripts under `scripts/`. The figure-to-script mapping lives in
[`REPRODUCE.md`](REPRODUCE.md). This rung is also GPU-free and uses only the core environment from rung 2.

**Rung 4: retraining a model from scratch (GPU).** This is the heavy path, and it is the rung that needs a
GPU. There are two friction points, the raw data and the per-family environments, and each has a one-step
turnkey option. For the data, `make data` runs every public download script in turn and prints a summary of
what landed under `data/`; the two access-controlled deposits (Chen 2025 via DDBJ/GEA login, Cano-Gamez via
the EGA data-access committee) still need a manual login and are named in that summary rather than fetched.
For the environments, you have a choice. You can build each model family's environment yourself from its
upstream repository, which is the arrangement below and the one scPerturBench uses. Or you can pull the
all-environments training image, which bundles the per-family conda environments so the family interpreters
are already present at `/opt/conda/envs/<env>/bin/python`; that image is built once with
`scripts/build_train_image.sh` (it conda-packs each env and assembles `Containerfile.train`) and is hosted
externally on Zenodo rather than in this repository, because it is large, on the order of tens of gigabytes.
Either way, once the data is fetched and an environment is available, `make train MODEL=<name>` retrains a
single model and re-scores it and `make train-all` runs the whole pipeline.

Each model family has its own
conda environment, because the upstream implementations carry conflicting Python, PyTorch, and CUDA pins, so
one container cannot hold them all; this is the same arrangement scPerturBench uses. The small GPU-free eval
image in rung 1 is the verified, in-repo one; the all-environments training image is the separate large
Zenodo artifact and is not built or checked inside this repository. Two commands drive the retraining:
`make train MODEL=<name>` retrains a single model and re-scores it, and `make train-all` runs the whole
pipeline. Both orchestrate the per-family runners under `scripts/`: they preflight each unit, set
`IVCBENCH_PRED_DUMP[_MEANS]` so a retrained model refreezes its prediction bundles, run the exact runner
command, and finish with `make reproduce-eval`. They do not remove the need for the per-family environments,
the raw data, or a GPU; a unit whose environment or data is absent on the host is reported as a clean skip
with the variable to set, not a crash. The exact runner command for every model is listed in the
auditable [`scripts/train_manifest.csv`](scripts/train_manifest.csv); both commands accept `--dry-run` to
print the resolved plan without running anything. So a single model is
`make train MODEL=cellot` once that family's environment is built from its upstream repository and the raw
data is fetched and pointed at by the family's `$IVCBENCH_*` variables.
[`REPRODUCE.md`](REPRODUCE.md) carries the per-family environment table with each model's upstream
repository, the `$IVCBENCH_*` variable reference, and the per-model commands and hyperparameters. The two
foundation models (scGPT, scFoundation on the unseen-gene cluster) were run through the scPerturBench eval
harness rather than an in-repo runner; the manifest marks them as such and the drivers report them as
not-runnable from this repository with a pointer rather than failing.

## Benchmark workflow

The benchmark is organized as:

1. Load a real or synthetic perturbation dataset.
2. Build the task-specific split, such as held-out lineage, donor, gene, or compound.
3. Run the leak audit before scoring.
4. Apply the model applicability registry.
5. Fit each model or baseline using only the training part of the held-out fold.
6. Score response-direction, distributional, and immune-program metrics.
7. Estimate uncertainty over the biological unit of the task, not over technical seeds.
8. Assemble cluster-level tables and manuscript figures.

The relevant modules are in:

- `src/ivcbench/splits/`
- `src/ivcbench/baselines/`
- `src/ivcbench/metrics/`
- `src/ivcbench/clusters/`

Seeds are collapsed within the biological unit before bootstrap summaries are computed.

## Data

Raw data are not distributed in this repository. Public accessions, controlled-access notes, and download
scripts are listed in [`data/README.md`](data/README.md) and [`scripts/datasets.csv`](scripts/datasets.csv).

| Dataset | Accession / DOI | Notes |
|---|---|---|
| Kang 2018 PBMC IFN-beta | GEO `GSE96583` | Public GEO data. |
| Soskic CD4+ activation | raw `EGAD00001008197`; processed trynkalab h5ad | Used for the donor-held-out task. |
| Shifrut primary-T KO | GEO `GSE119450` | Public GEO data. |
| Schmidt primary-T CRISPRa | GEO `GSE190604` | Public GEO data. |
| McCutcheon primary-T CRISPRi/a | GEO `GSE218985` | Public GEO data. |
| Chen FOXP3 Perturb-icCITE-seq | DDBJ `PRJDB16517` / GEA `E-GEAD-648` | Controlled/login access; see `data/README.md`. |
| Human Cytokine Dictionary | Parse + Allen / theislab summary table | Used for the leave-one-cytokine analysis. |
| Frangieh Perturb-CITE-seq | Zenodo `10.5281/zenodo.13350497` | Tumour-cell checkpoint analysis. |
| McCarthy/OP3 PBMC chemical perturbation | GEO `GSE279945` | Chemical perturbation task. |

## Citation

```bibtex
@unpublished{Lee2026ImmuneVirtualCell,
  title  = {Toward Immune Virtual Cells: An Immune-Aware Benchmark of Perturbation-Prediction Generalization},
  author = {Lee, Chanhee and Ryu, Jae Yong},
  year   = {2026},
  note   = {Manuscript under review}
}

@software{ivcbench,
  title     = {ivcbench: An Immune-Aware Benchmark of Perturbation-Prediction Generalization},
  author    = {Lee, Chanhee and Ryu, Jae Yong},
  year      = {2026},
  version   = {1.0.2},
  doi       = {10.5281/zenodo.20756159},
  publisher = {Zenodo}
}
```

GitHub also reads citation metadata from [`CITATION.cff`](CITATION.cff).

## License

The code in this repository is released under the [MIT license](LICENSE). Dataset-specific terms remain with
the original data providers; see [`data/README.md`](data/README.md).

## Funding

This work was supported by the G-LAMP Program of the National Research Foundation of Korea (NRF), funded by
the Ministry of Education (No. RS-2025-25441317).
