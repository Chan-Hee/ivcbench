# ivcbench

An immune-aware benchmark for testing whether single-cell perturbation-prediction
models generalize across immune contexts, unseen perturbations, unseen donors,
and readout modalities.

ivcbench accompanies **Toward Immune Virtual Cells: An Immune-Aware Benchmark of
Perturbation-Prediction Generalization** by Chanhee Lee and Jae Yong Ryu. This
repository contains benchmark code, deposited prediction bundles, result tables,
and figure scripts. It evaluates existing models; it does not introduce a new
model.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20756042-1682D4)](https://doi.org/10.5281/zenodo.20756042)

![Benchmark framework](results/_paper/Figure1.png)

## What You Can Reproduce

The primary public reproduction path is GPU-free. It re-scores the deposited
prediction bundles, rebuilds the 35-cell headline census, and checks the result
against the committed paper tables.

```bash
podman build -t ivcbench .
podman run --rm ivcbench
```

`docker` can be used in place of `podman`.

Without a container:

```bash
make setup
make reproduce
```

The consistency gate prints `DEPOSIT CONSISTENCY: PASS` when the deposited
bundles reproduce the committed headline numbers.

This path covers the 35 model-by-task Pearson-delta census and the floor-clearance
verdicts used for the main conclusions. The deposited bundles are compact
per-stratum mean bundles, so they reproduce Pearson-Δ exactly but do not contain
the per-cell prediction clouds required to recompute energy distance. The
distributional-fidelity axis is reproduced from the deposited result tables
(`results/*/results_raw.csv` and Supplementary Table S8); regenerating it from
raw predictions requires the larger per-cell bundle/retraining path described in
[REPRODUCE.md](REPRODUCE.md) and [predictions/COVERAGE.md](predictions/COVERAGE.md).
The GPU-free path does not retrain the original models or download raw
single-cell data.

## Main Results

The benchmark asks whether methods beat simple, pre-specified floor baselines
on five immune perturbation tasks:

| Task | Generalization question |
|---|---|
| T1 | Cell-context transfer under cytokine stimulation |
| T2 | Donor-held-out CD4+ activation |
| T3 | Unseen-gene CRISPR perturbations |
| T4 | Complex immune checkpoint and modality stress tests |
| T5 | Small-molecule perturbations and cell-context transfer |

Headline artifacts:

- [cross_cluster_headline.csv](results/_paper/cross_cluster_headline.csv)
- [cross_cluster_headline.md](results/_paper/cross_cluster_headline.md)
- [descriptive_fit_matrix.csv](results/_paper/descriptive_fit_matrix.csv)
- [predictions/COVERAGE.md](predictions/COVERAGE.md)

In brief, most conditioned models do not clear the simple floor baselines. Two
model-by-task cells do: CellOT on the Soskic donor-held-out task and an FP-ridge
chemistry prior on the OP3 cell-context task. The unseen-perturbation settings
are largely negative, including the unseen-gene CRISPR tasks and the
unseen-compound setting.

## Repository Map

| Path | Purpose |
|---|---|
| `src/ivcbench/` | Core package: schemas, loaders, split construction, leak audit, metrics, baselines, and runners. |
| `predictions/` | Deposited compact prediction bundles used by the GPU-free Pearson-Δ reproduction path. |
| `results/` | Paper-level result tables and generated figures. |
| `scripts/` | Reproduction, figure, download, and retraining/provenance scripts. |
| `model_runners/` | Thin wrappers for model-family environments used during retraining. |
| `data/README.md` | Dataset accessions, access notes, and raw-data layout. |
| `REPRODUCE.md` | Detailed reproduction and retraining guide. |

## Data

Raw data are not distributed in this repository. Public datasets can be fetched
with:

```bash
make data
bash scripts/download_all.sh --list
```

Some source datasets require login or data-access approval. See
[data/README.md](data/README.md) and [scripts/datasets.csv](scripts/datasets.csv)
for the per-dataset accessions, access status, and loaders.

The deposited prediction bundles under `predictions/` are sufficient for the
GPU-free headline Pearson-Δ reproduction path. Distributional-fidelity tables are
deposited under `results/` and are not recomputed from the compact bundles.

## Common Commands

```bash
make test            # leak-audit and smoke tests
make reproduce       # GPU-free bundle re-scoring plus consistency gate
make reproduce-eval  # write reproduced_results.csv from prediction bundles
make data            # download public raw datasets
make train MODEL=cellot ARGS=--dry-run
```

Retraining commands are provenance tools. They expect the required raw data,
model-family environment, and hardware to already be available.

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
  version   = {1.1.6},
  doi       = {10.5281/zenodo.20756042},
  publisher = {Zenodo}
}
```

GitHub also reads citation metadata from [CITATION.cff](CITATION.cff).

## License

The code is released under the [MIT license](LICENSE). Dataset-specific terms
remain with the original data providers.

## Funding

This work was supported by the G-LAMP Program of the National Research Foundation
of Korea (NRF), funded by the Ministry of Education (No. RS-2025-25441317); the
Korea Health Industry Development Institute (KHIDI), funded by the Ministry of
Health and Welfare, Republic of Korea (No. RS-2025-25459520); and the National
Research Foundation of Korea (NRF) grants funded by the Korean Government (MSIT;
grant no. RS-2025-02304296).
