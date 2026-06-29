# `scripts/`

**Most users do not need anything in this directory.** To reproduce the paper's
results, run `make reproduce` (GPU-free) or `make test` from the repository root;
see the top-level [`README.md`](../README.md) and [`REPRODUCE.md`](../REPRODUCE.md).

The scripts here are grouped by purpose:

| Group | Files | What they do |
|---|---|---|
| **Reproduction core** | `reproduce_eval.py`, `check_consistency.py`, `assemble_cross_cluster.py`, `assemble_fit_matrix.py`, `run_cluster.py`, `sync_results_raw.py`, `c2_donor_paired.py`, `headline_multiplicity.py` | Re-score the deposited bundles, rebuild the 35-cell census + bundle-sourced tables, and assert the committed numbers. This is the GPU-free path behind `make reproduce` / `make check`. |
| **Figures** | `figure_*.py`, `make_figure2_landscape_verdict.py`, `figure_immune_blindspot.py` | Rebuild each main and supplementary figure from the deposited result tables. The figure-to-script map is in `REPRODUCE.md`. (Figure 1 is a hand-drawn schematic, not script-generated.) |
| **Model runners** (provenance) | `cellot_*.py`, `cpa_*.py`, `scgen_*.py`, `scpram_*.py`, `state_*.py`, `chemcpa_*.py`, `graph_frangieh.py`, `cinemaot_*.py`, `pertadapt_*.py` + `train_one.sh`, `reproduce_all.sh`, `train_manifest.csv` | Retrain each model family and re-dump its prediction bundle. These need the per-family conda environments, a GPU, and the raw data; they back `make train` / `make train-all`. See the env table in `REPRODUCE.md`. |
| **Per-cluster analysis** | `c1_*.py`, `c2_*.py`, `c3_*.py`, `c4_*.py`, `c5_*.py` | Build the leak-safe splits, baselines, and immune-novelty analyses that produce each cluster's deposited Supplementary Tables. |
| **New-data corroboration** | `newdata_cytokine_loco.py`, `chen_checkpoint_replication.py`, `cellot_donor_learning_curve.py` | The three independent-dataset analyses (Supplementary Notes S11–S13). |
| **Data download** | `download_all.sh`, `download_public.sh`, `datasets.csv` | Fetch the public datasets. Controlled-access datasets are documented in [`../data/README.md`](../data/README.md). |
| **Manuscript-build helpers** | `normalize_plate.py`, `refresh_docx_figures.py`, `update_docx_captions.py`, `clean_paper_temp.py`, `figure_qc.py` | Author-only utilities that normalize the figure plates and embed them in the manuscript document. Not needed to reproduce any benchmark result. |

Every runner script inlines its own exact command, hyperparameters, and seeds in
its header, and reads external paths from `$IVCBENCH_*` environment variables, so
each runs on any host with no path edits.
