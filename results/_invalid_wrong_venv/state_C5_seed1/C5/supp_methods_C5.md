# Supplementary Methods — C5 (Small-molecule perturbation prediction)

> Data source: op3_GSE279945.

*Shared metrics, split philosophy, applicability
gating, and statistics are in **Benchmark Methods (shared)**; this document covers C5-specific
datasets, data handling, and split definitions.*

## Datasets
- **Szałata 2024 / OP3** (GSE279945; public): PBMC, 3 donors, 144 benchmark compounds + DMSO and positive controls, 24 h, T/B/NK/myeloid readout. The only PBMC chemical single-cell perturbation resource at benchmark scale.

## Preprocessing and data handling
Each dataset is ingested into the unified schema (Benchmark Methods). For C5 we (i) restrict to
the official cluster scope; (ii) log-normalize counts and select highly variable genes shared across
datasets within the cluster; (iii) map control/vehicle populations to the control token; (iv) harmonize
cell-type annotations to a common coarse/fine hierarchy; and (v) for resource-scale data, subsample to
a fixed number of cells per condition before splitting. All fitting and normalization use train cells
only; no statistic is estimated on held-out test cells (enforced by the leak auditor).

## Split definitions (C5)
- **C5_global_compound_holdout** — held out: `perturbation` ∈ ['5-(9-Isopropyl-8-methyl-2-morpholino-9H-purin-6-yl)pyrimidin-2-amine', 'ABT-199 (GDC-0199)', 'AT 7867', 'AZD3514', 'BMS-536924', 'Cabozantinib', 'Clomipramine', 'Clotrimazole', 'GSK-1070916', 'HYDROXYUREA', 'IN1451', 'LY2090314', 'Lapatinib', 'Masitinib', 'Mometasone Furoate', 'Navitoclax', 'PRT-062607', 'Perhexiline', 'Pitavastatin Calcium', 'Proscillaridin A;Proscillaridin-A', 'Ricolinostat', 'SCH-58261', 'Selumetinib', 'TL_HRAS26', 'Tivozanib', 'Topotecan', 'Vorinostat', 'YK 4-279']; regime: side-info inference; applicability column: `C5_unseen_cpd` (n_train = 51181, n_test = 12333, 112 strata).
  held compound removed from every donor/cell-type/plate/replicate; predicted only from chemistry side-info. Tanimoto distance used post-hoc to stratify error, never as input.
- **C5_loct_B** — held out: `cell_type_coarse` ∈ ['B']; regime: control-inference-only; applicability column: `C1_LOCT` (n_train = 47074, n_test = 16080, 141 strata).
  held lineage's treated cells hidden from train/val/norm/model-selection; only its DMSO/control cells are inference input (control_inference_only).
- **C5_loct_Mono** — held out: `cell_type_coarse` ∈ ['Mono']; regime: control-inference-only; applicability column: `C1_LOCT` (n_train = 48011, n_test = 15143, 140 strata).
  held lineage's treated cells hidden from train/val/norm/model-selection; only its DMSO/control cells are inference input (control_inference_only).
- **C5_loct_NK** — held out: `cell_type_coarse` ∈ ['NK']; regime: control-inference-only; applicability column: `C1_LOCT` (n_train = 49059, n_test = 14095, 141 strata).
  held lineage's treated cells hidden from train/val/norm/model-selection; only its DMSO/control cells are inference input (control_inference_only).
- **C5_loct_T_cells** — held out: `cell_type_coarse` ∈ ['T cells']; regime: control-inference-only; applicability column: `C1_LOCT` (n_train = 46398, n_test = 16756, 141 strata).
  held lineage's treated cells hidden from train/val/norm/model-selection; only its DMSO/control cells are inference input (control_inference_only).

## Side information
Compound-side conditioning uses RDKit Morgan fingerprints, Murcko scaffolds, and LINCS L1000 MOA/target annotation as external side information. Tanimoto distance is used only post-hoc to stratify error, never as a model input.

## Compute
~70 A100-equivalent GPU-hours (≈8% of the Section-3 budget).
