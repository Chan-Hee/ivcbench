# C5 execution notes

> Run-level provenance; not submission-ready Methods.

Data source: op3_GSE279945.

## Recorded splits

- **C5_global_compound_holdout**: hold `perturbation` in ['5-(9-Isopropyl-8-methyl-2-morpholino-9H-purin-6-yl)pyrimidin-2-amine', 'ABT-199 (GDC-0199)', 'AT 7867', 'AZD3514', 'BMS-536924', 'Cabozantinib', 'Clomipramine', 'Clotrimazole', 'GSK-1070916', 'HYDROXYUREA', 'IN1451', 'LY2090314', 'Lapatinib', 'Masitinib', 'Mometasone Furoate', 'Navitoclax', 'PRT-062607', 'Perhexiline', 'Pitavastatin Calcium', 'Proscillaridin A;Proscillaridin-A', 'Ricolinostat', 'SCH-58261', 'Selumetinib', 'TL_HRAS26', 'Tivozanib', 'Topotecan', 'Vorinostat', 'YK 4-279']; inference: pooled controls and the required intervention-side input; registry task: `C5_unseen_cpd`; n_train = 51181, n_test = 12333, n_test_strata = 112.
  held compound removed from every donor/cell-type/plate/replicate; predicted only from chemistry side-info. Tanimoto distance used post-hoc to stratify error, never as input.
- **C5_loct_B**: hold `cell_type_coarse` in ['B']; inference: held-context controls; registry task: `C5_LOCT`; n_train = 47074, n_test = 16080, n_test_strata = 141.
  held lineage's treated cells hidden from fitting/validation; upstream processing shared; only its DMSO/control cells are inference input (control_inference_only).
- **C5_loct_Mono**: hold `cell_type_coarse` in ['Mono']; inference: held-context controls; registry task: `C5_LOCT`; n_train = 48011, n_test = 15143, n_test_strata = 140.
  held lineage's treated cells hidden from fitting/validation; upstream processing shared; only its DMSO/control cells are inference input (control_inference_only).
- **C5_loct_NK**: hold `cell_type_coarse` in ['NK']; inference: held-context controls; registry task: `C5_LOCT`; n_train = 49059, n_test = 14095, n_test_strata = 141.
  held lineage's treated cells hidden from fitting/validation; upstream processing shared; only its DMSO/control cells are inference input (control_inference_only).
- **C5_loct_T_cells**: hold `cell_type_coarse` in ['T cells']; inference: held-context controls; registry task: `C5_LOCT`; n_train = 46398, n_test = 16756, n_test_strata = 141.
  held lineage's treated cells hidden from fitting/validation; upstream processing shared; only its DMSO/control cells are inference input (control_inference_only).

## Data handling and compute

Consult the selected loader, adapter and manifest for this run's preprocessing,
side inputs, training configuration and recorded timing. No dataset inventory,
GPU-hour estimate or peak-memory guarantee is inferred from the cluster label.
The shared execution notes define the membership audit and metric scope.
