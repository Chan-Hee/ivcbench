# Which file backs which printed table

`supplementary_tables/` holds the table of record for every Supplementary Table, and
`supplementary_tables/MANIFEST.csv` maps each one to its file. Several older intermediates sit
beside them in this directory under similar names. They are kept because figures and notes were
computed from them, but they are **not** the tables the paper prints, and two of them carry a
roster or a holdout slice the printed table does not.

| File | Status |
|---|---|
| `supplementary_tables/Supplementary_Table_S*.csv` | **Table of record.** Matches the printed table. |
| `Supplementary_Table_S7_OP3_programs.csv` | Superseded intermediate, different schema. It was computed over a roster that still included the CPA, scGen and STATE drug runs the interface rule excludes, so its per-program values differ from printed Table S7. Use `supplementary_tables/Supplementary_Table_S7.csv`. |
| `supplementary_tables/Supplementary_Table_S13.csv` | The **10 %** leave-one-gene-out slice over the current census roster. Printed Table S13 is the **50 %** slice over the roster the submitted panel scored, which its caption and Note S4 both state. The two are different analyses of the same axis, not two versions of one table; the 10 % slice is the one the census verdicts use. |
| `Supplementary_Table_S6_marker_readout.csv`, `Supplementary_Table_S20_panel_checks.csv` | Full-precision sources the printed tables are re-sourced from. Authoritative for precision. |
| `Supplementary_Table_S3_descriptive_fit_matrix.csv`, `Supplementary_Table_S12_T3_programs.csv`, `Supplementary_Table_S14_Tanimoto_current.csv`, `Supplementary_Table_S17_effect_stratification.csv` | Superseded 2026-09-09 snapshots kept for provenance. Each is smaller or older than the printed table and disagrees with it: S3 has 32 rows against 35, S12 a different schema, S14 the pre-STATE T5u roster including CINEMA-OT, S17 the pre-re-run margins. The producers beside them (`descriptive_fit_matrix.csv`, `op3_tanimoto_sensitivity.csv`, `t3_effect_stratification.csv`) are what the build reads. |
| `Supplementary_Table_S8_energy_distance.csv` | The producer. `supplementary_tables/Supplementary_Table_S8.csv` is copied from it every build, so the two are identical; before 2026-09-16 the deposit had no such path and sat 32 rows behind. |

Numbers in `supplementary_tables/` are printed at the precision the paper uses. Where full
precision matters, read the analysis CSVs in this directory (`census_uncertainty.csv`,
`census_unit_scores.csv`, `headline_multiplicity_adjusted.csv`, `matched_donor_curve_summary.csv`,
`donor_validation_summary.csv`, `compute_evidence_summary.csv`).

`panel_precision_attenuation.csv` carries one row per census evaluation (58) with a written
`power_statement` and `attenuation_statement` for each, alongside the interval, the margin
scale and the matched-mask split-half precision diagnostic they describe. It is the file
Note S2 points at for the per-cell power and attenuation statements; `status` there is the
census provenance label, which supersedes the older per-model summaries in this directory
(`cellot_summary.csv` still carries the submitted `adapted` label for the donor split).
