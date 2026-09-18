# Immune-program reanalysis — source data for Figure 3, Figure S6 and Tables S7, S12, S22

The revision replaced the primary immune-program readout. Figure 3 now scores each program as the
**equal-weight mean-expression shift** of its measured markers, calculated identically from the
predicted and the observed mean profile, on the same supported conditions and eligible units for
every model. The original top-5% rank score is retained as a sensitivity analysis, and the donor
analysis (Supplementary Note S6) keeps its own rank-based program-shift error.

These files are the full-precision record behind that analysis: the printed Supplementary Tables
S7, S12 and S22 round them, and Supplementary Notes S6–S7 describe the protocol. The plate itself
is deposited as `../figure_immune_blindspot.{png,pdf,tiff}` (Figure 3).

The analysis code is the revision package, which is not part of this archive. This directory
therefore supports **checking** the published values, not re-deriving them from cells.

## Unit values, membership and support

| file | rows | what it holds |
|---|---|---|
| `program_unit_metrics.csv` | 3648 | every model × program × unit correlation, with its n and estimability |
| `model_program_summary.csv` | — | the per-model, per-program summary the figure and tables read |
| `support_by_unit.csv` | 17 | the supported conditions per unit, identical across models |
| `observed_unit_selection.csv` | 296 | which units entered each program comparison, and why |
| `gene_coverage.csv` | 148 | program membership after intersection with measured genes and the benchmark mask |
| `source_alignment.csv` | — | alignment of each unit to its census cell |
| `census_roster.csv` | 58 | the 58-cell census roster this analysis is scoped to |
| `lineage_all_models.csv`, `lineage_stratum_scores.csv` | — | per-lineage inputs for panel (d) |

## Exclusion reasons

`figure3c_undefined.csv` lists the entries omitted from the plot with the reason for each;
`observed_unit_selection.csv` carries the eligibility outcome for every unit. Undefined reference
correlations are omitted from the plot and retained here, never replaced by zero.

## Original-rank sensitivities

`Table_S7_programs_all_sensitivities.csv` (312 rows) and
`Table_S12_programs_all_sensitivities.csv` (480 rows) give both operators on identical units;
`Figure_S6_source.csv` (552 rows) is the Figure S6 panel source;
`rank_degeneracy_counts.csv` records where the rank operator is degenerate.

## Response magnitude (Table S22)

`Table_S22_shared_support.csv` and `Table_S22_full_census_existing_estimator.csv` are the two
supports; `Table_S22_primary_and_full.csv` is the printed pairing and
`magnitude_sensitivity_protocol.json` the protocol. `legacy_full_support_magnitude*.csv` retain the
earlier full-support calculation for comparison.

## Figure panel sources

`figure3b_points.csv` (83), `figure3b_best.csv`, `figure3c_points.csv` (11), `panel_c.csv` and
`figure3d_lineage.csv` (12) are the plotted values of Figure 3b–d.

`ANALYSIS_NOTES.md` and `METHOD_REVIEW.md` are the analysis notes and the method review carried
over from the revision package.
