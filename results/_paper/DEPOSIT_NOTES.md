# Which file backs which printed table

`supplementary_tables/` holds the table of record for every Supplementary Table, and
`supplementary_tables/MANIFEST.csv` maps each one to its file. Several older intermediates sit
beside them in this directory under similar names. They are kept because figures and notes were
computed from them, but they are **not** the tables the paper prints, and two of them carry a
roster or a holdout slice the printed table does not.

| File | Status |
|---|---|
| `supplementary_tables/Supplementary_Table_S*.csv` | **Table of record.** Matches the printed table. |
| `superseded_Table_S7_OP3_programs.csv` | Superseded intermediate, different schema. It was computed over a roster that still included the CPA, scGen and STATE drug runs the interface rule excludes, so its per-program values differ from printed Table S7. Use `supplementary_tables/Supplementary_Table_S7.csv`. |
| `supplementary_tables/Supplementary_Table_S13.csv` | **Table of record**, the **50 %** slice the paper prints, derived from `results/C3/results_raw.csv` each build. It used to hold the 10 % slice instead, so the deposited file shared no value with the table it claimed to back. |
| `t3_by_dataset.csv` | The **10 %** leave-one-gene-out slice over the current census roster — the one the census verdicts use. A different analysis of the same axis, not another version of printed Table S13. |
| `Supplementary_Table_S6_marker_readout.csv`, `Supplementary_Table_S20_panel_checks.csv` | Full-precision sources the printed tables are re-sourced from. Authoritative for precision. |
| `superseded_Table_S3_descriptive_fit_matrix.csv`, `superseded_Table_S12_T3_programs.csv`, `superseded_Table_S14_Tanimoto_current.csv`, `superseded_Table_S17_effect_stratification.csv` | Superseded 2026-09-09 snapshots kept for provenance. Each is smaller or older than the printed table and disagrees with it: S3 has 32 rows against 35, S12 a different schema, S14 the pre-STATE T5u roster including CINEMA-OT, S17 the pre-re-run margins. The producers beside them (`descriptive_fit_matrix.csv`, `op3_tanimoto_sensitivity.csv`, `t3_effect_stratification.csv`) are what the build reads. |
| `Supplementary_Table_S8_energy_distance.csv` | The producer. `supplementary_tables/Supplementary_Table_S8.csv` is copied from it every build, so the two are identical; before 2026-09-16 the deposit had no such path and sat 32 rows behind. |
| `chemcpa_op3_unseen_compound_summary.csv` | **Correction, 2026-09-16.** Its `CPA_existing_score` (0.158691) was taken from that run's `pearson_delta_ontarget` column rather than `pearson_delta`, the census metric every other anchor in `scripts/chemcpa_evaluate.py` uses; on-target exclusion is a CRISPR convention and does not apply to a compound split. The like-for-like value is **0.106687**, so `chemCPA_minus_CPA_existing` is **-0.0067**, not -0.0587, and the verdict sentence's "0.159 (-0.047)" should read "0.107 (+0.005)". The anchor in the script is fixed; the CSV is left as the record of the run it describes because re-deriving it needs the model. No number in the manuscript, supplement, response letter, or any printed or deposited table depends on it -- the chemCPA note reports from `chemcpa_op3_unseen_compound_by_unit.csv`. |

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

## The immune-program reanalysis

`immune_program_revision/` holds the full-precision record behind Figure 3b-d, Figure S6 and
Supplementary Tables S7, S12 and S22: per-unit correlations, program membership after masking,
the supported conditions and every exclusion reason, the retained rank-score sensitivities, and
the frozen protocol. It is not a summary of those tables -- it is what they round.

Two scripts in this archive recompute it from the deposited prediction bundles alone, on CPU and
without fitting anything: `scripts/program_analysis.py` and `scripts/lineage_analysis.py`. Each
compares what it computes with the files in that directory and prints a verdict, and `make check`
runs both. `census_roster.csv` in there is `census_uncertainty.csv` verbatim, kept beside the
analysis it scopes.

## Supplementary figure artwork

The supplement embeds its own PNGs; only three have a byte-identical twin here
(`figure_cellcontext.png` = Fig. S1, `figS_c3_nearest_gene.png` = Fig. S3,
`figS_newdata_cytokine_loco.png` = Fig. S8). Two files in this directory carry a name a reader
would expect to be a supplementary figure and hold a **different** one:

| File | What it actually is |
|---|---|
| `figS_c4_pdl1_assay_power.png` | A single-panel chart of the observed Frangieh RNA and surface CD274 shifts. Printed Figure S4 is a three-panel figure; this is only its panel c. |
| `figS_chen_checkpoint_replication.png` | A two-panel normalization comparison. Printed Figure S5 is the revised three-panel figure embedded in the submitted supplement; `results/newdata/figS_chen_checkpoint_replication.png` preserves the earlier version. |

Figures S1, S4, S5, S6 and S8 are built in the manuscript repository
(`revision_claude/02_build/figures/`) and are not deposited here. The supplement does not claim
that its artwork lives in this directory; the record is the embedded image in the document.

## Main-figure artwork

`figure2_landscape_verdict.*` (Figure 2) and `figure_immune_blindspot.*` (Figure 3) here are the
files the paper prints. There is no `Figure2.*` in this directory; the submitted plate carries
that name only inside the manuscript package.

**`Figure1.{png,pdf,tiff}` here is NOT the paper's Figure 1.** It is a separate overview diagram,
"Immune perturbation prediction: from task to evidence", drawn by
`scripts/figure_benchmark_workflow.py`. The printed Figure 1 is the five-stage
Tasks / Splits / Methods / Metrics / Verdict schematic, and it has **no source in this
repository**: it is a design export with no script and no vector companion, so `make figures`
cannot produce it and never touches it. Read the printed figure from the submitted manuscript
package, not from this file.

## Superseded figures kept for provenance

These were drawn against the earlier **35-cell** census and disagree with the current one. They
are kept as the record of that stage and are not results of this revision. Do not read a score,
a roster or a floor-clearance count off any of them:

| File | Why it disagrees |
|---|---|
| `figure_landscape.{pdf,png}` | The 35-cell landscape. Gives scGen T1 +.06 and CPA T1 +.03 where `cross_cluster_headline.csv` now gives -0.0307 and -0.3939, plots a full CINEMA-OT row although CINEMA-OT carries no census cell, and shows two floor-clearers where the census now has eight. Superseded by `figure2_landscape_verdict.*`. |
| `figure_ranking.{pdf,png}`, `figure_perturbation.{pdf,png}`, `figure_within_family_fit.{pdf,png}` | Same vintage and the same 35-cell roster. |
| `figure_ranking_ORIGTEST.pdf` | A test render from that stage; not used anywhere. |

## Quarantined outputs: `results/_invalid_wrong_venv/`

That directory holds runs made against the wrong `ivcbench` package (the authors' historical
`benchmark/` copy rather than the released one), kept only so the record of what was executed is
complete. Nothing in the deposit reads it and no number in it is reported. Its own README says so;
`scripts/_env_guard.py` prevents a repeat.

## The legacy `Supplementary_Table_S<N>_*.csv` files in this directory

Eight files here carry a supplementary-table number in their NAME. Seven more did and have been renamed `superseded_Table_S<N>_*.csv`, because nothing reads them and a reader navigating by number would have landed on a roster that contradicts the printed table -- most sharply `S14_Tanimoto_current`, whose chemCPA slope is negative where the printed Table S14's is the only positive one. They are working artefacts
from earlier numbering, kept because figures and notes were computed from them. **The table of
record for every number is `supplementary_tables/Supplementary_Table_S<N>.csv`**, indexed by
`MANIFEST.csv`; resolve a table by that file, never by a name in this directory.

Three of the fifteen carry a number that now belongs to a different table:

| File here | What it contains | Its number in the paper |
|---|---|---|
| `Supplementary_Table_S20_panel_checks.csv` (13 rows) | immune-program panel checks | **S18**. Printed S20 is the matched donor learning curve (5 rows). |
| `Supplementary_Table_S25_matched_donor_learning_curve.csv` | the matched donor learning curve | **S20**. The supplement runs S1 to S23; there is no S25. |
| `Supplementary_Table_S11_multiplicity.csv` (58 contrasts) | the panel-wide multiplicity family | **S23** (27 contrasts). Printed S11 is the 8-row pre-specified headline family. |

Five more share a number with the printed table but are older and smaller:
`superseded_Table_S3_descriptive_fit_matrix.csv` (32 rows against 35), `superseded_Table_S7_OP3_programs.csv` (27 against 39),
`superseded_Table_S12_T3_programs.csv` (175 rows, a different schema), `superseded_Table_S14_Tanimoto_current.csv` (9 rows on the
pre-STATE T5u roster, including a CINEMA-OT row the census does not report), and
`superseded_Table_S15_training_configuration.csv` (46 rows with no Seeds column; printed S15c
has 65, and its scFoundation T2 row still describes an MLP decoder "for 512 response genes", which
the printed table corrects to all 381 genes that split scores). It was the only one of these five still
named `Supplementary_Table_...`, which read as a table of record; renamed to match the others.
`superseded_Table_S17_effect_stratification.csv` holds the pre-panel-mask quartile margins (Q1 -0.1505 where the
current run gives -0.1520).
`superseded_Table_S13_T3_by_dataset.csv` is a **third** file under that number and the easiest to mistake for a
current one: it is the 10 % leave-one-gene-out slice on a superseded roster (Chen's best
conditioned entry is scGPT at 0.386, where the current 10 % slice in `t3_by_dataset.csv` differs
and the printed 50 % table gives 0.393). Printed Table S13 is the 50 % slice in
`supplementary_tables/Supplementary_Table_S13.csv`; `t3_by_dataset.csv` is the current 10 % one.
Neither is this file.

The remaining five (`S1_dataset_inventory`, `S2_method_inventory`, `S5_op3_fine_lineage`,
`S6_marker_readout`, `S8_energy_distance`) agree with the printed table of the same number;
`S6` and `S20_panel_checks` are the full-precision sources the printed tables are re-sourced from,
and `S8_energy_distance` is the producer `supplementary_tables/Supplementary_Table_S8.csv` is
copied from each build.
