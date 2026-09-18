PY ?= ./.venv/bin/python
PIP ?= ./.venv/bin/pip

# Matplotlib stamps a CreationDate into every PDF it writes, so `make figures` left the tree dirty
# with pixel-identical PNGs beside the "changed" PDFs, and a real figure change was
# indistinguishable from a clock tick. Matplotlib honours SOURCE_DATE_EPOCH; exporting it here
# makes every deposited PDF byte-reproducible. The value is arbitrary and fixed.
export SOURCE_DATE_EPOCH = 1600000000

.PHONY: setup test reproduce reproduce-eval check census summaries figures

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-core.txt
	$(PIP) install -e .

test:
# -rs prints the reason for every skip. The six that skip by default are the whole panel-mask
# suite, so a bare "98 passed, 6 skipped" reads as full coverage of the 19,264-gene mask when
# it covers none of it.
	$(PY) -m pytest -q -rs

reproduce-eval:
	$(PY) scripts/reproduce_eval.py --manifest results/_paper/census_bundle_manifest.csv -o reproduced_results.csv

check:
	$(PY) scripts/check_deposit_completeness.py
	$(PY) scripts/check_consistency.py
	$(PY) scripts/check_chemcpa_provenance.py
	$(PY) scripts/check_release_docs.py

reproduce:
	$(MAKE) reproduce-eval
	$(MAKE) check

census:
	$(PY) scripts/assemble_cross_cluster.py
# write_headline_md.py renders the two human-readable companions of the CSVs written just
# above. It was never wired here, so both froze: the CSV was regenerated with the 58-cell
# census while the .md beside it still showed the 41-row, 13-model pre-revision roster, and
# within_family_consistency.md printed a 'split' verdict under a note claiming every family
# cell agreed. Rendered from the same CSVs, in the same target, they cannot drift apart.
	$(PY) scripts/write_headline_md.py
	$(PY) scripts/census_units.py
	$(PY) scripts/assemble_fit_matrix.py
	$(PY) scripts/headline_multiplicity.py
	$(PY) scripts/headline_family.py --apply

summaries:
	$(PY) scripts/t3_effect_stratification.py
	$(PY) scripts/assemble_secondary_checks.py
	$(PY) scripts/immune_readout_audit.py
	$(PY) scripts/assemble_target_diagnostics.py
	$(PY) scripts/assemble_s8_energy_distance.py
	$(PY) scripts/assemble_donor_validation.py
	$(PY) scripts/assemble_learning_curves.py
	$(PY) scripts/assemble_budget_sensitivity.py
	$(PY) scripts/assemble_compute_evidence.py

figures:
	$(PY) scripts/figure_benchmark_workflow.py
	$(PY) scripts/make_figure2_landscape_verdict.py --deposit --out-dir results/_paper --tiff
# Figure 3 was rebuilt during revision: the primary immune-program readout is now the
# equal-weight mean-expression shift, not the top-5% rank score. Both of the old main-Figure-3
# producers here predate that change. figure_immune_readouts.py's default run includes "3", whose
# magnitude panel reads a Supplementary Table S22 column ("Predicted / observed response norm")
# that the revised table no longer has, so the bare command raises KeyError; and
# figure3_blindspot.py writes figure_immune_blindspot.{png,pdf,tiff}, which would overwrite the
# deposited Figure 3 with the superseded rank-based plate. So this target now builds only the
# supplementary panels that producer still owns, and Figure 3 stays as deposited. Its
# full-precision unit values, program membership, support, exclusion reasons and original-rank
# sensitivities are in results/_paper/immune_program_revision/.
	$(PY) scripts/figure_immune_readouts.py --figures S3 S4 S5
# README called this target "Figures 2 and 3, Figures S2, S3 and S7"; S3 and S7 were not in it.
	$(PY) scripts/figure_c3_nearest_gene.py
	$(PY) scripts/figure_newdata_cytokine_loco.py
	$(PY) scripts/figure_reliability_ceiling.py
	$(PY) scripts/assemble_learning_curves.py
