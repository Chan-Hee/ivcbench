PY ?= ./.venv/bin/python
PIP ?= ./.venv/bin/pip

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
	$(PY) scripts/figure_immune_readouts.py
# figure_immune_readouts writes figure_immune_blindspot at 2157x2475, overwriting the
# deposited Figure 3 (1644x1608). figure3_blindspot.py reproduces the deposited file
# byte-identically and was missing from this target, so `make figures` replaced the
# paper's figure with a different one.
	$(PY) scripts/figure3_blindspot.py
	$(PY) scripts/figure_reliability_ceiling.py
	$(PY) scripts/assemble_learning_curves.py
