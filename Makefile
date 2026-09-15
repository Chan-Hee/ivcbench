PY ?= ./.venv/bin/python
PIP ?= ./.venv/bin/pip

.PHONY: setup test reproduce reproduce-eval check census summaries figures

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-core.txt
	$(PIP) install -e .

test:
	$(PY) -m pytest -q

reproduce-eval:
	$(PY) scripts/reproduce_eval.py --manifest results/_paper/census_bundle_manifest.csv -o reproduced_results.csv

check:
	$(PY) scripts/check_consistency.py
	$(PY) scripts/check_chemcpa_provenance.py

reproduce:
	$(MAKE) reproduce-eval
	$(MAKE) check

census:
	$(PY) scripts/assemble_cross_cluster.py
	$(PY) scripts/census_units.py
	$(PY) scripts/assemble_fit_matrix.py
	$(PY) scripts/headline_multiplicity.py

summaries:
	$(PY) scripts/t3_effect_stratification.py
	$(PY) scripts/assemble_secondary_checks.py
	$(PY) scripts/immune_readout_audit.py
	$(PY) scripts/assemble_target_diagnostics.py
	$(PY) scripts/assemble_donor_validation.py
	$(PY) scripts/assemble_learning_curves.py
	$(PY) scripts/assemble_budget_sensitivity.py
	$(PY) scripts/assemble_compute_evidence.py

figures:
	$(PY) scripts/figure_benchmark_workflow.py
	$(PY) scripts/make_figure2_landscape_verdict.py --deposit --out-dir results/_paper --tiff
	$(PY) scripts/figure_immune_readouts.py
	$(PY) scripts/figure_reliability_ceiling.py
	$(PY) scripts/assemble_learning_curves.py
