PY := ./.venv/bin/python
PIP := ./.venv/bin/pip

.PHONY: setup test pilot data.c5 cluster integrated-figure clean

setup:                       ## GPU-free smoke-test core (subset of requirements.txt); full figure/analysis rebuild uses `pip install -r requirements.txt`
	# `make setup` installs only the GPU-free smoke-test core deps below; the full
	# figure/analysis rebuild (real-data loaders, rdkit, AUCell, etc.) uses requirements.txt.
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q numpy scipy scikit-learn pandas pyyaml pytest matplotlib python-docx
	$(PIP) install -q -e .

test:                        ## run leak-audit + smoke tests (GPU-free)
	$(PY) -m pytest -q

pilot:                       ## C5 "1패스" on synthetic OP3-shaped data
	$(PY) scripts/run_c5_pilot.py

data.c5:                     ## download OP3 (GSE279945) — public, no DAC
	bash scripts/download_op3.sh

cluster:                     ## one paper cycle (REAL data): results + manifest + figure + draft, e.g. make cluster C=C1
	@test -n "$(C)" || (echo "usage: make cluster C=C1"; exit 1)
	$(PY) scripts/run_cluster.py --cluster $(C) --real   # --real: C1/C5 default to synthetic without it
	@echo "note: C3/C4 are always real (multi-dataset); --real is a no-op there"

integrated-figure:           ## regenerate the cross-cluster integrated benchmark figure
	$(PY) scripts/integrated_figure.py

clean:
	rm -rf .pytest_cache **/__pycache__ src/*.egg-info
