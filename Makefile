PY := ./.venv/bin/python
PIP := ./.venv/bin/pip

.PHONY: setup test pilot reproduce-eval census check reproduce train train-all reproduce-all train-image data data.c5 cluster integrated-figure clean

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

reproduce-eval:              ## predictions -> per-bundle metrics, GPU-free: re-score every deposited prediction bundle
	$(PY) scripts/reproduce_eval.py 'predictions/**/*.npz' 'predictions/*.npz' -o reproduced_results.csv

check:                       ## GPU-free gate: rebuild the census from the bundles and assert the committed paper numbers match
	$(PY) scripts/check_consistency.py

reproduce:                   ## full GPU-free reader path: re-score every bundle, then verify the committed census reproduces it
	$(MAKE) reproduce-eval
	$(MAKE) check

census:                      ## MAINTAINER ONLY: re-derive every bundle-sourced artifact (results_raw sync, headline S2b, S3, S4, C2 paired S5/S7, multiplicity S10)
	$(PY) scripts/sync_results_raw.py
	$(PY) scripts/assemble_cross_cluster.py
	$(PY) scripts/assemble_fit_matrix.py
	$(PY) scripts/c2_donor_paired.py
	$(PY) scripts/headline_multiplicity.py

train:                       ## retrain ONE model + reproduce it (heavy: needs that family's env+data+GPU), e.g. make train MODEL=cellot
	@test -n "$(MODEL)" || (echo "usage: make train MODEL=cellot"; exit 1)
	bash scripts/train_one.sh $(MODEL)

train-all:                   ## retrain everything + reproduce all results (heavy: per-family envs+data+GPUs); see scripts/train_manifest.csv
	bash scripts/reproduce_all.sh

reproduce-all:               ## FULL paper reproduction (GPU): download data -> train every model -> re-score -> final figure
	$(MAKE) data
	$(MAKE) train-all
	$(MAKE) integrated-figure

train-image:                 ## build the all-environments retraining image (large; conda-packs the heavy envs, see Containerfile.train)
	bash scripts/build_train_image.sh

data:                        ## download ALL public census raw data in one command (use --list via the script to preview)
	bash scripts/download_all.sh

data.c5:                     ## download OP3 (GSE279945), public, no DAC
	bash scripts/download_op3.sh

cluster:                     ## one paper cycle (REAL data): results + manifest + figure + draft, e.g. make cluster C=C1
	@test -n "$(C)" || (echo "usage: make cluster C=C1"; exit 1)
	$(PY) scripts/run_cluster.py --cluster $(C) --real   # --real: C1/C5 default to synthetic without it
	@echo "note: C3/C4 are always real (multi-dataset); --real is a no-op there"

integrated-figure:           ## regenerate the cross-cluster integrated benchmark figure
	$(PY) scripts/integrated_figure.py

clean:
	rm -rf .pytest_cache **/__pycache__ src/*.egg-info
