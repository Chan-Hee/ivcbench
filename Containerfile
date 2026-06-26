# GPU-free reproduction image for ivcbench.
#
# This image carries the deposited prediction bundles and the analysis environment, so the
# 35-cell census, the headline numbers, and the figures can be recomputed with one command —
# no conda, no GPU, and no raw single-cell data. Retraining the models is a separate, heavier
# path (each model family has its own conda environment); see REPRODUCE.md.
#
#   podman build -t ivcbench .
#   podman run --rm ivcbench                 # re-score the bundles, rebuild the census, verify it reproduces the paper
#   podman run --rm -v "$PWD/out:/ivcbench/out" ivcbench \
#       .venv/bin/python scripts/reproduce_eval.py 'predictions/**/*.npz' -o out/reproduced_results.csv
#
FROM python:3.13-slim

WORKDIR /ivcbench

# make + a compiler for the few source builds; everything else ships as manylinux wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends make git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Resolve the analysis dependencies first so the layer caches across code edits. The Makefile
# expects the interpreter at ./.venv/bin/python, so the venv path matches the documented commands.
COPY requirements.txt pyproject.toml ./
RUN python -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . .
RUN .venv/bin/pip install --no-cache-dir -e .

# Default run re-scores the bundles, rebuilds the 35-cell census, and asserts it reproduces the
# committed paper numbers (floor verdicts intact). `make reproduce` also writes reproduced_results.csv.
CMD ["make", "reproduce"]
