#!/usr/bin/env bash
cd "/data1/home/chlee/projects/immune virtual cell/ivcbench"
echo "RUNNING $(date -u +%FT%TZ) pid $$" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scgpt_12x.status"
env IVCBENCH_SCGPT_MODEL_DIR=/data1/home/chlee/projects/single_cell_fm/models/scGPT_human .venv/bin/python scripts/scgpt_donor_learning_curve.py --grid 96 --seeds 0 --epochs 30 --max-cells 32000 --gpu 1 --out results/newdata/scgpt_budget_12x.csv --timing-out results/newdata/scgpt_budget_12x_timing.json
code=$?
if [ $code -eq 0 ]; then echo "DONE:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scgpt_12x.status"; else echo "FAILED:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scgpt_12x.status"; fi
