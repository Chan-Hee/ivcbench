#!/usr/bin/env bash
cd "/data1/home/chlee/projects/immune virtual cell/ivcbench"
echo "RUNNING $(date -u +%FT%TZ) pid $$" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scgpt_epcurve_s0.status"
env IVCBENCH_SCGPT_MODEL_DIR=/data1/home/chlee/projects/single_cell_fm/models/scGPT_human IVCBENCH_SCGPT_TRACE=results/newdata/scgpt_epcurve_trace_s0.jsonl .venv/bin/python scripts/scgpt_donor_learning_curve.py --grid 96 --n-eval 4 --eval-chunk 0 2 --seeds 0 --epochs 20 --eval-epochs 1,2,4,6,8,10,13,16,20 --max-cells 8000 --gpu 0 --out results/newdata/scgpt_epcurve_s0.csv --timing-out results/newdata/scgpt_epcurve_s0_timing.json
code=$?
if [ $code -eq 0 ]; then echo "DONE:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scgpt_epcurve_s0.status"; else echo "FAILED:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scgpt_epcurve_s0.status"; fi
