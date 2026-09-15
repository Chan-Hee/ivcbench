#!/usr/bin/env bash
cd "/data1/home/chlee/projects/immune virtual cell/ivcbench"
echo "RUNNING $(date -u +%FT%TZ) pid $$" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scgpt_smoke.status"
env IVCBENCH_SCGPT_MODEL_DIR=/data1/home/chlee/projects/single_cell_fm/models/scGPT_human .venv/bin/python scripts/scgpt_donor_learning_curve.py --grid 8 --n-eval 10 --eval-chunk 0 10 --seeds 0 --epochs 3 --eval-epochs 1,2,3 --max-cells 400 --cap 60 --gpu 3 --out /tmp/claude-1007/-data1-home-chlee-projects-immune-virtual-cell/54314527-3353-48ae-95ef-60365155ee15/scratchpad/scgpt_smoke.csv --timing-out /tmp/claude-1007/-data1-home-chlee-projects-immune-virtual-cell/54314527-3353-48ae-95ef-60365155ee15/scratchpad/scgpt_smoke_timing.json
code=$?
if [ $code -eq 0 ]; then echo "DONE:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scgpt_smoke.status"; else echo "FAILED:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scgpt_smoke.status"; fi
