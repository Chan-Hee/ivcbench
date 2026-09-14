#!/usr/bin/env bash
cd "/data1/home/chlee/projects/immune virtual cell/ivcbench"
echo "RUNNING $(date -u +%FT%TZ) pid $$" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/state_C5.status"
.venv/bin/python scripts/run_cluster.py --cluster C5 --real --only STATE --seeds 0 --gpus 2 --outdir results/native_rerun/v2_state_C5
code=$?
if [ $code -eq 0 ]; then echo "DONE:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/state_C5.status"; else echo "FAILED:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/state_C5.status"; fi
