#!/usr/bin/env bash
cd "/data1/home/chlee/projects/immune virtual cell/ivcbench"
echo "RUNNING $(date -u +%FT%TZ) pid $$" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scgen_T5c.status"
env CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_cluster.py --cluster C5 --real --only scGen --seeds 0 --outdir results/native_rerun/v3_scgen_C5
code=$?
if [ $code -eq 0 ]; then echo "DONE:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scgen_T5c.status"; else echo "FAILED:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scgen_T5c.status"; fi
