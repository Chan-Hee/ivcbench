#!/usr/bin/env bash
cd "/data1/home/chlee/projects/immune virtual cell/ivcbench"
echo "RUNNING $(date -u +%FT%TZ) pid $$" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/v2_scpram_T1.status"
# Model-asset env, asserted. A missing checkpoint path must fail the JOB here, loudly, rather
# than reach the runner and come back as ran=False -- which reads like a model limitation.
. "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/env.sh" || { echo "FAILED:90 $(date -u +%FT%TZ) runs/env.sh assertion" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/v2_scpram_T1.status"; exit 90; }
.venv/bin/python scripts/run_obligation.py --model scPRAM --task T1 --gpu 1 --out results/native_rerun/v6_scpram_T1.csv
code=$?
if [ $code -eq 0 ]; then echo "DONE:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/v2_scpram_T1.status"; else echo "FAILED:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/v2_scpram_T1.status"; fi
