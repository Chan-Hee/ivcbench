#!/usr/bin/env bash
cd "/data1/home/chlee/projects/immune virtual cell/ivcbench"
echo "RUNNING $(date -u +%FT%TZ) pid $$" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scpram_T5c.status"
.venv/bin/python scripts/run_obligation.py --model scPRAM --task T5c --gpu 0 --out results/native_rerun/v3_scpram_T5c.json
code=$?
if [ $code -eq 0 ]; then echo "DONE:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scpram_T5c.status"; else echo "FAILED:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scpram_T5c.status"; fi
