#!/usr/bin/env bash
cd "/data1/home/chlee/projects/immune virtual cell/ivcbench"
echo "RUNNING $(date -u +%FT%TZ) pid $$" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scf_T3_u3.status"
# Model-asset env, asserted. A missing checkpoint path must fail the JOB here, loudly, rather
# than reach the runner and come back as ran=False -- which reads like a model limitation.
. "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/env.sh" || { echo "FAILED:90 $(date -u +%FT%TZ) runs/env.sh assertion" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scf_T3_u3.status"; exit 90; }
mkdir -p "$IVCBENCH_PRED_DUMP"; date +%s > "$IVCBENCH_PRED_DUMP/.scf_T3_u3.t0"
.venv/bin/python scripts/run_obligation.py --model scFoundation --task T3 --chunk 3 5 --gpu 2 --out results/native_rerun/v7_scfoundation_T3_u3.csv
code=$?
if [ $code -eq 0 ]; then
  # rc=0 is not evidence of a filled cell: jobs have exited 0 with ran=False everywhere, and
  # with no bundle deposited at all. Check the output before calling this DONE.
  # the job's own result CSV, if its command names one with --out
  OUTCSV=$(printf '%s\n' ".venv/bin/python scripts/run_obligation.py --model scFoundation --task T3 --chunk 3 5 --gpu 2 --out results/native_rerun/v7_scfoundation_T3_u3.csv" | grep -oP '(?<=--out )\S+' | head -1)
  if "/data1/home/chlee/projects/immune virtual cell/ivcbench/.venv/bin/python" "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/postflight.py" "scf_T3_u3" "$IVCBENCH_PRED_DUMP" $OUTCSV; then
    echo "DONE:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scf_T3_u3.status"
  else
    echo "UNUSABLE:4 $(date -u +%FT%TZ) postflight" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scf_T3_u3.status"
  fi
else
  echo "FAILED:$code $(date -u +%FT%TZ)" > "/data1/home/chlee/projects/immune virtual cell/ivcbench/runs/scf_T3_u3.status"
fi
