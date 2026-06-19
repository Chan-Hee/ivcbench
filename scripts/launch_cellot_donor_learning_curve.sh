#!/usr/bin/env bash
# Launch the CellOT donor-count LEARNING CURVE on Soskic C2 as TWO detached GPU jobs (GPUs 0 and 1).
# Each GPU runs one seed of the full grid {8,16,32,64,96} training-donors over a FIXED 10-donor eval set.
# setsid+nohup => survives disconnection. --skip-existing => resumable. Separate CSV/log per seed avoids
# write contention; a downstream merge concatenates them.
set -euo pipefail
# Repo root is resolved from this script's location; no path edits needed.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# The CellOT conda env's python (conflicting torch/python pins => its own env).
# Override with the env var, e.g. IVCBENCH_CELLOT_PY=/path/to/envs/cellot/bin/python
PY="${IVCBENCH_CELLOT_PY:-python}"
SCRIPT="$ROOT/scripts/cellot_donor_learning_curve.py"
OUTDIR="$ROOT/results/newdata"
mkdir -p "$OUTDIR"

GRID="8 16 32 64 96"
NEVAL=10
AE_ITERS=12000
CELLOT_ITERS=8000

launch_seed () {
  local gpu="$1" seed="$2"
  local out="$OUTDIR/cellot_donor_learning_curve_seed${seed}.csv"
  local timing="$OUTDIR/cellot_donor_learning_curve_seed${seed}_timing.json"
  local log="$OUTDIR/cellot_donor_learning_curve_seed${seed}.log"
  echo "[launch] GPU $gpu seed $seed -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" setsid nohup "$PY" "$SCRIPT" \
    --grid $GRID --n-eval "$NEVAL" --seeds "$seed" \
    --ae-iters "$AE_ITERS" --cellot-iters "$CELLOT_ITERS" \
    --out "$out" --timing-out "$timing" --skip-existing \
    >> "$log" 2>&1 < /dev/null &
  echo "    pid=$! (detached; CUDA_VISIBLE_DEVICES=$gpu)"
}

launch_seed 0 0
launch_seed 1 1
echo "[launch] both detached jobs started."
