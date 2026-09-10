#!/usr/bin/env bash
# Launch the scGPT donor-count LEARNING CURVE on Soskic C2 as THREE detached GPU jobs (GPUs 0,1,2).
#
# Answers Reviewer 2 comment 3: the deposited donor curve is CellOT's, and the reviewer reads a model
# that DECLINES with more donors as under-trained. This runs the same curve for a PRETRAINED model
# FINE-TUNED END-TO-END (scGPT from scGPT_human, ScGPTC1 adapter -> scgpt_c1_runner.py) on
# bit-identical splits: the same fixed 10 eval donors, the same 96-donor train pool, the same seeded
# k-subsets, the same train-fold response-gene panel and the same CellMean/DonorShift floors.
#
# DESIGN, and what it costs relative to the CellOT curve
#   grid   [8,16,32,64,96]  UNCHANGED
#   n_eval 10               UNCHANGED (all ten CellOT eval donors)
#   seeds  [0] not [0,1]    THE ONLY REDUCTION -- the cheapest one in comparability terms, because
#                           (a) model_runners/scgpt_c1_runner.py calls set_seed(0) unconditionally, so
#                               `seed` never perturbs scGPT's initialisation or batch order: it only
#                               re-draws WHICH k donors are sampled, and at k=96 the draw is the whole
#                               96-donor pool for every seed, making a second seed a bit-identical
#                               rerun of the top grid point;
#                           (b) in the deposited CellOT curve seed 0 alone reproduces the 8->96 trend
#                               to 0.001 (-0.0711 vs -0.0702 over both seeds), while the across-eval-
#                               donor SD (~0.08) that IS retained dominates the seed spread (~0.02).
#                           Two seeds would cost 10.9 GPU-h; one costs 5.5 GPU-h.
#
# COST (measured on idle L40s, epochs=10 / seqlen=2048 / max 8000 stimulated training cells):
#   k=8   269 s per (grid,seed,eval-donor) unit  (mean of 10; 4,800 stim cells -> 6,000 steps)
#   k>=16 425 s per unit (the 8,000-stim-cell cap binds from k=14 up, so 16/32/64/96 all cost the same)
#   peak GPU memory 3,675 MiB; one-time CellSet load ~145 s per process.
#
# Sharding is over EVAL DONORS, so every shard covers the whole grid. Each shard is pre-seeded with
# whatever is already in the canonical CSV and run with --skip-existing, so completed units are never
# recomputed. Run `$0 merge` when the three jobs finish.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${IVCBENCH_PY:-$ROOT/.venv/bin/python}"
SCRIPT="$ROOT/scripts/scgpt_donor_learning_curve.py"
OUTDIR="$ROOT/results/newdata"
LOGDIR="$ROOT/outputs/scgpt_curve"
CANON="$OUTDIR/scgpt_donor_learning_curve.csv"
mkdir -p "$OUTDIR" "$LOGDIR"

export IVCBENCH_SCGPT_MODEL_DIR="${IVCBENCH_SCGPT_MODEL_DIR:-/data1/home/chlee/projects/single_cell_fm/models/scGPT_human}"

GRID="8 16 32 64 96"
NEVAL=10
SEEDS=0
NSHARD=3
GPUS=(0 1 2)          # GPU 3 is in use by another job

if [ "${1:-launch}" = "merge" ]; then
  "$PY" - "$OUTDIR" <<'PYEOF'
import sys, glob, json
import pandas as pd
out = sys.argv[1].rstrip('/') + '/'
canon = out + 'scgpt_donor_learning_curve.csv'
key = ['n_train_donors', 'seed', 'eval_donor', 'metric']
frames = [pd.read_csv(f) for f in sorted(glob.glob(out + 'scgpt_donor_learning_curve_shard*.csv'))]
if glob.glob(canon):
    frames.insert(0, pd.read_csv(canon))
m = pd.concat(frames, ignore_index=True).drop_duplicates(subset=key, keep='first')
m.sort_values(key).reset_index(drop=True).to_csv(canon, index=False)
tim, seen = [], set()
for f in [out + 'scgpt_donor_learning_curve_timing.json'] + sorted(
        glob.glob(out + 'scgpt_donor_learning_curve_shard*_timing.json')):
    try:
        rows = json.load(open(f))
    except Exception:
        continue
    for r in rows:
        k = (r['n_train_donors'], r['seed'], r['eval_donor'])
        if k not in seen:
            seen.add(k); tim.append(r)
json.dump(tim, open(out + 'scgpt_donor_learning_curve_timing.json', 'w'), indent=2)
pe = m[(m.metric == 'pearson_delta') & (m.delta_vs_primary.astype(str) != '')].copy()
for c in ('scgpt_score', 'baseline_score', 'delta_vs_primary'):
    pe[c] = pe[c].astype(float)
print(f'MERGED {canon}  ({len(m)} rows, {len(tim)} units)')
print(pe.groupby('n_train_donors').agg(scgpt=('scgpt_score', 'mean'),
                                       floor=('baseline_score', 'mean'),
                                       delta=('delta_vs_primary', 'mean'),
                                       n=('eval_donor', 'size')).round(4).to_string())
PYEOF
  exit 0
fi

for i in $(seq 0 $((NSHARD-1))); do
  gpu="${GPUS[$i]}"
  out="$OUTDIR/scgpt_donor_learning_curve_shard${i}.csv"
  timing="$OUTDIR/scgpt_donor_learning_curve_shard${i}_timing.json"
  log="$LOGDIR/scgpt_donor_learning_curve_shard${i}_gpu${gpu}.log"
  # pre-seed the shard with everything already scored so --skip-existing does not recompute it
  [ -f "$CANON" ] && [ ! -f "$out" ] && cp "$CANON" "$out"
  echo "[launch] shard $i/$NSHARD on GPU $gpu -> $log"
  setsid nohup "$PY" "$SCRIPT" \
    --grid $GRID --n-eval "$NEVAL" --seeds $SEEDS --eval-chunk "$i" "$NSHARD" --gpu "$gpu" \
    --out "$out" --timing-out "$timing" --skip-existing \
    >> "$log" 2>&1 < /dev/null &
  echo "    pid=$! (detached)"
done
echo "[launch] $NSHARD detached shards started; run '$0 merge' when they finish."
