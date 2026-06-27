#!/usr/bin/env bash
# reproduce_all_bundles.sh - L2 turn-key: retrain every census model FROM SCRATCH and re-dump its
# prediction bundle into IVCBENCH_PRED_DUMP, so the bundle-sourced census (scripts/assemble_cross_cluster.py)
# reproduces every paper number from a fresh GitHub checkout. This is the GPU "Tier-2" backbone behind
# `make reproduce-all`. It orchestrates the SAME proven per-cell runners that produced the deposit (not the
# unvalidated manifest path), each with IVCBENCH_PRED_DUMP[_MEANS] set so a re-run materialises the
# model-output layer the GPU-free re-scorer reads.
#
#   bash scripts/reproduce_all_bundles.sh [--test K] [--dump DIR] [--gpus 0,1,2,3]
#
#     --test K   run only the first K units per cluster (de-risk subset; default 0 = all)
#     --dump DIR where bundles land (default: predictions/ ; use a scratch dir + diff to keep the
#                deposit frozen, or run the whole thing in a throwaway clone)
#     --gpus     comma GPU ids for the sharded bespoke runners (default 0,1,2,3)
#
# Deterministic models reproduce bit-identically; stochastic ones (CPA, ...) within run-to-run spread
# (~<=0.03 per unit). Models whose weights/code are not redistributable (scFoundation; optionally
# scGPT/AttentionPert when their external src/ckpt is absent) are SKIPPED with a notice and reproduced
# from their deposited bundle (the GPU-free path) instead. See REPRODUCE.md for the env + data setup and
# RUNNER_FIDELITY_AUDIT.md for what each producer dumps.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

TEST=0; DUMP="${IVCBENCH_PRED_DUMP:-$ROOT/predictions}"; GPUS="0,1,2,3"
while [ $# -gt 0 ]; do
  case "$1" in
    --test) TEST="$2"; shift 2 ;;
    --dump) DUMP="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$DUMP"
export IVCBENCH_PRED_DUMP="$DUMP" IVCBENCH_PRED_DUMP_MEANS=1
IFS=',' read -ra G <<<"$GPUS"; NG=${#G[@]}
TARG=""; [ "$TEST" -gt 0 ] && TARG="--test $TEST"

CORE="$ROOT/.venv/bin/python"
CELLOT="${IVCBENCH_CELLOT_PY:-$HOME/miniconda3/envs/cellot/bin/python}"
CPAENV="${IVCBENCH_IVC_CPA_PYTHON:-$HOME/miniconda3/envs/ivc-cpa/bin/python}"
STENV="${IVCBENCH_IVC_STATE_PYTHON:-$HOME/miniconda3/envs/ivc-state/bin/python}"
SPENV="${IVCBENCH_IVC_SCPRAM_PYTHON:-$HOME/miniconda3/envs/ivc-scpram/bin/python}"
log(){ echo "[$(date '+%F %T')] $*"; }

# The deposit organises bundles into per-cluster subdirs (predictions/C1/, ...) but dump_bundle writes
# FLAT into $DUMP. Flatten any existing subdir bundles into $DUMP first, so each regenerated bundle
# OVERWRITES its deposited counterpart in place instead of leaving a flat-vs-subdir duplicate that the
# recursive census glob would double-count. Names are unique on (cluster,model,split) so this never
# clobbers two distinct bundles; deposit-only models (e.g. scFoundation) keep their bundle for re-score.
if find "$DUMP" -mindepth 2 -name '*.npz' -print -quit | grep -q .; then
  log "flattening per-cluster subdirs in $DUMP for in-place regeneration"
  find "$DUMP" -mindepth 2 -name '*.npz' -exec mv -n {} "$DUMP/" \;
  find "$DUMP" -mindepth 1 -type d -empty -delete
fi

log "L2 reproduce-all-bundles | dump=$DUMP | test=$TEST | gpus=$GPUS"
log "IVCBENCH_PRED_DUMP=$IVCBENCH_PRED_DUMP IVCBENCH_PRED_DUMP_MEANS=1"

# ---- C1 cytokine/Kang LOCT: floors + scGen + CPA (run_job dumps; self-contained envs) ---------------
log "[C1] run_cluster --cluster C1"
$CORE scripts/run_cluster.py --cluster C1 --real --seeds 0 $TARG

# ---- C2 donor/Soskic LODO ---------------------------------------------------------------------------
# scheme B (cluster=C2_LODO): floors + scGen via run_job, with --only so the bespoke OT/latent models
# below are not duplicated under the C2_LODO key.
log "[C2] scheme-B floors+scGen via run_cluster --only"
$CORE scripts/run_cluster.py --cluster C2 --real --seeds 0 $TARG \
  --only "ctrl-pred,cell-mean,donor-shift,linear-PCA,scGen"
# scheme A (cluster=C2): CellOT/CPA/STATE/scPRAM bespoke, sharded across GPUs (106 donors). --test caps.
SK=""; [ "$TEST" -gt 0 ] && SK="--test $TEST"
log "[C2] bespoke CellOT/CPA/STATE/scPRAM (sharded $NG-way) + PertAdapt"
shard(){ # $1=env $2=script $3..=extra ; shards --chunk i NG across the GPU list
  local env="$1" scr="$2"; shift 2
  for i in "${!G[@]}"; do
    CUDA_VISIBLE_DEVICES="${G[$i]}" "$env" "scripts/$scr" --chunk "$i" "$NG" --seeds 0 --cap 300 $SK "$@" \
      >"$DUMP/.log_${scr%.py}_g$i.txt" 2>&1 &
  done; wait
}
shard "$CELLOT" cellot_soskic.py --ae-iters 12000 --cellot-iters 8000
shard "$CPAENV" cpa_soskic.py --cpa-epochs 60
shard "$STENV"  state_soskic.py --steps 400
shard "$SPENV"  scpram_soskic.py --epochs 100
# PertAdapt needs the scFoundation ckpt + GO mask; skip cleanly if absent.
if [ -n "${IVCBENCH_SCFOUNDATION_CKPT:-}" ] && [ -f "${IVCBENCH_SCFOUNDATION_CKPT:-/nonexistent}" ]; then
  log "[C2] PertAdapt"; CUDA_VISIBLE_DEVICES="${G[0]}" "${IVCBENCH_SCFOUNDATION_PYTHON:-$HOME/miniconda3/envs/scfoundation/bin/python}" \
    scripts/pertadapt_soskic.py --chunk 0 1 --seeds 0 --cap 300 $SK >"$DUMP/.log_pertadapt.txt" 2>&1 || log "  PertAdapt failed (see log)"
else log "[C2] PertAdapt SKIPPED (scFoundation ckpt absent) -> use deposited bundle"; fi

# ---- C3 gene/CRISPR LO-gene: run_cluster runs floors + GEARS/CPA/STATE/CINEMA-OT/scGen (self-contained)
# and, when their external src/ckpt is present, scGPT/scFoundation/AttentionPert/PertAdapt. Absent ones
# are skipped by the subprocess adapter and reproduced from their deposited bundle.
log "[C3] run_cluster --cluster C3 (self-contained + any present foundation/graph models)"
$CORE scripts/run_cluster.py --cluster C3 --real --seeds 0 $TARG

# ---- C4 complex/Frangieh modality-LO-KO: floors via run_cluster + bespoke OT/latent/graph/state/cond --
log "[C4] run_cluster floors + bespoke frangieh"
$CORE scripts/run_cluster.py --cluster C4 --real --seeds 0 $TARG
CUDA_VISIBLE_DEVICES="${G[0]}" "$CELLOT" scripts/cellot_frangieh.py --chunk 0 1 --gpu 0 >"$DUMP/.log_cellot_frangieh.txt" 2>&1 || log "  cellot_frangieh failed"
CUDA_VISIBLE_DEVICES="${G[0]}" "$CPAENV"  scripts/cpa_frangieh.py    --chunk 0 1 --gpu 0 >"$DUMP/.log_cpa_frangieh.txt" 2>&1 || log "  cpa_frangieh failed"
CUDA_VISIBLE_DEVICES="${G[0]}" "$SPENV"   scripts/scpram_frangieh.py --chunk 0 1 --gpu 0 >"$DUMP/.log_scpram_frangieh.txt" 2>&1 || log "  scpram_frangieh failed"
CUDA_VISIBLE_DEVICES="${G[0]}" "$CORE"    scripts/state_frangieh.py  --chunk 0 1 --gpu 0 --steps 400 >"$DUMP/.log_state_frangieh.txt" 2>&1 || log "  state_frangieh failed"
CUDA_VISIBLE_DEVICES="${G[0]}" "$CORE"    scripts/graph_frangieh.py  --full >"$DUMP/.log_graph_frangieh.txt" 2>&1 || log "  graph_frangieh failed (GEARS ok; AttentionPert needs external src)"
"$CORE" scripts/run_c4_conditioned.py >"$DUMP/.log_c4_cond.txt" 2>&1 || log "  run_c4_conditioned failed"

# ---- C5 small-mol/OP3: floors + FP-ridge + CPA/scGen/STATE/CINEMA-OT via run_cluster, + chemCPA --------
log "[C5] run_cluster --cluster C5 (FP-ridge + framework)"
$CORE scripts/run_cluster.py --cluster C5 --real --seeds 0 $TARG
log "[C5] chemCPA native -> evaluate (dumps bundle)"
CUDA_VISIBLE_DEVICES="${G[0]}" "$CPAENV" scripts/chemcpa_native_op3.py 0 outputs/additional_models >"$DUMP/.log_chemcpa_native.txt" 2>&1 || log "  chemcpa_native failed"
"$CORE" scripts/chemcpa_evaluate.py >"$DUMP/.log_chemcpa_eval.txt" 2>&1 || log "  chemcpa_evaluate failed"

# ---- Assemble the bundle-sourced census + consistency tables ----------------------------------------
log "[assemble] bundle-sourced 35-cell census"
"$CORE" scripts/assemble_cross_cluster.py
log "DONE. Census -> results/_paper/cross_cluster_headline.csv (re-scored from $DUMP)."
log "Compare a scratch --dump against the frozen deposit with: python scripts/compare_bundles.py $DUMP"
