#!/usr/bin/env bash
# train_one.sh - a provenance path: retrain ONE model family and refreeze its
# prediction bundles so `make reproduce-eval` can re-score it GPU-free.
#
#   scripts/train_one.sh <model> [--dry-run]
#
# It reads scripts/train_manifest.csv (the single source of truth: one row per runnable
# (model, cluster) unit), preflights each row's conda interpreter and raw-data variables, runs the
# exact runner command from the manifest with IVCBENCH_PRED_DUMP[_MEANS] set, then re-scores.
#
# This is the HEAVY path. It orchestrates the per-family runners; it does NOT remove the need for the
# per-family conda environments (built from each upstream repo per REPRODUCE.md), the raw data
# (per data/README.md), or a GPU. A missing env or dataset is reported as a clean SKIP, never a crash.
# The GPU-free path (`make reproduce-eval`) needs none of those.
set -euo pipefail

# Resolve repo root from this script's own location, then cd there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/_train_lib.sh"

usage() { echo "usage: scripts/train_one.sh <model> [--dry-run]" >&2; }

MODEL=""
DRY_RUN=0
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown flag: ${arg}" >&2; usage; exit 2 ;;
    *) if [ -z "${MODEL}" ]; then MODEL="${arg}"; else echo "unexpected argument: ${arg}" >&2; usage; exit 2; fi ;;
  esac
done

if [ -z "${MODEL}" ]; then
  usage
  echo "" >&2
  echo "valid model names (from ${MANIFEST}):" >&2
  manifest_models | sed 's/^/  /' >&2
  exit 2
fi

# Collect matching rows (case-insensitive on the model column).
mapfile -t ROWS < <(manifest_rows_for_model "${MODEL}")
if [ "${#ROWS[@]}" -eq 0 ]; then
  echo "no manifest row matches model '${MODEL}'." >&2
  echo "" >&2
  echo "valid model names (from ${MANIFEST}):" >&2
  manifest_models | sed 's/^/  /' >&2
  exit 2
fi

echo "=================================================================="
echo " ivcbench retrain (provenance): ${MODEL}   (${#ROWS[@]} unit(s))"
[ "${DRY_RUN}" -eq 1 ] && echo " (--dry-run: resolved commands are printed, nothing heavy executes)"
echo "=================================================================="

SUMMARY=()
RAN_ANY=0
for row in "${ROWS[@]}"; do
  rc=0
  run_manifest_row "${row}" "${DRY_RUN}" || rc=$?
  case "${rc}" in
    0) SUMMARY+=("RAN     ${ROW_MODEL} ${ROW_CLUSTER}"); RAN_ANY=1 ;;
    10) SUMMARY+=("DRYRUN  ${ROW_MODEL} ${ROW_CLUSTER}") ;;
    20) SUMMARY+=("SKIP    ${ROW_MODEL} ${ROW_CLUSTER}  (${SKIP_REASON})") ;;
    *)  SUMMARY+=("FAILED  ${ROW_MODEL} ${ROW_CLUSTER}  (runner exit ${rc})") ;;
  esac
done

# Re-score the (possibly newly refrozen) bundles, unless this was a dry run.
if [ "${DRY_RUN}" -eq 0 ] && [ "${RAN_ANY}" -eq 1 ]; then
  echo ""
  echo "------------------------------------------------------------------"
  echo " re-scoring bundles GPU-free (make reproduce-eval)"
  echo "------------------------------------------------------------------"
  reproduce_eval_and_report
else
  echo ""
  echo "(no unit executed a runner -> skipping reproduce-eval)"
fi

echo ""
echo "=================================================================="
echo " summary for ${MODEL}"
echo "=================================================================="
for s in "${SUMMARY[@]}"; do echo "  ${s}"; done
