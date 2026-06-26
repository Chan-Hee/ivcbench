#!/usr/bin/env bash
# reproduce_all.sh - Level 3 of the reproduction ladder: retrain EVERY model family and reproduce all
# results, then re-score GPU-free.
#
#   scripts/reproduce_all.sh [--dry-run]
#
# It iterates scripts/train_manifest.csv (CPU `ivc` rows first - floors, FP-ridge, linear-shift - then
# the GPU families), calling the SAME per-row preflight+run logic as scripts/train_one.sh (both source
# scripts/_train_lib.sh, so there is one implementation). A model whose conda env or raw data is missing
# on this host is reported as a clean SKIP; one bad model never aborts the whole run. It finishes with a
# COVERAGE line over the 35-cell census.
#
# THIS IS THE HEAVY PATH. It needs the per-family conda environments (built from each upstream repo per
# REPRODUCE.md), the raw data (per data/README.md), and GPUs for the GPU families. It does NOT replace
# them. Level 1 (`make reproduce-eval`) reproduces the census from the deposited bundles with none of
# that: no conda, no GPU, no raw data, and is the right rung for most readers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/_train_lib.sh"

DRY_RUN=0
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) echo "usage: scripts/reproduce_all.sh [--dry-run]"; exit 0 ;;
    *) echo "unknown flag: ${arg}" >&2; echo "usage: scripts/reproduce_all.sh [--dry-run]" >&2; exit 2 ;;
  esac
done

echo "=================================================================="
echo " ivcbench Level 3: retrain everything + reproduce all results"
echo "=================================================================="
echo " HEAVY PATH. Needs the per-family conda environments (REPRODUCE.md),"
echo " the raw data (data/README.md), and GPUs for the GPU families."
echo " Level 1 'make reproduce-eval' reproduces the census from the"
echo " deposited bundles with no conda, no GPU, and no raw data."
[ "${DRY_RUN}" -eq 1 ] && echo " (--dry-run: prints the resolved plan; nothing heavy executes)"
echo ""

# ---------------------------------------------------------------------------------------------------
# PREFLIGHT REPORT - one screen: for every manifest row, will it RUN or be SKIPPED on this host, and why.
# Printed BEFORE anything heavy starts so the user sees the plan up front.
# ---------------------------------------------------------------------------------------------------
echo "------------------------------------------------------------------"
echo " PREFLIGHT REPORT (this host)"
echo "------------------------------------------------------------------"
printf " %-20s %-7s %-7s %-7s  %s\n" "MODEL" "CLUSTER" "GPU" "STATUS" "DETAIL"
N_READY=0
N_SKIP=0
while IFS="${SEP}" read -r m c gpu env py_var data_vars runner_cmd notes; do
  [ -z "${m}" ] && continue
  if row_is_ready "${m}" "${env}" "${py_var}" "${data_vars}" "${runner_cmd}"; then
    printf " %-20s %-7s %-7s %-7s  %s\n" "${m}" "${c}" "${gpu}" "READY" ""
    N_READY=$((N_READY + 1))
  else
    printf " %-20s %-7s %-7s %-7s  %s\n" "${m}" "${c}" "${gpu}" "SKIP" "${PREFLIGHT_DETAIL}"
    N_SKIP=$((N_SKIP + 1))
  fi
done < <(manifest_rows_ordered)
echo ""
echo " ${N_READY} unit(s) READY to run, ${N_SKIP} will be SKIPPED on this host."
echo " (READY units below will execute their runners; SKIP units only need their env/data set up.)"
echo ""

# ---------------------------------------------------------------------------------------------------
# EXECUTE - CPU `ivc` rows first, then GPU families. Same per-row logic as train_one.sh.
# ---------------------------------------------------------------------------------------------------
SUMMARY=()
RAN_ANY=0
while IFS= read -r record; do
  [ -z "${record}" ] && continue
  rc=0
  run_manifest_row "${record}" "${DRY_RUN}" || rc=$?
  case "${rc}" in
    0) SUMMARY+=("RAN     ${ROW_MODEL} ${ROW_CLUSTER}"); RAN_ANY=1 ;;
    10) SUMMARY+=("DRYRUN  ${ROW_MODEL} ${ROW_CLUSTER}") ;;
    20) SUMMARY+=("SKIP    ${ROW_MODEL} ${ROW_CLUSTER}  (${SKIP_REASON})") ;;
    *)  SUMMARY+=("FAILED  ${ROW_MODEL} ${ROW_CLUSTER}  (runner exit ${rc})") ;;
  esac
done < <(manifest_rows_ordered)

# ---------------------------------------------------------------------------------------------------
# RE-SCORE + COVERAGE
# ---------------------------------------------------------------------------------------------------
if [ "${DRY_RUN}" -eq 0 ] && [ "${RAN_ANY}" -eq 1 ]; then
  echo ""
  echo "------------------------------------------------------------------"
  echo " re-scoring bundles GPU-free (make reproduce-eval)"
  echo "------------------------------------------------------------------"
  reproduce_eval_and_report

  echo ""
  echo "------------------------------------------------------------------"
  echo " COVERAGE over the 35-cell census"
  echo "------------------------------------------------------------------"
  "${VENV_PY}" - <<'PY'
import csv
from pathlib import Path

# Census cells are the (cluster-family, model) pairs the bundle layer covers; count distinct ones that
# now have a reproduced row. Map split-level cluster tags back to their census family (C2/C2_LODO -> C2).
def fam(c):
    return c.split("_")[0] if c else c

repro = Path("reproduced_results.csv")
have = set()
if repro.exists():
    with repro.open(newline="") as fh:
        for r in csv.DictReader(fh):
            have.add((fam(r["cluster"]), r["model"]))

print(f"reproduced (cluster-family, model) cells with a bundle: {len(have)}")
print("note: the deposited census is 35 model-by-task cells; this run reproduces whatever the READY")
print("units above emitted. Cells whose env/data were SKIPPED above are not refrozen by this run; they")
print("retain their deposited bundles, which `make reproduce-eval` still scores. See predictions/COVERAGE.md")
print("for the authoritative 35/35 cell-by-cell account.")
PY
else
  echo ""
  echo "(no unit executed a runner -> skipping reproduce-eval + coverage)"
  echo "Every SKIP above needs that family's conda env + raw data set up (REPRODUCE.md, data/README.md)."
fi

echo ""
echo "=================================================================="
echo " Level 3 summary"
echo "=================================================================="
for s in "${SUMMARY[@]}"; do echo "  ${s}"; done
