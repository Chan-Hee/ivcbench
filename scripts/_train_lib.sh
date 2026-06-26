#!/usr/bin/env bash
# _train_lib.sh - ONE implementation of manifest parsing + per-row preflight + per-row run, sourced by
# both scripts/train_one.sh (Level 2) and scripts/reproduce_all.sh (Level 3). Assumes the caller has
# already cd'd to the repo root and that ${REPO_ROOT} is set.
#
# Per-row contract (run_manifest_row): exit codes are returned via `return`, not `exit`, so one bad row
# never aborts a multi-row driver. After each call the caller can read ROW_MODEL / ROW_CLUSTER and, on a
# skip, SKIP_REASON. Return codes: 0 ran ok, 10 dry-run only, 20 skipped (preflight), >0 runner failure.

MANIFEST="${REPO_ROOT}/scripts/train_manifest.csv"
VENV_PY="${REPO_ROOT}/.venv/bin/python"
# Conda root used by ivcbench.baselines.heavy.env_python to locate envs/<env>/bin/python.
CONDA_ROOT_DIR="${CONDA_ROOT:-${HOME}/miniconda3}"

# --- manifest field parsing (RFC-style: fields may be double-quoted and contain commas) ---------------
# Emits one record per data row, fields in manifest column order, separated by the ASCII Unit Separator
# (0x1f). A non-whitespace separator is required so that EMPTY fields (e.g. py_var on the `ivc` rows) are
# preserved: bash `read`/awk collapse runs of a whitespace IFS, which would silently drop empty columns.
SEP=$'\x1f'
_manifest_records() {
  python3 - "${MANIFEST}" <<'PY'
import csv, sys
SEP = "\x1f"
with open(sys.argv[1], newline="") as fh:
    r = csv.reader(fh)
    header = next(r)
    for row in r:
        if not row or not any(c.strip() for c in row):
            continue
        print(SEP.join(row))
PY
}

# All distinct model names in the manifest (preserves first-seen order).
manifest_models() {
  _manifest_records | awk -F"${SEP}" '!seen[$1]++ {print $1}'
}

# All manifest rows whose model column matches $1 case-insensitively (US-separated, as _manifest_records).
manifest_rows_for_model() {
  local want="$1"
  _manifest_records | awk -F"${SEP}" -v w="${want}" 'tolower($1)==tolower(w)'
}

# All manifest rows, CPU `ivc` rows first then GPU families (stable within each group).
manifest_rows_ordered() {
  { _manifest_records | awk -F"${SEP}" '$4=="ivc"'
    _manifest_records | awk -F"${SEP}" '$4!="ivc"'; }
}

# Resolve the interpreter for a row given (py_var, conda_env). Echoes the path; empty if unresolvable.
# Resolution order: explicit $py_var if set -> $CONDA_ROOT/envs/<env>/bin/python -> "" .
# The `ivc` core rows (empty py_var) resolve to the repo .venv python.
resolve_interpreter() {
  local py_var="$1" conda_env="$2"
  if [ -z "${py_var}" ]; then
    echo "${VENV_PY}"; return 0
  fi
  local override="${!py_var:-}"
  if [ -n "${override}" ]; then
    echo "${override}"; return 0
  fi
  local guess="${CONDA_ROOT_DIR}/envs/${conda_env}/bin/python"
  if [ -x "${guess}" ]; then
    echo "${guess}"; return 0
  fi
  echo ""
}

# Is a host ready to RUN this row? Sets PREFLIGHT_DETAIL on failure. Returns 0 = ready, 1 = not ready.
# This is a pure check (no side effects), so the Level-3 preflight report can call it too.
row_is_ready() {
  local conda_env="$2" py_var="$3" data_vars="$4" runner_cmd="$5"
  PREFLIGHT_DETAIL=""
  if [ -z "${runner_cmd}" ]; then
    PREFLIGHT_DETAIL="no in-repo runner (scPerturBench harness)"
    return 1
  fi
  # interpreter
  local interp; interp="$(resolve_interpreter "${py_var}" "${conda_env}")"
  if [ -z "${interp}" ] || [ ! -x "${interp}" ]; then
    if [ -n "${py_var}" ]; then
      PREFLIGHT_DETAIL="set \$${py_var} to ${conda_env}'s bin/python (or create \$CONDA_ROOT/envs/${conda_env})"
    else
      PREFLIGHT_DETAIL="core .venv missing - run \`make setup\`"
    fi
    return 1
  fi
  # data paths
  local v
  for v in ${data_vars}; do
    if [ -z "${!v:-}" ]; then
      PREFLIGHT_DETAIL="set \$${v} (raw data path; see data/README.md)"
      return 1
    fi
  done
  return 0
}

# Run (or dry-run) a single manifest row. $1 = TAB-separated record, $2 = dry_run (0/1).
# Sets ROW_MODEL, ROW_CLUSTER and (on skip) SKIP_REASON. Returns 0/10/20/<runner exit>.
run_manifest_row() {
  local record="$1" dry="$2"
  IFS="${SEP}" read -r ROW_MODEL ROW_CLUSTER row_gpu conda_env py_var data_vars runner_cmd notes <<<"${record}"
  SKIP_REASON=""

  echo ""
  echo "------------------------------------------------------------------"
  echo " ${ROW_MODEL}  (${ROW_CLUSTER})   env=${conda_env}  gpu=${row_gpu}"
  echo "------------------------------------------------------------------"

  if ! row_is_ready "${ROW_MODEL}" "${conda_env}" "${py_var}" "${data_vars}" "${runner_cmd}"; then
    SKIP_REASON="${PREFLIGHT_DETAIL}"
    if [ -z "${runner_cmd}" ]; then
      echo " SKIP: ${ROW_MODEL} ${ROW_CLUSTER} is run through the scPerturBench eval harness, not from"
      echo "       this repo (its in-repo registry path needs a model_runners/ dir that is absent)."
      echo "       To reproduce this cell see REPRODUCE.md > Retraining from scratch (scPerturBench)."
    else
      echo " SKIP: ${PREFLIGHT_DETAIL}."
      echo "       Reference: REPRODUCE.md (env table + \$IVCBENCH_* reference) and data/README.md."
    fi
    return 20
  fi

  local interp; interp="$(resolve_interpreter "${py_var}" "${conda_env}")"
  # Build the argv: the @PY@ token is the interpreter (resolved path may contain spaces, so it is kept as
  # a single array element, never word-split); the rest of runner_cmd is split on whitespace into args
  # (the manifest uses only simple space-separated flag tokens, no spaces-within-an-arg).
  local rest="${runner_cmd#@PY@}"          # drop the leading @PY@ token
  local args=(); read -ra args <<<"${rest}"

  # Refreeze compact mean bundles into IVCBENCH_PRED_DUMP (default: predictions/) so reproduce-eval
  # re-scores the retrained model. Set BEFORE the runner sees the environment.
  export IVCBENCH_PRED_DUMP="${IVCBENCH_PRED_DUMP:-predictions}"
  export IVCBENCH_PRED_DUMP_MEANS=1

  echo " interpreter : ${interp}"
  echo " IVCBENCH_PRED_DUMP=${IVCBENCH_PRED_DUMP}  IVCBENCH_PRED_DUMP_MEANS=1"
  echo " command     : ${interp} ${args[*]}"

  if [ "${dry}" -eq 1 ]; then
    echo " (--dry-run: not executing)"
    return 10
  fi

  # Run without -e so a single runner failure is captured, not fatal to the driver. The interpreter is
  # quoted as one word so a path containing spaces is preserved.
  set +e
  ( "${interp}" "${args[@]}" )
  local rc=$?
  set -e
  if [ "${rc}" -ne 0 ]; then
    echo " runner exited ${rc} (see output above); continuing with the rest."
  else
    echo " done."
  fi
  return "${rc}"
}

# Re-score every deposited / refrozen bundle GPU-free and print how many rows came back.
reproduce_eval_and_report() {
  local out="reproduced_results.csv"
  "${VENV_PY}" scripts/reproduce_eval.py 'predictions/**/*.npz' 'predictions/*.npz' -o "${out}"
  local n
  n="$(tail -n +2 "${out}" | grep -c . || true)"
  echo "re-scored ${n} bundles -> ${out}"
}
