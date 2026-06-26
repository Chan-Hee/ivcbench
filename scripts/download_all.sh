#!/usr/bin/env bash
# download_all.sh: one command for ALL public census raw data.
#
#   bash scripts/download_all.sh            # fetch everything public
#   bash scripts/download_all.sh --list     # print the plan, download nothing
#
# It runs each public download script in turn, each guarded so one failure reports and the rest still
# run, then names the access-controlled datasets that need a manual login/DAC, and ends with a SUMMARY
# of what is now present under data/ and which IVCBENCH_* path variables to export before retraining.
# This covers the PUBLIC census only; the access-controlled deposits (Chen, Cano-Gamez) are listed but
# not fetched, because they sit behind a login / data-access committee.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS="${ROOT}/scripts"

LIST_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --list) LIST_ONLY=1 ;;
    -h|--help) echo "usage: bash scripts/download_all.sh [--list]"; exit 0 ;;
    *) echo "unknown flag: ${arg}" >&2; echo "usage: bash scripts/download_all.sh [--list]" >&2; exit 2 ;;
  esac
done

# Public download steps: "label|script|covers".
PUBLIC_STEPS=(
  "C1 Kang + C3 Shifrut/Schmidt/McCutcheon + C4 Belk|download_public.sh|GSE96583, GSE119450, GSE190604, GSE218985, GSE203592"
  "C2 Soskic processed h5ad|download_soskic.sh|trynkalab processed CD4+ activation h5ad"
  "C5 OP3 / Szalata|download_op3.sh|GSE279945 chemical-perturbation matrix"
  "C4 Frangieh Perturb-CITE-seq|download_frangieh.sh|scPerturb Zenodo 13350497 h5ad"
)

echo "=================================================================="
echo " ivcbench raw-data download (PUBLIC census)"
echo "=================================================================="
echo " Destination layout: data/<cluster>/<dataset>/ (see data/README.md)."
[ "${LIST_ONLY}" -eq 1 ] && echo " (--list: printing the plan; nothing is downloaded)"
echo ""

echo "------------------------------------------------------------------"
echo " PLAN (public, fetched by this script)"
echo "------------------------------------------------------------------"
printf " %-46s  %s\n" "DATASET" "SCRIPT / SOURCE"
for step in "${PUBLIC_STEPS[@]}"; do
  IFS="|" read -r label script covers <<<"${step}"
  printf " %-46s  scripts/%s\n" "${label}" "${script}"
  printf " %-46s  (%s)\n" "" "${covers}"
done
echo ""
echo "------------------------------------------------------------------"
echo " ACCESS-CONTROLLED (manual, NOT fetched here)"
echo "------------------------------------------------------------------"
echo " Chen 2025 FOXP3 Perturb-icCITE-seq (C3)"
echo "   DDBJ/GEA login: PRJDB16517 / E-GEAD-648  ->  data/C3/chen/"
echo " Cano-Gamez CD4+ effectorness (C1)"
echo "   EGA DAC: EGAS00001003215  (see scripts/apply_ega_dac.md)  ->  data/C1/cano_gamez/"
echo ""

if [ "${LIST_ONLY}" -eq 1 ]; then
  echo "------------------------------------------------------------------"
  echo " (--list) After downloading, export the raw-data path variables before retraining:"
  echo "   IVCBENCH_KANG_PATH, IVCBENCH_SOSKIC_PATH, IVCBENCH_OP3_PATH, IVCBENCH_FRANGIEH_DIR,"
  echo "   IVCBENCH_SCPERTURBENCH_DATASET_DIR  (the \$IVCBENCH_* table in REPRODUCE.md)."
  echo " See data/README.md for the per-dataset accessions and download notes."
  exit 0
fi

# --- run each public step, guarded; collect a status line per step --------------------------------
STATUS=()
for step in "${PUBLIC_STEPS[@]}"; do
  IFS="|" read -r label script covers <<<"${step}"
  echo "=================================================================="
  echo " >> ${label}   (scripts/${script})"
  echo "=================================================================="
  rc=0
  bash "${SCRIPTS}/${script}" || rc=$?
  if [ "${rc}" -eq 0 ]; then
    STATUS+=("OK      ${label}")
  else
    STATUS+=("FAILED  ${label}  (scripts/${script} exit ${rc})")
  fi
  echo ""
done

# --- presence check: an expected sentinel file per dataset under data/ -----------------------------
# "label|glob-relative-to-data/". A glob that matches at least one non-empty file counts as PRESENT.
EXPECT=(
  "C1 Kang|C1/kang/GSE96583_RAW.tar"
  "C2 Soskic|C2/soskic/*.h5ad"
  "C3 Shifrut|C3/shifrut/GSE119450_RAW.tar"
  "C3 Schmidt|C3/schmidt/GSE190604_matrix.mtx.gz"
  "C3 McCutcheon|C3/mccutcheon/GSE218985_RAW.tar"
  "C4 Belk|C4/belk/GSE203592_integrated_v2.rds.gz"
  "C4 Frangieh|C4/frangieh/*.h5ad"
  "C5 OP3|C5/op3/*.h5ad"
)

present_glob() {  # $1 = glob relative to data/ ; 0 if at least one non-empty match exists
  local f
  shopt -s nullglob
  # quote the base path: this checkout's path contains a space ("immune virtual cell"), which an
  # unquoted glob word-splits on, so present datasets were falsely reported MISSING. Leave $1 unquoted
  # so any '*' in the sentinel still expands.
  for f in "${ROOT}/data/"$1; do [ -s "${f}" ] && { shopt -u nullglob; return 0; }; done
  shopt -u nullglob
  return 1
}

echo "=================================================================="
echo " SUMMARY"
echo "=================================================================="
echo " Per-step result:"
for s in "${STATUS[@]}"; do echo "   ${s}"; done
echo ""
echo " Datasets now present under data/ (sentinel-file check):"
N_PRESENT=0
N_MISSING=0
for e in "${EXPECT[@]}"; do
  IFS="|" read -r label glob <<<"${e}"
  if present_glob "${glob}"; then
    printf "   PRESENT  %-16s (%s)\n" "${label}" "${glob}"
    N_PRESENT=$((N_PRESENT + 1))
  else
    printf "   MISSING  %-16s (%s)\n" "${label}" "${glob}"
    N_MISSING=$((N_MISSING + 1))
  fi
done
echo ""
echo " ${N_PRESENT} present, ${N_MISSING} missing (of the public sentinels above)."
echo " Access-controlled Chen (C3) and Cano-Gamez (C1) are never fetched here; obtain them by login/DAC."
echo ""
echo " Before retraining, export the raw-data path variables (the \$IVCBENCH_* table in REPRODUCE.md):"
echo "   IVCBENCH_KANG_PATH, IVCBENCH_SOSKIC_PATH, IVCBENCH_OP3_PATH, IVCBENCH_FRANGIEH_DIR,"
echo "   IVCBENCH_SCPERTURBENCH_DATASET_DIR  (see data/README.md for the per-dataset notes)."
