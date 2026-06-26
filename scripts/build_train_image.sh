#!/usr/bin/env bash
# build_train_image.sh: build the scPerturBench-style ALL-ENVIRONMENTS training image.
#
#   bash scripts/build_train_image.sh                 # conda-pack each env, then podman build
#   bash scripts/build_train_image.sh --pack-only     # only produce the env tarballs, skip the build
#   bash scripts/build_train_image.sh --envs "cellot ivc-state"   # restrict to a subset
#
# Level 1 (the in-repo Containerfile / `podman build -t ivcbench .`) is the small, verified, GPU-free
# eval image. THIS is the heavy retraining image: it bundles the per-family conda environments so a
# reader with a GPU can run `make train-all` without building each upstream env by hand. conda-pack
# freezes each existing env into build/train_envs/<env>.tar.gz; Containerfile.train unpacks them into
# /opt/conda/envs/<env>. The result is LARGE (the prior full build was about 70 GB), so it is an AUTHOR
# RELEASE ARTIFACT built once here and hosted on Zenodo, NOT shipped in the git repo (build/ is ignored).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# The per-family heavy environments to bundle (override with --envs "a b c").
ENVS="cellot ivc-cpa ivc-scpram ivc-state scgpt scfoundation scperturbench_eval"
OUT="${ROOT}/build/train_envs"
IMAGE="ivcbench-train"
PACK_ONLY=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --pack-only) PACK_ONLY=1; shift ;;
    --envs) ENVS="${2:?--envs needs a quoted env list}"; shift 2 ;;
    -h|--help) echo "usage: bash scripts/build_train_image.sh [--pack-only] [--envs \"a b c\"]"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; echo "usage: bash scripts/build_train_image.sh [--pack-only] [--envs \"a b c\"]" >&2; exit 2 ;;
  esac
done

command -v conda-pack >/dev/null 2>&1 || { echo "!! conda-pack not found on PATH (conda install -c conda-forge conda-pack)." >&2; exit 1; }
mkdir -p "${OUT}"

echo "=================================================================="
echo " ivcbench training image: conda-pack the per-family environments"
echo "=================================================================="
echo " envs : ${ENVS}"
echo " out  : ${OUT}"
[ "${PACK_ONLY}" -eq 1 ] && echo " (--pack-only: tarballs only; skipping podman build)"
echo ""

PACKED=()
for env in ${ENVS}; do
  tarball="${OUT}/${env}.tar.gz"
  if [ -s "${tarball}" ]; then
    echo ">> ${env}: tarball already exists, skipping pack (${tarball})"
    PACKED+=("${env}")
    continue
  fi
  echo ">> ${env}: conda-pack ..."
  # --ignore-editable-packages: the upstream model repos are pip-installed editable into their envs
  #   (e.g. cellot), which conda-pack otherwise refuses; the repo is COPY'd into the image separately.
  # --ignore-missing-files: tolerate files a package recorded but that are absent on disk.
  if conda pack -n "${env}" -o "${tarball}" --n-threads 4 --ignore-editable-packages --ignore-missing-files; then
    PACKED+=("${env}")
  else
    echo "!! ${env}: conda-pack failed or env absent on this host, continuing without it." >&2
    rm -f "${tarball}"
  fi
done

echo ""
echo "------------------------------------------------------------------"
echo " packed ${#PACKED[@]} env(s): ${PACKED[*]:-<none>}"
if [ -d "${OUT}" ]; then
  echo " total tarball size:"
  du -ch "${OUT}"/*.tar.gz 2>/dev/null | tail -n 1 | sed 's/^/   /'
fi
echo ""
echo " NOTE: the resulting image bundles every packed env and is LARGE (tens of GB; the prior full"
echo "       build was about 70 GB). It is an author release artifact for Zenodo, not a repo image."
echo ""

# Write an env file that maps each family's IVCBENCH_*_PY variable to its in-image interpreter, so a
# reader can `source` it inside the container and the train driver resolves every family directly.
ENVFILE="${OUT}/train_image.env"
{
  echo "# Source this inside the ivcbench-train image so the train driver finds each family's python."
  echo "# (The driver also auto-resolves /opt/conda/envs/<env>/bin/python when CONDA_ROOT=/opt/conda.)"
  echo "export CONDA_ROOT=/opt/conda"
  echo "export IVCBENCH_CELLOT_PY=/opt/conda/envs/cellot/bin/python"
  echo "export IVCBENCH_IVC_CPA_PYTHON=/opt/conda/envs/ivc-cpa/bin/python"
  echo "export IVCBENCH_IVC_SCPRAM_PYTHON=/opt/conda/envs/ivc-scpram/bin/python"
  echo "export IVCBENCH_IVC_STATE_PYTHON=/opt/conda/envs/ivc-state/bin/python"
  echo "export IVCBENCH_SCGPT_PYTHON=/opt/conda/envs/scgpt/bin/python"
  echo "export IVCBENCH_SCFOUNDATION_PYTHON=/opt/conda/envs/scfoundation/bin/python"
  echo "export IVCBENCH_SCPERTURBENCH_EVAL_PYTHON=/opt/conda/envs/scperturbench_eval/bin/python"
} > "${ENVFILE}"
echo " wrote ${ENVFILE} (per-family IVCBENCH_*_PY for use inside the image)."

if [ "${PACK_ONLY}" -eq 1 ]; then
  echo ""
  echo "DONE (--pack-only): tarballs in ${OUT}. Run without --pack-only to build the image."
  exit 0
fi

command -v podman >/dev/null 2>&1 || { echo "!! podman not found; tarballs are ready in ${OUT}. Install podman (or use docker) to build." >&2; exit 1; }

echo ""
echo "------------------------------------------------------------------"
echo " podman build -f Containerfile.train --ignorefile .dockerignore.train -t ${IMAGE} ."
echo "------------------------------------------------------------------"
# .dockerignore.train re-includes build/train_envs/ (the packed envs this image COPYs); the default
# .dockerignore excludes build/ entirely so the small eval image never ingests these multi-GB tarballs.
podman build -f Containerfile.train --ignorefile .dockerignore.train -t "${IMAGE}" .
echo ""
echo "DONE: built ${IMAGE} (LARGE). Inside it, run \`make train-all\` with the family pythons at"
echo "      /opt/conda/envs/<env>/bin/python (source ${ENVFILE} to set the IVCBENCH_*_PY variables)."
