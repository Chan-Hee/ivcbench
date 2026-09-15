#!/usr/bin/env bash
# CPA x T5u, re-trained with the chemical projection actually in the optimizer.
#
# The deposited 0.1116 came from cpa-tools 0.8.8, whose configure_optimizers omits
# PerturbationNetwork.pert_transformation -- verified in this env: 2 tensors, requires_grad=True,
# covered_by_upstream_optimizers=False. Chemistry therefore reached the decoder through a FIXED
# RANDOM linear map for that whole run, which is not the published behaviour. scripts/
# chemcpa_native_op3.py installs scripts/cpa_optimizer_shim.py, so this re-run trains it.
# Same protocol as the deposit: seeds 0,1,2 as technical repeats, max_cells 60000, epochs 60,
# cov_mode constant.
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="$HOME/miniconda3/envs/ivc-cpa/bin/python"
OUT="outputs/native_rerun/chemcpa_t5u_shim"
mkdir -p "$OUT"
for s in 0 1 2; do
  echo "=== chemCPA seed $s (shim) ==="
  "$PY" scripts/chemcpa_native_op3.py "$s" "$OUT" 60000 60 constant
done
echo "=== evaluate (seeds 0,1,2) ==="
"$PY" scripts/chemcpa_evaluate.py 0,1,2 "$OUT"
