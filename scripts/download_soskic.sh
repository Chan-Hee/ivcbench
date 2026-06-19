#!/usr/bin/env bash
# Soskic 2022 (C2) — PROCESSED single-cell h5ad, PUBLIC via the Trynka lab object store.
# This is the open processed release noted in the paper; it covers the C2 task (donor x timepoint x
# activation) WITHOUT the EGA DAC (which gates only the raw sequencing, EGAD00001008197).
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/C2/soskic"
mkdir -p "$DEST"
BASE="https://trynkalab.cog.sanger.ac.uk"
for f in restingCells_CD4only_HVGs_processed.h5ad \
         stimulatedCells_highlyActiveCD4_16h_HVGs_processed.h5ad \
         stimulatedCells_highlyActiveCD4_40h_HVGs_processed.h5ad \
         stimulatedCells_highlyActiveCD4_5d_HVGs_processed.h5ad \
         stimulatedCells_lowlyActiveCD4_HVGs_processed.h5ad; do
  echo ">> $f"
  curl -fL -C - -o "$DEST/$f" "$BASE/$f"
done
sha256sum "$DEST"/*.h5ad > "$DEST/soskic.sha256"
echo "DONE: Soskic processed h5ad (C2) — no DAC needed for the processed task"
