#!/usr/bin/env bash
# Download OP3 / Szałata 2024 (C5) — PUBLIC, no DAC required.
# GEO series: GSE279945. Confirms the exact supplementary filename before downloading.
set -euo pipefail

ACC="GSE279945"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/C5/op3"
mkdir -p "$DEST"

# GEO supplementary files live under this stable path (NNN = first digits, rest masked with 'nnn').
BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE279nnn/${ACC}/suppl"

echo ">> Listing supplementary files for ${ACC} ..."
curl -fsSL "${BASE}/" -o "${DEST}/_listing.html" || {
  echo "!! Could not list ${BASE}/ — confirm the accession / network and retry." >&2
  exit 1
}
grep -oE 'href="[^"]+"' "${DEST}/_listing.html" | sed 's/href="//; s/"//' | grep -viE '^\?|/$' \
  | tee "${DEST}/_files.txt"

echo
echo ">> Review ${DEST}/_files.txt, then download the processed matrix (e.g. *.h5ad / *RAW.tar):"
echo "   curl -fL -o '${DEST}/<file>' '${BASE}/<file>'"
echo
echo ">> After download, record checksum + provenance:"
echo "   cd <repo root> && sha256sum data/C5/op3/* >> data/manifest.csv"
echo
echo "NOTE: ingest with ivcbench.data.loaders.op3:load -> unified CellSet (subsample if needed)."
