#!/usr/bin/env bash
# Kang 2018 (C1 anchor) — PUBLIC GSE96583. Best-effort fetch of GEO supplementary.
set -euo pipefail
DEST="$(dirname "$0")/../data/C1/kang"; mkdir -p "$DEST"
BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl"
echo ">> listing $BASE"
curl -fsSL "${BASE}/" -o "${DEST}/_listing.html"
grep -oE 'href="[^"]+"' "${DEST}/_listing.html" | sed 's/href="//; s/"//' | grep -viE '^\?|/$' | tee "${DEST}/_files.txt"
