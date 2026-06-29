#!/usr/bin/env bash
# Frangieh 2021 Perturb-CITE-seq (C4), PUBLIC processed h5ad, fetched from the scPerturb deposit.
# The files live on Zenodo record 13350497 (DOI 10.5281/zenodo.13350497). Rather than hardcode the
# exact filenames (scPerturb has renamed them across releases), we ask the Zenodo REST API for the
# record's file list and pick the Frangieh .h5ad entries by name. Resumable; records checksums.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/C4/frangieh"
RECORD="13350497"
API="https://zenodo.org/api/records/${RECORD}"
PAGE="https://zenodo.org/records/${RECORD}"
mkdir -p "${DEST}"

echo ">> Asking the Zenodo API for record ${RECORD}'s file list ..."
JSON="$(curl -fsSL "${API}" 2>/dev/null || true)"
if [ -z "${JSON}" ]; then
  echo "!! Could not reach the Zenodo API (${API})." >&2
  echo "   Open the record page and download the Frangieh *.h5ad files by hand into:" >&2
  echo "     ${DEST}" >&2
  echo "   ${PAGE}" >&2
  exit 1
fi

# Extract (filename, download-url) pairs for Frangieh .h5ad entries (case-insensitive on "frangieh").
# Zenodo's file objects expose the link under .links.self (current API) or .links.download (older form).
mapfile -t PAIRS < <(printf '%s' "${JSON}" | python3 - <<'PY'
import json, sys
data = json.load(sys.stdin)
for f in data.get("files", []):
    key = f.get("key") or f.get("filename") or ""
    if "frangieh" in key.lower() and key.lower().endswith(".h5ad"):
        links = f.get("links", {}) or {}
        url = links.get("self") or links.get("download") or ""
        if url:
            print(f"{key}\t{url}")
PY
)

if [ "${#PAIRS[@]}" -eq 0 ]; then
  echo "!! No Frangieh *.h5ad file found in record ${RECORD} via the API." >&2
  echo "   The record may have been re-versioned; check the file list on the record page and" >&2
  echo "   download the Frangieh Perturb-CITE-seq h5ad(s) by hand into:" >&2
  echo "     ${DEST}" >&2
  echo "   ${PAGE}" >&2
  exit 1
fi

echo ">> Found ${#PAIRS[@]} Frangieh h5ad file(s) on record ${RECORD}:"
for p in "${PAIRS[@]}"; do echo "   - ${p%%$'\t'*}"; done

for p in "${PAIRS[@]}"; do
  name="${p%%$'\t'*}"
  url="${p#*$'\t'}"
  out="${DEST}/${name}"
  if [ -s "${out}" ]; then
    echo ">> ${name} already present (non-empty), skipping."
    continue
  fi
  echo ">> ${name}"
  curl -fL -C - -o "${out}" "${url}"
done

echo "== recording checksums =="
(
  cd "${ROOT}"
  find data/C4/frangieh -type f -name '*.h5ad' -print0 | xargs -0 sha256sum > data/C4/frangieh/frangieh.sha256
)
echo "DONE: Frangieh Perturb-CITE-seq h5ad (C4), scPerturb Zenodo ${RECORD}"
