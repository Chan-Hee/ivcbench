#!/usr/bin/env bash
# Download the PUBLIC GEO datasets (no DAC) for C1 + C3. Resumable; records checksums.
# DAC (Cano-Gamez, Soskic), preprint (Zhu, Moonen, Belk, Zhou, Oesinghaus), and auth-gated
# (Frangieh SCP1064) datasets are NOT fetched here — see scripts/datasets.csv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

geo() { local a=$1; printf 'https://ftp.ncbi.nlm.nih.gov/geo/series/%s/%s/suppl' \
        "$(echo "$a" | sed -E 's/[0-9]{3}$/nnn/')" "$a"; }
dl()  { local url=$1 dest=$2; mkdir -p "$dest"; echo ">> $(basename "$url")";
        curl -fL -C - -o "$dest/$(basename "$url")" "$url"; }

# Kang 2018 (C1) — matrices + per-cell metadata (cell type, stim/ctrl) in batch tsne.df
B=$(geo GSE96583)
for f in GSE96583_RAW.tar GSE96583_batch1.genes.tsv.gz GSE96583_batch2.genes.tsv.gz \
         GSE96583_batch1.total.tsne.df.tsv.gz GSE96583_batch2.total.tsne.df.tsv.gz \
         GSE96583_genes.txt.gz; do dl "$B/$f" "$ROOT/data/C1/kang"; done

# Shifrut 2018 (C3)
dl "$(geo GSE119450)/GSE119450_RAW.tar" "$ROOT/data/C3/shifrut"

# Schmidt 2022 (C3) — 10x mtx + guide calls
B=$(geo GSE190604)
for f in GSE190604_barcodes.tsv.gz GSE190604_features.tsv.gz GSE190604_matrix.mtx.gz \
         GSE190604_cellranger-guidecalls-aggregated-unfiltered.txt.gz; do
  dl "$B/$f" "$ROOT/data/C3/schmidt"; done

# McCutcheon 2023 (C3) — scRNA subseries of GSE218988
dl "$(geo GSE218985)/GSE218985_RAW.tar" "$ROOT/data/C3/mccutcheon"

# Chen 2025 (C3) — Perturb-icCITE-seq FOXP3; 4 libraries (10x mtx + protospacer calls)
B=$(geo GSE255832)
for lib in MMA198_1 MMA198_2 MMA200_1 MMA202_1; do
  for suf in barcodes.tsv.gz features.tsv.gz matrix.mtx.gz protospacer_calls_per_cell.csv.gz; do
    dl "$B/GSE255832_${lib}_${suf}" "$ROOT/data/C3/chen"; done; done

# Belk 2022 (C4) — Perturb-seq integrated Seurat object (super-series GSE203593)
dl "$(geo GSE203592)/GSE203592_integrated_v2.rds.gz" "$ROOT/data/C4/belk"

echo "== recording checksums =="
find "$ROOT/data/C1/kang" "$ROOT/data/C3" "$ROOT/data/C4/belk" -type f \
     ! -name '*.sha256' ! -name '_*' -print0 | xargs -0 sha256sum > "$ROOT/data/public_downloads.sha256"
echo "DONE: public GEO datasets for C1 + C3"
