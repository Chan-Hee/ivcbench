"""Chen 2025 (C3) loader — DDBJ PRJDB16517 / GEA E-GEAD-648, primary human CD4⁺ T FOXP3
Perturb-icCITE-seq (CROP-seq). Real Chen data (NOT GSE255832, which is Pretto 2025 mouse).

Structure (E-GEAD-648.processed.zip → per-run 10x dirs `GEA/transfer_*/<run>/`):
  • 15 libraries (Factor Value[library] in the SDRF), each with several modality runs:
      - gene expression  (36,601 features, all "Gene Expression")          → used
      - CROP-seq guide capture (907 sgRNAs named `<GENE>_<n>`, `NTC_<n>`)   → used (target assignment)
      - intracellular CITE (46 antibodies `intra_*`), hashing/ADT/TCR        → ignored
  • Target panel ≈ 295 FOXP3-regulator genes + NTC controls; guide → gene = strip `_<n>`.
  • GEX barcodes carry a `-1` suffix, guide barcodes do not → stripped to match.

Per library we read the GEX + guide matrices, intersect barcodes, assign each cell its target gene by
argmax over the guide-count matrix, then assemble across libraries onto a shared gene space and run
the unified preprocessing. Modality runs are classified by their feature table (no hard-coded run
IDs), so the loader is robust to the SDRF ordering.
"""
from __future__ import annotations

import re
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..crispr import assemble, is_control_guide, read_10x_mtx, strip_trailing_index
from ..preprocess import PreprocessConfig
from ..schema import CONTROL_TOKEN

ZIP = "E-GEAD-648.processed.zip"
SDRF = "E-GEAD-648.sdrf.txt"


def _strip_bc(bc: str) -> str:
    return re.sub(r"-\d+$", "", str(bc))


def _orient(counts, barcodes, names):
    """Force (n_cells, n_features). The GEX and guide .mtx files in E-GEAD-648 are stored with
    opposite orientation, so the uniform transpose in read_10x_mtx leaves one of them wrong; align
    by matching the axes to len(barcodes)/len(features)."""
    if counts.shape == (len(names), len(barcodes)) and len(names) != len(barcodes):
        return counts.T.tocsr()
    return counts


def _libraries_from_sdrf(root: Path) -> dict[str, list[str]]:
    """{library -> [run accession, ...]} from the SDRF (Factor Value[library] / Comment[SRA_RUN])."""
    import csv
    rows = list(csv.reader(open(root / SDRF), delimiter="\t"))
    hdr = rows[0]
    ci = {h: i for i, h in enumerate(hdr)}
    run_i, lib_i = ci["Comment[SRA_RUN]"], ci["Factor Value[library]"]
    libs: dict[str, list[str]] = {}
    for r in rows[1:]:
        libs.setdefault(r[lib_i], []).append(r[run_i])
    return libs


def _tarball_for(zf: zipfile.ZipFile, run: str) -> str | None:
    """Find the tarball member whose name contains this run id (handles combined `DRRa_DRRb`)."""
    for n in zf.namelist():
        if n.endswith(".tar.gz") and run in n:
            return n
    return None


def _extract_run(root: Path, zf: zipfile.ZipFile, member: str) -> Path | None:
    """Extract one tarball (if needed) and return the dir holding matrix/barcodes/features."""
    ex = root / "extracted"
    stem = member.split("/")[-1].replace(".tar.gz", "")
    marker = ex / stem
    if not marker.exists():
        ex.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as f, tarfile.open(fileobj=f, mode="r:gz") as t:
            t.extractall(marker)
    hits = list(marker.rglob("matrix.mtx.gz"))
    return hits[0].parent if hits else None


def _classify(dirpath: Path) -> str:
    """gex / guide / other from the 10x features table."""
    import gzip
    feats = dirpath / "features.tsv.gz"
    names, gex = [], 0
    with gzip.open(feats, "rt") as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            names.append(parts[0])
            if len(parts) >= 3 and parts[2] == "Gene Expression":
                gex += 1
    if gex > 10000:
        return "gex"
    if 500 <= len(names) <= 2000 and sum(bool(re.search(r"_\d+$", n)) for n in names) > 0.6 * len(names):
        return "guide"
    return "other"


def load(path: str | Path = "data/C3/chen", cfg: PreprocessConfig = PreprocessConfig(),
         libraries: list[str] | None = None, subsample_per_gene: int = 200, seed: int = 0):
    root = Path(path)
    rng = np.random.default_rng(seed)
    libs = _libraries_from_sdrf(root)
    want = libraries or list(libs)

    blocks = []
    with zipfile.ZipFile(root / ZIP) as zf:
        for lib in want:
            gex_dir = guide_dir = None
            for run in libs[lib]:
                member = _tarball_for(zf, run)
                if member is None:
                    continue
                d = _extract_run(root, zf, member)
                if d is None:
                    continue
                kind = _classify(d)
                if kind == "gex" and gex_dir is None:
                    gex_dir = d
                elif kind == "guide" and guide_dir is None:
                    guide_dir = d
            if gex_dir is None or guide_dir is None:
                continue

            gex_counts, gex_bc, genes, _ = read_10x_mtx(gex_dir)
            g_counts, g_bc, g_names, _ = read_10x_mtx(guide_dir)
            gex_counts = _orient(gex_counts, gex_bc, genes)
            g_counts = _orient(g_counts, g_bc, g_names)
            gex_bc = [_strip_bc(b) for b in gex_bc]
            g_bc = [_strip_bc(b) for b in g_bc]

            # assign each guide-matrix cell its target gene by argmax over guide counts
            g_arg = np.asarray(g_counts.argmax(axis=1)).ravel()
            g_tot = np.asarray(g_counts.sum(axis=1)).ravel()
            guide_gene = {b: (strip_trailing_index(g_names[a]) if t > 0 else None)
                          for b, a, t in zip(g_bc, g_arg, g_tot)}

            # keep GEX cells that have a confident guide call
            keep_rows, perts, is_ctrl = [], [], []
            for i, b in enumerate(gex_bc):
                gene = guide_gene.get(b)
                if gene is None:
                    continue
                ctrl = is_control_guide(gene)
                keep_rows.append(i)
                perts.append(CONTROL_TOKEN if ctrl else gene)
                is_ctrl.append(ctrl)
            if not keep_rows:
                continue
            keep_rows = np.array(keep_rows)
            perts = np.array(perts, dtype=object)
            is_ctrl = np.array(is_ctrl, dtype=bool)

            # cap cells per target within the library (bound memory; matches other C3 loaders)
            sel = []
            for lab in pd.unique(perts):
                idx = np.where(perts == lab)[0]
                sel.append(idx if len(idx) <= subsample_per_gene
                           else rng.choice(idx, subsample_per_gene, replace=False))
            sel = np.sort(np.concatenate(sel))
            rows = keep_rows[sel]
            obs = pd.DataFrame({
                "cell_type_coarse": "CD4T", "cell_type_fine": "CD4Tconv",
                "perturbation": perts[sel], "condition": "Stim", "donor_id": "mixture",
                "timepoint": "NA", "batch": lib, "is_control": is_ctrl[sel],
            })
            blocks.append(dict(counts=gex_counts[rows], genes=genes, obs=obs))

    if not blocks:
        raise RuntimeError("chen: no libraries assembled (check the E-GEAD-648 zip path)")
    cs = assemble(blocks, dataset="chen_E-GEAD-648", cfg=cfg,
                  uns={"accession": "PRJDB16517 / E-GEAD-648", "modality_label": "KO"})
    cs.uns["genes_perturbed"] = sorted(set(cs.obs["perturbation"]) - {CONTROL_TOKEN})
    return cs
