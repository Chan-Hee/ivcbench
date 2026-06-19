"""Schmidt 2022 (C3) loader — GSE190604, primary human T-cell CRISPRa Perturb-seq.

One aggregated 10x matrix (genes x cells) + a cellranger guide-call table. Control = 'NO-TARGET'.
Guide naming 'GENE-<N>' -> target gene by stripping the trailing index.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from ..crispr import assemble, parse_guidecalls
from ..preprocess import PreprocessConfig
from ..schema import CONTROL_TOKEN


def _read_lines(path, col=0):
    with gzip.open(path, "rt") as f:
        return [l.rstrip("\n").split("\t")[col] for l in f]


def load(path: str | Path = "data/C3/schmidt", cfg: PreprocessConfig = PreprocessConfig(),
         subsample_per_gene: int = 300, seed: int = 0):
    root = Path(path)
    M = scipy.io.mmread(gzip.open(root / "GSE190604_matrix.mtx.gz", "rb")).tocsr()  # genes x cells
    counts = M.T.tocsr()                                                            # cells x genes
    barcodes = _read_lines(root / "GSE190604_barcodes.tsv.gz", 0)
    feats = [l.split("\t") for l in _read_lines_raw(root / "GSE190604_features.tsv.gz")]
    genes = [p[1] if len(p) > 1 else p[0] for p in feats]

    gc = parse_guidecalls(root / "GSE190604_cellranger-guidecalls-aggregated-unfiltered.txt.gz")
    gc = gc.drop_duplicates("cell_barcode").set_index("cell_barcode")

    gene_col, ctrl_col = [], []
    for bc in barcodes:
        if bc in gc.index:
            gene_col.append(gc.at[bc, "gene"]); ctrl_col.append(bool(gc.at[bc, "is_control"]))
        else:
            gene_col.append(None); ctrl_col.append(False)
    gene_col = np.array(gene_col, dtype=object)
    ctrl_col = np.array(ctrl_col)
    pert = np.array([CONTROL_TOKEN if c else g for g, c in zip(gene_col, ctrl_col)], dtype=object)

    keep = np.array([p is not None for p in pert])
    keep_idx = _cap_per_label(pert, keep, subsample_per_gene, seed)
    counts = counts[keep_idx]
    pert = pert[keep_idx]
    ctrl_col = ctrl_col[keep_idx]

    obs = pd.DataFrame({
        "cell_type_coarse": "CD4T", "cell_type_fine": "CD4T", "perturbation": pert,
        "condition": "CRISPRa", "donor_id": "agg", "timepoint": "NA",
        "batch": "GSE190604", "is_control": ctrl_col,
    })
    cs = assemble([dict(counts=counts, genes=genes, obs=obs)], dataset="schmidt_GSE190604", cfg=cfg,
                  uns={"accession": "GSE190604", "modality_label": "CRISPRa"})
    cs.uns["genes_perturbed"] = sorted(set(cs.obs["perturbation"]) - {CONTROL_TOKEN})
    return cs


def _read_lines_raw(path):
    with gzip.open(path, "rt") as f:
        return [l.rstrip("\n") for l in f]


def _cap_per_label(labels: np.ndarray, base_keep: np.ndarray, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.where(base_keep)[0]
    out = []
    for lab in pd.unique(labels[pos]):
        idx = pos[labels[pos] == lab]
        out.append(idx if len(idx) <= cap else rng.choice(idx, cap, replace=False))
    return np.sort(np.concatenate(out)) if out else np.array([], dtype=int)
