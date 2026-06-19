"""Shared CRISPR Perturb-seq ingestion (C3) — reused by every gene-perturbation loader.

Provides 10x matrix reading and sgRNA→target-gene assembly so Shifrut / Schmidt / McCutcheon / Chen
loaders differ only in how they parse their guide-call file, not in how cells become a CellSet.
The true-LO-gene split holds out a target gene's ALL sgRNAs; control = non-targeting guides.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from .preprocess import PreprocessConfig, preprocess
from .schema import CONTROL_TOKEN, CellSet

_CTRL_RE = re.compile(r"(CTRL\d+|NO[_-]?TARGET|NON[_-]?TARGET|NONTARGETING|SAFE[_-]?HARBOR|AAVS1|SCRAMBLE|NTC|^NT$)",
                      re.IGNORECASE)


def is_control_guide(gene_token: str) -> bool:
    return bool(_CTRL_RE.search(str(gene_token)))


def target_gene_last_token(guide_id: str, sep: str = ".") -> str:
    """e.g. 'ES.sg26.PDCD1' -> 'PDCD1'."""
    return str(guide_id).split(sep)[-1]


def strip_trailing_index(guide_id: str) -> str:
    """e.g. 'ABCB10-1' -> 'ABCB10', 'BATF3_2' -> 'BATF3' ('NO-TARGET' stays)."""
    return re.sub(r"[-_]\d+$", "", str(guide_id))


def parse_guidecalls(path, gene_fn=strip_trailing_index, singlet_only: bool = True) -> pd.DataFrame:
    """Parse a 10x cellranger guide-call table (cols: cell_barcode, num_features, feature_call, ...).
    Returns DataFrame[cell_barcode, gene, is_control] (singlet-assigned cells only by default)."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    if singlet_only and "num_features" in df.columns:
        df = df[df["num_features"] == 1]
    call = df["feature_call"].astype(str)
    gene = call.map(gene_fn)
    is_ctrl = call.map(is_control_guide) | gene.map(is_control_guide)
    return pd.DataFrame({"cell_barcode": df["cell_barcode"].astype(str),
                         "gene": gene, "is_control": is_ctrl})


def split_guide_matrix(counts, gene_names, feat_types, gene_fn=strip_trailing_index):
    """For matrices with embedded 'CRISPR Guide Capture' features (e.g. McCutcheon): assign each cell
    its guide by argmax over guide rows, and return (gene-expression submatrix, gene symbols,
    per-cell target gene, per-cell is_control)."""
    feat_types = list(feat_types)
    guide_idx = [i for i, t in enumerate(feat_types) if "guide" in t.lower() or "crispr" in t.lower()]
    gene_idx = [i for i in range(len(feat_types)) if i not in set(guide_idx)]
    G = counts[:, guide_idx].tocsr()
    top = np.asarray(G.argmax(axis=1)).ravel()
    has = np.asarray(G.sum(axis=1)).ravel() > 0
    guide_names = [gene_names[i] for i in guide_idx]
    per_cell_gene = np.array([gene_fn(guide_names[t]) if h else None for t, h in zip(top, has)],
                             dtype=object)
    per_cell_ctrl = np.array([bool(is_control_guide(g)) if g is not None else False
                              for g in per_cell_gene])
    expr = counts[:, gene_idx].tocsr()
    return expr, [gene_names[i] for i in gene_idx], per_cell_gene, per_cell_ctrl


def _open(p: Path):
    p = Path(p)
    return gzip.open(p, "rt") if p.suffix == ".gz" else open(p)


def read_10x_mtx(directory: str | Path):
    """Read a 10x dir (matrix.mtx[.gz] + genes/features.tsv[.gz] + barcodes.tsv[.gz]).
    Returns (counts cells x features csr, barcodes, feature_symbols, feature_types)."""
    d = Path(directory)
    def find(*names):
        for n in names:
            for c in (d / n, d / (n + ".gz")):
                if c.exists():
                    return c
        raise FileNotFoundError(f"{names} not in {d}")
    mtx = scipy.io.mmread(str(find("matrix.mtx"))).tocsr()  # features x cells
    with _open(find("barcodes.tsv")) as fh:
        barcodes = [l.split("\t")[0].strip() for l in fh]
    genes, types = [], []
    with _open(find("features.tsv", "genes.tsv")) as fh:
        for l in fh:
            parts = l.rstrip("\n").split("\t")
            genes.append(parts[1] if len(parts) > 1 else parts[0])    # symbol if present
            types.append(parts[2] if len(parts) > 2 else "Gene Expression")
    return mtx.T.tocsr(), barcodes, genes, types  # -> cells x features


def assemble(blocks: list[dict], *, dataset: str, modality_col: str | None = None,
             cfg: PreprocessConfig = PreprocessConfig(), uns: dict | None = None) -> CellSet:
    """Concatenate per-sample blocks on the shared gene space, then run unified preprocessing.

    Each block: {counts (cells x genes csr), genes (list), obs (DataFrame with the 8 schema cols)}.
    """
    common = sorted(set.intersection(*[set(b["genes"]) for b in blocks]))
    pos = {g: i for i, g in enumerate(common)}
    mats, obs_parts = [], []
    for b in blocks:
        idx = np.array([b["genes"].index(g) for g in common])
        mats.append(b["counts"][:, idx])
        obs_parts.append(b["obs"])
    counts = sp.vstack(mats).tocsr()
    obs = pd.concat(obs_parts, ignore_index=True)
    obs["is_control"] = obs["is_control"].astype(bool)
    return preprocess(counts, common, obs, cfg=cfg,
                      uns={"dataset": dataset, "modality": "rna", **(uns or {})})


def guide_map_to_obs(barcodes: list[str], guide_df: pd.DataFrame, bc_col: str, guide_col: str,
                     *, donor: str, condition: str, batch: str, cell_type: str = "CD4T",
                     gene_fn=target_gene_last_token) -> pd.DataFrame:
    """Build the 8-column obs for a sample, assigning each cell its target gene / control."""
    g = (guide_df.drop_duplicates(bc_col).set_index(bc_col)[guide_col]
         if guide_df[bc_col].duplicated().any() else guide_df.set_index(bc_col)[guide_col])
    rows = []
    for bc in barcodes:
        gid = g.get(bc)
        if gid is None or (isinstance(gid, float) and np.isnan(gid)):
            gene, ctrl = None, False  # unassigned
        else:
            gene = gene_fn(gid)
            ctrl = is_control_guide(gene)
        pert = CONTROL_TOKEN if ctrl else gene
        rows.append(dict(cell_type_coarse=cell_type, cell_type_fine=cell_type,
                         perturbation=pert if pert is not None else "unassigned",
                         condition=condition, donor_id=donor, timepoint="NA",
                         batch=batch, is_control=ctrl))
    return pd.DataFrame(rows)
