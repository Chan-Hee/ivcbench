"""Kang 2018 / GSE96583 (C1) loader — real PBMC IFN-β cytokine-response ingestion.

Uses the canonical IFN-β experiment = GSE96583 **batch2** (lanes 2.1 + 2.2; the batch with the
`stim` ctrl/stim column). Reads the two MatrixMarket lanes from GSE96583_RAW.tar, reconstructs the
merged cell index exactly as the published tSNE metadata does (colliding barcodes between lanes get
the `-1`→`-11` suffix; counts reconcile: 28 752 `-1` + 313 `-11` = 29 065 = 14 619 + 14 446), joins
per-cell (donor `ind`, `stim`, `cell` type, singlet/doublet) from GSE96583_batch2.total.tsne.df,
keeps singlets, library-log-normalizes + HVG-selects, and maps to the unified CellSet schema:
perturbation = 'IFN-beta' (stim) / 'control' (ctrl), cell_type_coarse from `cell`, donor_id = `ind`.

This is scGen's original benchmark dataset (cross-cell-type IFN-β response prediction), so C1-real is
also a pipeline-validation anchor: a latent model that does well here is behaving as published.
"""
from __future__ import annotations

import gzip
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..preprocess import PreprocessConfig, preprocess
from ..schema import CONTROL_TOKEN

# Kang `cell` label -> coarse lineage token
_CT_MAP = {
    "CD4 T cells": "CD4T", "CD8 T cells": "CD8T", "CD14+ Monocytes": "Mono_CD14",
    "FCGR3A+ Monocytes": "Mono_FCGR3A", "NK cells": "NK", "B cells": "B",
    "Dendritic cells": "DC", "Megakaryocytes": "Mk",
}
_BATCH2_LANES = ["GSM2560248_2.1.mtx.gz", "GSM2560249_2.2.mtx.gz"]
_BATCH2_BARCODES = ["GSM2560248_barcodes.tsv.gz", "GSM2560249_barcodes.tsv.gz"]


def _read_mtx_genes_x_cells(fobj) -> sp.csr_matrix:
    from scipy.io import mmread
    return sp.csr_matrix(mmread(fobj)).T.tocsr()        # → cells × genes


def load(path: str | Path = "data/C1/kang", subsample_per_group: int = 80,
         n_hvg: int = 2000, seed: int = 0) -> CellSet:  # noqa: F821
    path = Path(os.environ.get("IVCBENCH_KANG_PATH", str(path)))
    tar_p = path / "GSE96583_RAW.tar"
    genes_p = path / "GSE96583_batch2.genes.tsv.gz"
    meta_p = path / "GSE96583_batch2.total.tsne.df.tsv.gz"
    if not tar_p.exists():
        raise FileNotFoundError(f"Kang: {tar_p} not found")

    # genes (ensembl, symbol) — batch2 single gene set, 35 635 genes
    with gzip.open(genes_p, "rt") as fh:
        gdf = pd.read_csv(fh, sep="\t", header=None)
    symbols = gdf.iloc[:, 1].astype(str).to_numpy()

    # per-cell metadata, indexed by the merged barcode key
    with gzip.open(meta_p, "rt") as fh:
        meta = pd.read_csv(fh, sep="\t", index_col=0)
    meta.columns = [c.strip() for c in meta.columns]

    Xs, keys = [], []
    seen: set[str] = set()
    with tarfile.open(tar_p) as tar:
        for lane_i, (mtx_name, bc_name) in enumerate(zip(_BATCH2_LANES, _BATCH2_BARCODES)):
            with gzip.open(tar.extractfile(mtx_name)) as fh:
                X = _read_mtx_genes_x_cells(fh)          # cells × genes
            with gzip.open(tar.extractfile(bc_name), "rt") as fh:
                bcs = [b.strip() for b in fh]
            assert X.shape[0] == len(bcs), f"{mtx_name}: {X.shape[0]} cells vs {len(bcs)} barcodes"
            # reconstruct the published merged index: lane1 keeps '-1'; a lane2 barcode that already
            # appeared in lane1 gets an extra '1' ('-1'→'-11'), matching the tSNE-df convention.
            lane_keys = []
            for b in bcs:
                k = b if (lane_i == 0 or b not in seen) else b + "1"
                lane_keys.append(k)
            seen.update(bcs)
            Xs.append(X)
            keys.extend(lane_keys)
    X_all = sp.vstack(Xs).tocsr()
    keys = np.asarray(keys)

    # align cells to metadata (only barcodes present in the published tSNE df), keep singlets
    in_meta = np.array([k in meta.index for k in keys])
    X_all, keys = X_all[in_meta], keys[in_meta]
    m = meta.loc[keys]
    singlet = m["multiplets"].astype(str).str.lower().eq("singlet").to_numpy()
    valid_ct = m["cell"].astype(str).isin(_CT_MAP).to_numpy()
    valid_stim = m["stim"].astype(str).isin(["ctrl", "stim"]).to_numpy()
    keep = singlet & valid_ct & valid_stim
    X_all, m = X_all[keep], m.loc[keep]

    stim = m["stim"].astype(str).to_numpy()
    is_ctrl = stim == "ctrl"
    pert = np.where(is_ctrl, CONTROL_TOKEN, "IFN-beta")
    ct = m["cell"].astype(str).map(_CT_MAP).to_numpy()
    donor = m["ind"].astype(str).to_numpy()

    # cap cells per (stim, cell type, donor) to bound heavy-model runtime
    rng = np.random.default_rng(seed)
    grp = pd.Series(stim).astype(str) + "|" + pd.Series(ct) + "|" + pd.Series(donor)
    pos = np.arange(X_all.shape[0])
    sel = []
    for _, g in pd.Series(pos).groupby(grp.to_numpy()):
        v = g.to_numpy()
        sel.append(v if len(v) <= subsample_per_group else rng.choice(v, subsample_per_group, replace=False))
    idx = np.sort(np.concatenate(sel))
    X_all = X_all[idx]
    pert, ct, donor, is_ctrl = pert[idx], ct[idx], donor[idx], is_ctrl[idx]

    obs_out = pd.DataFrame({
        "cell_type_coarse": ct, "cell_type_fine": ct, "perturbation": pert,
        "condition": np.where(is_ctrl, "ctrl", "IFN-beta"), "donor_id": donor,
        "timepoint": "6h", "batch": donor, "is_control": is_ctrl,
    })

    cs = preprocess(X_all, list(symbols), obs_out, side_info={},
                    uns={"dataset": "kang_GSE96583", "accession": "GSE96583",
                         "n_cells_total": int(X_all.shape[0]),
                         "cytokines": ["IFN-beta"]},
                    cfg=PreprocessConfig(n_hvg=n_hvg))
    return cs


if __name__ == "__main__":      # quick manual check
    cs = load()
    print("cells", cs.n_cells, "genes", cs.n_genes)
    print("cell types", sorted(set(cs.obs["cell_type_coarse"])))
    print("control frac", float(cs.obs["is_control"].mean()))
