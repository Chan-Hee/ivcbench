"""Cano-Gamez et al. 2020 CD4+ T-cell cytokine dataset -> CellSet (donor-transfer replication).

Nat Commun 2020;11:1801. The processed count matrix is public (no controlled access): EMBL-EBI
BioStudies S-BSST2978, the deposit the paper's own data-availability statement points to. Only the raw
reads sit behind EGA (EGAS00001003215 / EGAD00001005290).

We use the Soskic-analogous contrast: resting (UNS) versus anti-CD3/anti-CD28 activation without
polarizing cytokines (Th0), in naive and memory CD4+ T cells, for each of the four donors. This supports a
four-fold leave-one-donor-out replication of the donor-transfer result. Two differences from the Soskic
anchor must be stated wherever the replication is reported: the single-cell arm is profiled five days after
stimulation (Soskic is 0 h versus 16 h), and there are four donors rather than 106.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..preprocess import PreprocessConfig, preprocess
from ..schema import CONTROL_TOKEN

DEFAULT_DIR = "data/C1/cano_gamez"
_MTX = "NCOMMS-19-7936188_scRNAseq_raw_UMIs.mtx"
_GENES = "NCOMMS-19-7936188_scRNAseq_genes.tsv"
_BARCODES = "NCOMMS-19-7936188_scRNAseq_barcodes.tsv"
_META = "NCOMMS-19-7936188_metadata.txt"
ACTIVATION = "Th0"  # anti-CD3/CD28, no polarizing cytokine
RESTING = "UNS"


def load(
    path: str = DEFAULT_DIR,
    *,
    n_hvg: int = 2000,
    seed: int = 0,
    subsample_per_group: int = 400,
    conditions=(RESTING, ACTIVATION),
):
    """Return a CellSet of resting vs activated CD4+ T cells, four donors, naive and memory.

    Reads the deposited 10x triplet directly (no AnnData dependency), so the loader runs unchanged in
    every per-model environment.
    """
    import os
    import scipy.io as sio
    import scipy.sparse as sp

    meta = pd.read_csv(os.path.join(path, _META), sep="\t", index_col=0)
    genes = pd.read_csv(os.path.join(path, _GENES), sep="\t", header=None)
    var_names = genes.iloc[:, -1].astype(str).tolist()
    keep = meta["cytokine.condition"].astype(str).isin(list(conditions)).to_numpy()
    cache = os.path.join(path, "_counts_cells_x_genes.npz")
    if os.path.exists(cache):
        X_all = sp.load_npz(cache)
    else:  # first call parses the MatrixMarket file (~2 min) and caches the transposed CSR
        X_all = sp.csr_matrix(sio.mmread(os.path.join(path, _MTX)).T.tocsr())
        sp.save_npz(cache, X_all)
    if X_all.shape[0] != len(meta):
        raise ValueError(f"matrix has {X_all.shape[0]} cells, metadata has {len(meta)}")
    a = type("A", (), {})()
    a.X = X_all[keep]
    a.var_names = var_names
    a.n_obs = int(keep.sum())
    o = meta.loc[keep]
    is_ctrl = (o["cytokine.condition"].astype(str) == RESTING).to_numpy()
    ct = o["cell.type"].astype(str).to_numpy()
    donor = o["donor.id"].astype(str).to_numpy()

    # cap cells per (condition, cell type, donor) so the fold sizes are comparable across donors
    rng = np.random.default_rng(seed)
    grp = (
        pd.Series(np.where(is_ctrl, RESTING, ACTIVATION))
        + "|"
        + pd.Series(ct)
        + "|"
        + pd.Series(donor)
    )
    sel = []
    for _, g in pd.Series(np.arange(a.n_obs)).groupby(grp.to_numpy()):
        v = g.to_numpy()
        sel.append(
            v
            if len(v) <= subsample_per_group
            else rng.choice(v, subsample_per_group, replace=False)
        )
    idx = np.sort(np.concatenate(sel))
    X = a.X[idx]
    ct, donor, is_ctrl = ct[idx], donor[idx], is_ctrl[idx]

    obs = pd.DataFrame(
        {
            "cell_type_coarse": ct,
            "cell_type_fine": ct,
            "perturbation": np.where(is_ctrl, CONTROL_TOKEN, ACTIVATION),
            "condition": np.where(is_ctrl, "resting", "anti-CD3/CD28"),
            "donor_id": donor,
            "timepoint": "5d",
            "batch": donor,
            "is_control": is_ctrl,
        }
    )
    return preprocess(
        X,
        list(map(str, a.var_names)),
        obs,
        side_info={},
        uns={
            "dataset": "cano_gamez_2020",
            "accession": "S-BSST2978",
            "n_cells_total": int(X.shape[0]),
            "timepoint": "5d",
            "note": "resting vs Th0 activation; four donors; naive and memory CD4 T",
        },
        cfg=PreprocessConfig(n_hvg=n_hvg),
    )
