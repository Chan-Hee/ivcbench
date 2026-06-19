"""OP3 / Szałata 2024 (C5) loader — real GSE279945 single-cell ingestion.

Reads GSE279945_sc_counts_processed.h5ad, subsamples per (compound × cell type × donor), library-size
+ log1p normalizes, selects highly variable genes, maps obs to the unified schema, and attaches RDKit
Morgan fingerprints (from the per-cell SMILES) as compound-side conditioning. Returns a CellSet that
every downstream code path (splits, audit, baselines, metrics) already consumes.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..preprocess import PreprocessConfig, preprocess
from ..schema import CONTROL_TOKEN

# OP3 cell_type -> short coarse token
_CT_MAP = {
    "T cells CD4+": "CD4T", "T cells CD8+": "CD8T", "T regulatory cells": "Treg",
    "NK cells": "NK", "B cells": "B", "Myeloid cells": "Mono",
}


def _col(obs: pd.DataFrame, *cands: str) -> str | None:
    for c in cands:
        if c in obs.columns:
            return c
    low = {c.lower(): c for c in obs.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _morgan(smiles: str, n_bits: int = 1024, radius: int = 2):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprintAsNumPy(m).astype(np.float32)


def load(path: str | Path = "data/C5/op3/GSE279945_sc_counts_processed.h5ad",
         subsample_per_group: int = 40, n_hvg: int = 2000, seed: int = 0) -> CellSet:
    # default path matches the repo data layout (cf. chen.load default) so the registry's
    # _lazy("...op3","load") with no args resolves the on-disk OP3 file; override via env if needed.
    path = os.environ.get("IVCBENCH_OP3_PATH", str(path))
    import anndata

    ad = anndata.read_h5ad(str(path), backed="r")
    obs = ad.obs.copy()
    c_ct = _col(obs, "cell_type")
    c_sm = _col(obs, "sm_name", "compound", "perturbation")
    c_sml = _col(obs, "SMILES", "smiles", "canonical_smiles")
    c_ctrl = _col(obs, "control", "is_control")
    c_donor = _col(obs, "donor_id", "donor")
    c_plate = _col(obs, "plate_name", "plate", "library_id")
    if not all([c_ct, c_sm]):
        raise ValueError(f"OP3 obs missing cell_type/sm_name; columns: {list(obs.columns)[:30]}")

    is_ctrl = (obs[c_ctrl].astype(str).str.lower().isin(["true", "1"]) if c_ctrl is not None
               else obs[c_sm].astype(str).str.contains("Dimethyl", case=False, na=False))

    # subsample row positions: cap cells per (compound, cell type, donor)
    rng = np.random.default_rng(seed)
    key = obs[c_sm].astype(str) + "|" + obs[c_ct].astype(str) + "|" + (
        obs[c_donor].astype(str) if c_donor else "d0")
    pos = np.arange(len(obs))
    keep = []
    for _, g in pd.Series(pos).groupby(key.to_numpy()):
        v = g.to_numpy()
        keep.append(v if len(v) <= subsample_per_group
                    else rng.choice(v, subsample_per_group, replace=False))
    idx = np.sort(np.concatenate(keep))

    sub = ad[idx].to_memory()
    counts = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    sobs = sub.obs
    sm = sobs[c_sm].astype(str).to_numpy()
    ctrl = is_ctrl.to_numpy()[idx]
    pert = np.where(ctrl, CONTROL_TOKEN, sm)
    ct = sobs[c_ct].astype(str).map(lambda x: _CT_MAP.get(x, x)).to_numpy()
    donor = sobs[c_donor].astype(str).to_numpy() if c_donor else np.array(["d0"] * len(sobs))
    plate = sobs[c_plate].astype(str).to_numpy() if c_plate else donor

    obs_out = pd.DataFrame({
        "cell_type_coarse": ct, "cell_type_fine": ct, "perturbation": pert,
        "condition": "24h", "donor_id": donor, "timepoint": "24h",
        "batch": plate, "is_control": ctrl,
    })

    # Morgan fingerprints per non-control compound (from SMILES)
    fingerprint = {}
    if c_sml is not None:
        sml_map = (sobs[[c_sm, c_sml]].astype(str).drop_duplicates(c_sm)
                   .set_index(c_sm)[c_sml].to_dict())
        for name, smi in sml_map.items():
            if name in set(pert) and name != CONTROL_TOKEN:
                fp = _morgan(smi)
                if fp is not None:
                    fingerprint[name] = fp

    # unified preprocessing (shared with every cluster) — QC + library-log-norm + HVG
    cs = preprocess(counts, list(sub.var_names), obs_out, side_info={"fingerprint": fingerprint},
                    uns={"dataset": "op3_GSE279945", "accession": "GSE279945",
                         "n_cells_total": int(ad.n_obs), "immune_program": {"immunomod_moa": []}},
                    cfg=PreprocessConfig(n_hvg=n_hvg))
    cs.uns["compounds"] = sorted(set(cs.obs["perturbation"]) - {CONTROL_TOKEN})
    return cs
