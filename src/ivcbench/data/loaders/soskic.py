"""Soskic 2022 CD4-activation DONOR-axis loader (C2) — real 106-donor leave-one-donor-out ingestion.

Soskic et al. (Nat Genet 2022, GSE catalogue at the Cellular Genetics portal) profiled CD4+ T cells
from a large healthy cohort at rest and after anti-CD3/CD28 activation. The processed HVG portal files
give paired 0h (resting) and 16h (stimulated, highly-active) cells for the donors that passed QC.

This loader is the framework-native re-wrap of scripts/c2_soskic_donor.py:load_soskic_donor — SAME
106 donors, SAME per-(donor x condition x celltype) cap, SAME shared-gene joint re-standardization —
so the framework C2 path reproduces the bespoke per-donor scores. It maps to the unified CellSet:

    perturbation = 'stimulation' (16h) / control token (0h resting)
    is_control   = (timepoint == '0h')
    cell_type_coarse = Cell_type (CD4_Naive / CD4_Memory)  -- a within-donor stratum covariate
    donor_id     = Donor   (the C2 LODO generalization axis)
    timepoint    = '0h' / '16h'
    batch        = Plate

C2 is the Fig-1 OT-STRONG paired-stimulation donor-transfer cell: hold one donor's 16h cells out
entirely, predict its response from its OWN 0h cells. The held donor's stim cells never enter training
(leak-safe; enforced by the framework audit_split).
"""
from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as spx

from ..schema import CONTROL_TOKEN, CellSet, validate_cellset

# default data location (overridable via $IVCBENCH_SOSKIC_PATH)
_DEFAULT_DIR = "data/C2/soskic"
_REST = "restingCells_CD4only_HVGs_processed.h5ad"
_STIM = "stimulatedCells_highlyActiveCD4_16h_HVGs_processed.h5ad"

# Stimulation = the perturbation label for the 16h cells (control = 0h resting). One free-form condition.
_STIM_TOKEN = "stimulation"


def _dense(a):
    return a.toarray() if spx.issparse(a) else np.asarray(a)


def load(path: str | Path = _DEFAULT_DIR, cap_per_donor_cond: int = 300, seed: int = 0) -> CellSet:
    """106-donor Soskic CD4 0h/16h CellSet on the shared HVGs, jointly re-standardized.

    Per (donor x condition x celltype) cap so every donor is represented comparably and heavy-model
    runtime is bounded. Identical numerics to scripts/c2_soskic_donor.py:load_soskic_donor.
    """
    path = Path(os.environ.get("IVCBENCH_SOSKIC_PATH", str(path)))
    rest_p, stim_p = path / _REST, path / _STIM
    if not rest_p.exists() or not stim_p.exists():
        raise FileNotFoundError(f"Soskic: {rest_p} / {stim_p} not found")

    r = ad.read_h5ad(rest_p)
    s = ad.read_h5ad(stim_p)
    shared = sorted(set(map(str, r.var_names)) & set(map(str, s.var_names)))

    rng = np.random.default_rng(seed)
    Xs, meta = [], []
    for adata, ctrl in [(r, True), (s, False)]:
        adata = adata[:, shared]
        ct = adata.obs["Cell_type"].astype(str).to_numpy()
        don = adata.obs["Donor"].astype(str).to_numpy()
        plate = adata.obs["Plate"].astype(str).to_numpy()
        for state in ["CD4_Naive", "CD4_Memory"]:
            for d in np.unique(don):
                idx = np.where((ct == state) & (don == d))[0]
                if len(idx) == 0:
                    continue
                if len(idx) > cap_per_donor_cond:
                    idx = rng.choice(idx, cap_per_donor_cond, replace=False)
                Xs.append(_dense(adata.X[idx]).astype(np.float32))
                meta.append(pd.DataFrame(dict(
                    cell_type_coarse=state, cell_type_fine=state,
                    perturbation=(CONTROL_TOKEN if ctrl else _STIM_TOKEN), condition=_STIM_TOKEN,
                    donor_id=d, timepoint=("0h" if ctrl else "16h"), is_control=bool(ctrl),
                    batch=plate[idx])))
    X = np.vstack(Xs)
    obs = pd.concat(meta, ignore_index=True)
    # joint re-standardization on the shared panel (matches the bespoke pipeline exactly)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)

    cs = CellSet(
        X=X, obs=obs.reset_index(drop=True), var_names=shared, side_info={},
        uns=dict(dataset="soskic_CD4_16h_donor", accession="Soskic2022_CD4_activation",
                 modality="rna", timepoint="16h", n_cells_total=int(X.shape[0]),
                 n_donors=int(obs.donor_id.nunique()), cytokines=[]),
    )
    validate_cellset(cs)
    return cs


if __name__ == "__main__":      # quick manual check
    cs = load()
    print("cells", cs.n_cells, "genes", cs.n_genes, "donors", cs.obs.donor_id.nunique())
    print("cell types", sorted(set(cs.obs["cell_type_coarse"])))
    print("control frac", float(cs.obs["is_control"].mean()))
