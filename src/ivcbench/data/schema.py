"""Unified data contract every dataset loader must produce.

Decoupling the in-memory container from AnnData keeps the split/audit/metric core testable
without scanpy/anndata installed. Real loaders (data/loaders/*.py) read .h5ad and emit a CellSet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Canonical obs columns. Loaders fill what applies; absent axes use the sentinel "NA".
OBS_COLUMNS = [
    "cell_type_coarse",  # e.g. CD4T / CD8T / B / NK / Mono  (C1 LOCT, C5 LOCT axis)
    "cell_type_fine",    # sub-lineage (C1 within-Oesinghaus resolution)
    "perturbation",      # cytokine / gene / compound label; control => the control token
    "condition",         # free-form experimental condition (cocktail, activation, etc.)
    "donor_id",          # C2 LODO axis; C5 blocking factor
    "timepoint",         # C2 temporal axis (0h/16h/40h/5d); "NA" if not time-resolved
    "batch",             # plate / replicate / sequencing batch
    "is_control",        # bool: control/vehicle cell (DMSO, PBS, 0h, NTC...) — never a held-out drug
]

CONTROL_TOKEN = "control"  # value of `perturbation` for control cells (DMSO/PBS/NTC mapped here)


@dataclass
class CellSet:
    """Minimal AnnData-like container the whole core operates on.

    X         : (n_cells, n_genes) float32 — log-normalized expression (or modality readout)
    obs       : DataFrame with OBS_COLUMNS, RangeIndex 0..n-1
    var_names : list[str] of length n_genes
    side_info : optional dict for perturbation-side representations
                (C5: {'fingerprint': {compound: np.ndarray}, 'scaffold': {...}}; C3 gene embeddings; ...)
    uns       : free metadata (dataset id, modality, provenance, checksums)
    """

    X: np.ndarray
    obs: pd.DataFrame
    var_names: list[str]
    side_info: dict[str, Any] = field(default_factory=dict)
    uns: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        return self.X.shape[1]

    def gene_index(self, names) -> np.ndarray:
        pos = {g: i for i, g in enumerate(self.var_names)}
        return np.array([pos[g] for g in names if g in pos], dtype=int)

    def subset(self, idx: np.ndarray) -> "CellSet":
        idx = np.asarray(idx, dtype=int)
        return CellSet(
            X=self.X[idx],
            obs=self.obs.iloc[idx].reset_index(drop=True),
            var_names=self.var_names,
            side_info=self.side_info,
            uns=self.uns,
        )


def validate_cellset(cs: CellSet) -> None:
    """Hard schema checks — run at the boundary between a loader and the core."""
    assert cs.X.ndim == 2, "X must be 2D (cells x genes)"
    assert cs.X.shape[0] == len(cs.obs), "X rows must match obs rows"
    assert cs.X.shape[1] == len(cs.var_names), "X cols must match var_names"
    missing = [c for c in OBS_COLUMNS if c not in cs.obs.columns]
    assert not missing, f"obs missing required columns: {missing}"
    assert cs.obs["is_control"].dtype == bool, "is_control must be boolean"
    # control cells must carry the control token in `perturbation`
    ctrl = cs.obs["is_control"].to_numpy()
    pert = cs.obs["perturbation"].to_numpy()
    assert (pert[ctrl] == CONTROL_TOKEN).all(), (
        "every is_control cell must have perturbation == CONTROL_TOKEN"
    )
    assert not np.isnan(cs.X).any(), "X contains NaN"
