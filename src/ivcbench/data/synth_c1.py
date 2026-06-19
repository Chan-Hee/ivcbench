"""Synthetic C1 (cytokine-response)-shaped fixture.

Mimics the structure of the C1 anchors (Kang IFN-β PBMC · Cano-Gamez CD4 naive/memory ·
Oesinghaus 90-cytokine breadth): coarse lineages with fine sub-lineages, a cytokine perturbation
axis (with a strong Type-I IFN program induced by IFN-β), donors, and naive/memory state encoded in
the fine label. Lets the full C1 cycle (resolution LOCT · state transfer · LOcyt) run GPU-free
before Kang/Cano-Gamez/Oesinghaus data lands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CONTROL_TOKEN, CellSet

# coarse lineage -> fine sub-lineages (state encoded in the fine label, e.g. CD4 naive/memory)
LINEAGES = {
    "CD4T": ["CD4_naive", "CD4_memory"],
    "CD8T": ["CD8_naive", "CD8_memory"],
    "B": ["B_naive", "B_memory"],
    "NK": ["NK_dim", "NK_bright"],
    "Mono": ["Mono_class", "Mono_nonclass"],
}
CYTOKINES = ["IFNb", "IFNg", "IL2", "IL4", "IL6", "IL10", "IL17", "TNF", "IL21", "IL12", "TGFb"]
DONORS = ["d1", "d2", "d3", "d4"]


def make_c1_like(n_genes: int = 200, cells_per_group: int = 4, seed: int = 0) -> CellSet:
    rng = np.random.default_rng(seed)
    perts = [CONTROL_TOKEN] + CYTOKINES  # control == PBS

    # Type-I IFN program: first 20 genes, strongly induced by IFN-β (partially by IFN-γ)
    program_idx = np.arange(20)
    shared_cyt_axis = rng.normal(0, 0.6, n_genes)  # common stimulation response
    cyt_effect = {}
    for c in CYTOKINES:
        eff = 0.5 * shared_cyt_axis + 0.9 * rng.normal(0, 1.0, n_genes)
        if c == "IFNb":
            eff[program_idx] += 3.0
        if c == "IFNg":
            eff[program_idx] += 1.2
        cyt_effect[c] = eff

    fine_mod = {f: rng.normal(1.0, 0.30, n_genes) for ls in LINEAGES.values() for f in ls}
    fine_base = {f: rng.normal(0, 1.0, n_genes) for ls in LINEAGES.values() for f in ls}
    donor_eff = {d: rng.normal(0, 0.25, n_genes) for d in DONORS}

    rows, X = [], []
    for coarse, fines in LINEAGES.items():
        for fine in fines:
            for d in DONORS:
                for p in perts:
                    mu = fine_base[fine] + donor_eff[d]
                    if p != CONTROL_TOKEN:
                        mu = mu + cyt_effect[p] * fine_mod[fine]
                    X.append(rng.normal(mu, 0.5, (cells_per_group, n_genes)).astype(np.float32))
                    rows += [dict(
                        cell_type_coarse=coarse, cell_type_fine=fine, perturbation=p,
                        condition="6h", donor_id=d, timepoint="6h", batch=f"b_{d}",
                        is_control=(p == CONTROL_TOKEN),
                    )] * cells_per_group
    X = np.vstack(X).astype(np.float32)
    obs = pd.DataFrame(rows)
    obs["is_control"] = obs["is_control"].astype(bool)
    var_names = [f"g{i:03d}" for i in range(n_genes)]
    return CellSet(
        X=X, obs=obs, var_names=var_names,
        side_info={"cytokines": CYTOKINES},
        uns={
            "dataset": "synthetic_c1_like",
            "modality": "rna",
            "immune_program": {"type_i_ifn": [var_names[i] for i in program_idx]},
            "cytokines": CYTOKINES,
        },
    )
