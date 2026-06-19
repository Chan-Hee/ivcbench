"""Synthetic OP3 (C5)-shaped fixture.

Generates a CellSet with the same *structure* as Szałata/OP3 — PBMC lineages x donors x compounds
(+ DMSO control) x genes — with compound- and cell-type-specific expression shifts and a chemistry
side-info block (Morgan-like fingerprints where similar compounds share bits). Used to validate the
end-to-end pipeline (split -> leak audit -> baselines -> metrics) with zero GPU and zero real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import CONTROL_TOKEN, CellSet

CELL_TYPES = ["CD4T", "CD8T", "B", "NK", "Mono"]
DONORS = ["d1", "d2", "d3"]


def make_op3_like(
    n_compounds: int = 15,
    n_genes: int = 200,
    cells_per_group: int = 6,
    fp_bits: int = 64,
    seed: int = 0,
) -> CellSet:
    rng = np.random.default_rng(seed)
    compounds = [f"cpd{i:02d}" for i in range(1, n_compounds + 1)]
    perts = [CONTROL_TOKEN] + compounds  # DMSO == control

    # --- chemistry side-info: latent scaffold -> fingerprint, so similar compounds share bits ---
    n_scaffold = 4
    scaffold_proto = (rng.random((n_scaffold, fp_bits)) < 0.3).astype(np.float32)
    cpd_scaffold = {c: rng.integers(0, n_scaffold) for c in compounds}
    fingerprint = {}
    for c in compounds:
        base = scaffold_proto[cpd_scaffold[c]].copy()
        flip = rng.random(fp_bits) < 0.05  # small per-compound perturbation of the scaffold
        base[flip] = 1 - base[flip]
        fingerprint[c] = base

    # --- ground-truth effects: per-compound gene shift, modulated by cell type ---
    # each compound = a shared immunomodulatory axis (captured by mean-shift baselines) + a
    # compound-specific component (only chemistry-aware models can target it).
    shared_axis = rng.normal(0, 1.0, n_genes)
    cpd_effect = {c: 0.6 * shared_axis + 0.8 * rng.normal(0, 1.0, n_genes) for c in compounds}
    ct_modulation = {ct: rng.normal(1.0, 0.25, n_genes) for ct in CELL_TYPES}  # multiplicative
    ct_baseline = {ct: rng.normal(0, 1.0, n_genes) for ct in CELL_TYPES}
    donor_effect = {d: rng.normal(0, 0.3, n_genes) for d in DONORS}

    rows, X = [], []
    for ct in CELL_TYPES:
        for d in DONORS:
            for p in perts:
                mu = ct_baseline[ct] + donor_effect[d]
                if p != CONTROL_TOKEN:
                    mu = mu + cpd_effect[p] * ct_modulation[ct]
                cells = rng.normal(mu, 0.5, size=(cells_per_group, n_genes)).astype(np.float32)
                X.append(cells)
                for _ in range(cells_per_group):
                    rows.append(
                        dict(
                            cell_type_coarse=ct,
                            cell_type_fine=ct,
                            perturbation=p,
                            condition="24h",
                            donor_id=d,
                            timepoint="24h",
                            batch=f"plate_{d}",
                            is_control=(p == CONTROL_TOKEN),
                        )
                    )
    X = np.vstack(X).astype(np.float32)
    obs = pd.DataFrame(rows)
    obs["is_control"] = obs["is_control"].astype(bool)
    var_names = [f"g{i:03d}" for i in range(n_genes)]

    # an "immunomod MOA"-like program: genes most moved by an arbitrary reference compound
    ref = cpd_effect[compounds[0]]
    program_idx = np.argsort(-np.abs(ref))[:20]
    return CellSet(
        X=X,
        obs=obs,
        var_names=var_names,
        side_info={"fingerprint": fingerprint, "scaffold": cpd_scaffold},
        uns={
            "dataset": "synthetic_op3_like",
            "modality": "rna",
            "immune_program": {"immunomod_moa": [var_names[i] for i in program_idx]},
            "compounds": compounds,
        },
    )
