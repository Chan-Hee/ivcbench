"""Frangieh 2021 (C4 Axis-2) loader — RNA vs targeted-protein (CITE) modality generalization.

scPerturb-standardized h5ad pair (Zenodo 13350497): melanoma + TIL co-culture CRISPR-KO screen with
matched RNA and 24-marker CITE-seq readouts on the SAME 218,331 cells (249 perturbations = 248 KO genes
+ control; 3 conditions: Control / IFNγ / Co-culture). We expose either modality as a CellSet so the
identical leave-one-KO-gene-out split + simple baselines run in RNA space and in protein space — the
modality axis (does an unseen-KO response recover equally well in transcriptome vs surface proteome?).

To isolate the KO effect from the stimulation confound we restrict to one condition (default IFNγ — the
canonical Frangieh perturbation context with the strongest surface-marker modulation). Both modalities
use the IDENTICAL preprocessing (library-log-norm) so the RNA-vs-protein comparison is apples-to-apples;
the 4 isotype-control antibodies are dropped from the protein panel (20 informative markers kept).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..preprocess import PreprocessConfig, preprocess
from ..schema import CONTROL_TOKEN

_ISOTYPE = {"Rat_IgG2a", "Mouse_IgG1", "Mouse_IgG2a", "Mouse_IgG2b"}   # protein isotype controls → drop
_DIR = "data/C4/frangieh"


def load(modality: str = "rna", condition: str = "IFNγ", subsample_per_group: int = 60,
         n_hvg: int = 2000, seed: int = 0) -> CellSet:  # noqa: F821
    import anndata
    base = Path(os.environ.get("IVCBENCH_FRANGIEH_DIR", _DIR))
    fname = "FrangiehIzar2021_RNA.h5ad" if modality == "rna" else "FrangiehIzar2021_protein.h5ad"
    a = anndata.read_h5ad(str(base / fname), backed="r")
    obs = a.obs

    keep_cond = obs["perturbation_2"].astype(str) == condition
    pert_all = obs["perturbation"].astype(str)
    # subsample per perturbation within the chosen condition (cap cells/KO)
    rng = np.random.default_rng(seed)
    pos = np.where(keep_cond.to_numpy())[0]
    sel = []
    for _, g in pd.Series(pos).groupby(pert_all.to_numpy()[pos]):
        v = g.to_numpy()
        sel.append(v if len(v) <= subsample_per_group else rng.choice(v, subsample_per_group, replace=False))
    idx = np.sort(np.concatenate(sel))

    sub = a[idx].to_memory()
    counts = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    var = list(map(str, sub.var_names))
    if modality != "rna":                                 # drop isotype-control antibodies
        keep_v = [i for i, v in enumerate(var) if v not in _ISOTYPE]
        counts = counts[:, keep_v]; var = [var[i] for i in keep_v]

    pert = sub.obs["perturbation"].astype(str).to_numpy()
    is_ctrl = pert == "control"
    pert = np.where(is_ctrl, CONTROL_TOKEN, pert)
    obs_out = pd.DataFrame({
        "cell_type_coarse": "melanoma", "cell_type_fine": "melanoma", "perturbation": pert,
        "condition": condition, "donor_id": "frangieh", "timepoint": "co-culture",
        "batch": condition, "is_control": is_ctrl,
    })
    # RNA: standard QC + HVG. Protein (20-marker CITE): the RNA gene-count QC (min 200 genes/cell)
    # would wipe a 20-feature panel, so relax it and keep all markers.
    cfg = (PreprocessConfig(n_hvg=n_hvg) if modality == "rna"
           else PreprocessConfig(n_hvg=64, min_genes_per_cell=1, min_cells_per_gene=0))
    cs = preprocess(counts, var, obs_out, side_info={},
                    uns={"dataset": f"frangieh_{modality}", "accession": "Zenodo-13350497",
                         "modality": ("RNA" if modality == "rna" else "protein-CITE"),
                         "condition": condition, "n_cells_total": int(a.n_obs),
                         "genes_perturbed": sorted(set(pert) - {CONTROL_TOKEN})},
                    cfg=cfg)
    cs.uns["genes_perturbed"] = sorted(set(cs.obs["perturbation"]) - {CONTROL_TOKEN})
    return cs


if __name__ == "__main__":
    for m in ["rna", "protein"]:
        cs = load(modality=m)
        print(m, "cells", cs.n_cells, "feat", cs.n_genes, "perts", len(cs.uns["genes_perturbed"]),
              "ctrl_frac", round(float(cs.obs["is_control"].mean()), 3))
