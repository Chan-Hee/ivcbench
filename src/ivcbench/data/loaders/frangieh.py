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


def _read_h5ad_robust(path):
    """Open an h5ad, falling back to a raw h5py read + in-memory construct when the installed anndata is
    too old to parse the file's encoding. The cellot env ships anndata 0.7.6, which cannot read the
    scPerturb frangieh's newer obsm/layers encoding; h5py reads the raw HDF5 regardless of encoding
    version, and 0.7.6 can still *construct* an AnnData from in-memory arrays. Returns (adata, backed)."""
    import anndata
    try:
        return anndata.read_h5ad(path, backed="r"), True
    except Exception:
        return _read_h5ad_h5py(path), False


def _read_h5ad_h5py(path):
    """Raw h5py read -> in-memory AnnData. Robust to (a) anndata too old to parse the file encoding and
    (b) backed-sparse-CSC fancy row-indexing quirks in some envs (e.g. ivc-scpram's anndata/scipy)."""
    import anndata
    import h5py
    with h5py.File(path, "r") as f:
        Xg = f["X"]
        shape = tuple(int(x) for x in Xg.attrs["shape"])
        maker = sp.csc_matrix if "csc" in str(Xg.attrs.get("encoding-type", "")) else sp.csr_matrix
        mat = maker((Xg["data"][:], Xg["indices"][:], Xg["indptr"][:]), shape=shape)
        obs = _read_h5_dataframe(f["obs"])
        var = _read_h5_dataframe(f["var"])
    return anndata.AnnData(X=mat.tocsr(), obs=obs, var=var)


def _read_h5_dataframe(g):
    """Reconstruct a DataFrame from an anndata-h5py dataframe group (encoding-version-robust): handles
    categorical columns stored as {codes, categories} groups and plain array columns."""
    import h5py
    idx_key = g.attrs.get("_index", "_index")
    idx_key = idx_key.decode() if isinstance(idx_key, bytes) else idx_key
    order = g.attrs.get("column-order", [k for k in g.keys() if k != idx_key])
    order = [c.decode() if isinstance(c, bytes) else c for c in order]

    def _col(item):
        if isinstance(item, h5py.Group):                       # categorical: {codes, categories}
            cats = [c.decode() if isinstance(c, bytes) else c for c in item["categories"][:]]
            return pd.Categorical.from_codes(item["codes"][:], cats)
        v = item[:]
        return v.astype(str) if v.dtype.kind == "S" else v

    cols = {k: _col(g[k]) for k in order if k in g}
    df = pd.DataFrame(cols)
    if idx_key in g:
        df.index = pd.Index(_col(g[idx_key]))
    return df


def load(modality: str = "rna", condition: str = "IFNγ", subsample_per_group: int = 60,
         n_hvg: int = 2000, seed: int = 0) -> CellSet:  # noqa: F821
    import anndata
    base = Path(os.environ.get("IVCBENCH_FRANGIEH_DIR", _DIR))
    fname = "FrangiehIzar2021_RNA.h5ad" if modality == "rna" else "FrangiehIzar2021_protein.h5ad"
    a, _backed = _read_h5ad_robust(str(base / fname))
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

    try:
        sub = a[idx].to_memory() if _backed else a[idx].copy()
    except Exception:   # backed-CSC fancy row-index quirk in some envs -> re-read fully in memory via h5py
        sub = _read_h5ad_h5py(str(base / fname))[idx].copy()
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
