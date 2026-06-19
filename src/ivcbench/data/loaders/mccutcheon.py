"""McCutcheon 2023 (C3) loader — GSE218985, primary human T-cell CRISPRi & CRISPRa Perturb-seq.

Per-sample 10x with guides embedded as 'CRISPR Guide Capture' features (e.g. 'BATF3_2'); each cell's
target gene is the argmax guide. CRISPRi and CRISPRa are loaded separately (modality-stratified).
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..crispr import assemble, read_10x_mtx, split_guide_matrix
from ..preprocess import PreprocessConfig
from ..schema import CONTROL_TOKEN

# (RAW member, donor) for each modality
SAMPLES = {
    "CRISPRi": [("GSM6761464_CRISPRi_D1", "D1"), ("GSM6761465_CRISPRi_D2", "D2"),
                ("GSM6761466_CRISPRi_D3", "D3")],
    "CRISPRa": [("GSM6761467_CRISPRa_D1", "D1"), ("GSM6761468_CRISPRa_D2", "D2"),
                ("GSM6761469_CRISPRa_D3", "D3")],
}


def load(path: str | Path = "data/C3/mccutcheon", modality: str = "CRISPRi",
         cfg: PreprocessConfig = PreprocessConfig(), subsample_per_gene: int = 300, seed: int = 0):
    root = Path(path)
    ex = root / "extracted"
    if not ex.exists():
        with tarfile.open(root / "GSE218985_RAW.tar") as t:
            t.extractall(ex)
    rng = np.random.default_rng(seed)

    blocks = []
    for member, donor in SAMPLES[modality]:
        sd = ex / member
        if not sd.exists():
            with tarfile.open(ex / f"{member}.tar.gz") as t:
                t.extractall(sd)
        counts, barcodes, feats, types = read_10x_mtx(sd)
        expr, genes, per_gene, per_ctrl = split_guide_matrix(counts, feats, types)
        pert = np.array([CONTROL_TOKEN if c else g for g, c in zip(per_gene, per_ctrl)], dtype=object)
        keep = np.array([p is not None for p in pert])
        # cap per perturbation within the sample
        pos = np.where(keep)[0]
        sel = []
        for lab in pd.unique(pert[pos]):
            idx = pos[pert[pos] == lab]
            sel.append(idx if len(idx) <= subsample_per_gene
                       else rng.choice(idx, subsample_per_gene, replace=False))
        sel = np.sort(np.concatenate(sel)) if sel else np.array([], dtype=int)
        obs = pd.DataFrame({
            "cell_type_coarse": "CD8T", "cell_type_fine": "CD8T", "perturbation": pert[sel],
            "condition": modality, "donor_id": donor, "timepoint": "NA",
            "batch": member, "is_control": per_ctrl[sel],
        })
        blocks.append(dict(counts=expr[sel], genes=genes, obs=obs))

    cs = assemble(blocks, dataset=f"mccutcheon_{modality}_GSE218985", cfg=cfg,
                  uns={"accession": "GSE218985", "modality_label": modality})
    cs.uns["genes_perturbed"] = sorted(set(cs.obs["perturbation"]) - {CONTROL_TOKEN})
    return cs
