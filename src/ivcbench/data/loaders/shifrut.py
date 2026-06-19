"""Shifrut 2018 (C3) loader — GSE119450, primary human T-cell CRISPR-KO Perturb-seq.

Per-sample 10x matrices + CellBC_sgRNA.csv (cols: Cell.BC, gRNA.ID='ES.sg<N>.<GENE>', control =
'CTRL<NNN>'). Uses the shared CRISPR ingestion + unified preprocessing.
"""
from __future__ import annotations

import re
import tarfile
from pathlib import Path

import pandas as pd

from ..crispr import assemble, guide_map_to_obs, read_10x_mtx
from ..preprocess import PreprocessConfig
from ..schema import CONTROL_TOKEN

# (matrix tar.gz, guide csv.gz, donor, condition)
SAMPLES = [
    ("GSM3375483_D1N_matrix.tar.gz", "GSM3375487_D1N_CellBC_sgRNA.csv.gz", "D1", "NoStim"),
    ("GSM3375484_D1S_matrix.tar.gz", "GSM3375488_D1S_CellBC_sgRNA.csv.gz", "D1", "Stim"),
    ("GSM3375485_D2N_matrix.tar.gz", "GSM3375489_D2N_CellBC_sgRNA.csv.gz", "D2", "NoStim"),
    ("GSM3375486_D2S_matrix.tar.gz", "GSM3375490_D2S_CellBC_sgRNA.csv.gz", "D2", "Stim"),
]


def _strip(bc: str) -> str:
    return re.sub(r"-\d+$", "", str(bc))


def load(path: str | Path = "data/C3/shifrut", cfg: PreprocessConfig = PreprocessConfig()):
    root = Path(path)
    ex = root / "extracted"
    if not ex.exists():
        with tarfile.open(root / "GSE119450_RAW.tar") as t:
            t.extractall(ex)

    blocks = []
    for mtar, gcsv, donor, cond in SAMPLES:
        sd = ex / mtar.replace(".tar.gz", "")
        if not sd.exists():
            with tarfile.open(ex / mtar) as t:
                t.extractall(sd)
        counts, barcodes, genes, _ = read_10x_mtx(sd)
        barcodes = [_strip(b) for b in barcodes]
        gdf = pd.read_csv(ex / gcsv)
        gdf["Cell.BC"] = gdf["Cell.BC"].map(_strip)
        obs = guide_map_to_obs(barcodes, gdf, "Cell.BC", "gRNA.ID", cell_type="CD8T",
                               donor=donor, condition=cond, batch=f"{donor}_{cond}")
        keep = (obs["perturbation"] != "unassigned").to_numpy()
        blocks.append(dict(counts=counts[keep], genes=genes,
                           obs=obs[keep].reset_index(drop=True)))

    cs = assemble(blocks, dataset="shifrut_GSE119450", cfg=cfg, uns={"accession": "GSE119450"})
    cs.uns["genes_perturbed"] = sorted(set(cs.obs["perturbation"]) - {CONTROL_TOKEN})
    return cs
