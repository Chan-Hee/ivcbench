#!/usr/bin/env python3
"""Assemble assay qualifications from existing checkpoint results; no fitting.

The protein fits are separate from RNA fits. Sign matching of a constant
training-mean protein shift does not establish knockout-specific prediction,
receptor biology, assay sensitivity, or a post-transcriptional mechanism.
"""
from __future__ import annotations

import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from ivcbench.data.loaders.frangieh import _read_h5_dataframe

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results/_paper"


def raw_assay_context():
    rows = []
    for modality in ("protein", "RNA"):
        path = ROOT / "data/C4/frangieh" / f"FrangiehIzar2021_{modality}.h5ad"
        with h5py.File(path) as data:
            obs, var = _read_h5_dataframe(data["obs"]), _read_h5_dataframe(data["var"])
            mask = (obs.perturbation_2.astype(str) == "IFNγ").to_numpy()
            names = var.index.astype(str).tolist()
            targets = (
                [
                    "CD279",
                    "CD274",
                    "Rat_IgG2a",
                    "Mouse_IgG1",
                    "Mouse_IgG2a",
                    "Mouse_IgG2b",
                ]
                if modality == "protein"
                else ["PDCD1", "CD274"]
            )
            for name in targets:
                record = dict(
                    dataset="Frangieh",
                    modality=modality,
                    condition="IFNγ",
                    marker=name,
                    n_cells=int(mask.sum()),
                    present=name in names,
                    source=str(path.relative_to(ROOT)),
                )
                if name in names:
                    matrix = data["X"]
                    if matrix.attrs["encoding-type"] != "csc_matrix":
                        raise ValueError("Expected CSC matrix for column-wise audit")
                    j = names.index(name)
                    start, stop = matrix["indptr"][j : j + 2]
                    vector = np.zeros(len(obs))
                    vector[matrix["indices"][start:stop]] = matrix["data"][start:stop]
                    vector = vector[mask]
                    record.update(
                        mean=float(vector.mean()),
                        median=float(np.median(vector)),
                        p95=float(np.quantile(vector, 0.95)),
                        positive_fraction=float(np.mean(vector > 0)),
                    )
                rows.append(record)
    return pd.DataFrame(rows)


def main():
    context = raw_assay_context()
    context.to_csv(PAPER / "frangieh_raw_checkpoint_context.csv", index=False)
    chen = json.loads(
        (ROOT / "results/newdata/chen_checkpoint_replication_summary.json").read_text()
    )
    records = []
    for frac in (25, 50):
        for norm, container in (
            ("library-log", chen),
            ("centered log1p", chen["robustness_CLR"]),
        ):
            for marker in ("PD-1_CD279", "PD-L1_CD274"):
                row = container[f"frac{frac}"][marker]
                records.append(
                    dict(
                        dataset="Chen",
                        held_fraction=frac,
                        normalization=norm,
                        marker=marker.split("_")[0],
                        mean=row["obs_mean"],
                        ci_lo=row["ci"][0],
                        ci_hi=row["ci"][1],
                        sign_match=row["sign_match_frac"],
                        n_held=chen[f"frac{frac}"][marker]["n_held_KO"],
                        interval_scope=(
                            "historical normal interval over KOs; shared control fixed;"
                            " unadjusted"
                        ),
                    )
                )
    pd.DataFrame(records).to_csv(
        PAPER / "chen_checkpoint_normalization.csv", index=False
    )
    (PAPER / "checkpoint_readout_provenance.json").write_text(
        json.dumps(
            dict(
                frangieh=(
                    "IFNγ melanoma cells; RNA/protein fitted independently, not"
                    " cross-modal prediction"
                ),
                chen=(
                    "surface panel from FOXP3-regulator screen; separate dataset and"
                    " processing"
                ),
                normalization=(
                    "centered log1p = log1p(count) minus the across-marker cell mean;"
                    " not claimed to be a universal ADT standard"
                ),
                statistical_scope=(
                    "intervals conditional on shared controls; overlapping holdouts not"
                    " independent replications"
                ),
                interpretation=(
                    "neither interval crossing zero nor constant-shift sign matching"
                    " establishes an assay detection limit or post-transcriptional"
                    " mechanism"
                ),
                raw_rna_note=(
                    "PDCD1 absent from the supplied RNA feature list, not direct"
                    " evidence of biological absence"
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print(context.to_string(index=False))
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
