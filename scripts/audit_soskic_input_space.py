#!/usr/bin/env python3
"""Inspect the supplied Soskic matrices without inventing unprocessed counts.

The source paper describes group-wise HVG selection, covariate regression,
scaling and clipping at 10. Negative entries and the stored cap are checked in
the actual inputs. An extra fold-local standardization cannot undo these steps.
"""
from pathlib import Path
import hashlib
import json

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results/_paper"
FILES = [
    "restingCells_CD4only_HVGs_processed.h5ad",
    "stimulatedCells_highlyActiveCD4_16h_HVGs_processed.h5ad",
]


def main():
    summaries, genes = [], []
    for condition, name in zip(["0h", "16h"], FILES):
        path = ROOT / "data/C2/soskic" / name
        with h5py.File(path, "r") as f:
            x = f["X"]
            n, p = x.shape
            names = f["var"]["index"].astype(str)
            sums, squares, negative, cap = [np.zeros(p, dtype=float) for _ in range(4)]
            minimum, maximum = np.full(p, np.inf), np.full(p, -np.inf)
            for start in range(0, n, 4096):
                block = np.asarray(x[start : start + 4096], float)
                sums += block.sum(0)
                squares += (block * block).sum(0)
                negative += (block < 0).sum(0)
                cap += (block == 10).sum(0)
                minimum = np.minimum(minimum, block.min(0))
                maximum = np.maximum(maximum, block.max(0))
            means = sums / n
            sd = np.sqrt(np.maximum(squares / n - means * means, 0))
            summaries.append(
                dict(
                    condition=condition,
                    input_file=name,
                    n_cells=n,
                    n_genes=p,
                    raw_or_count_layer_present=bool("raw" in f or "layers" in f),
                    median_gene_mean=float(np.median(means)),
                    median_gene_sd=float(np.median(sd)),
                    minimum=float(minimum.min()),
                    maximum=float(maximum.max()),
                    negative_fraction=float(negative.sum() / (n * p)),
                    fraction_at_upper_cap=float(cap.sum() / (n * p)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
            genes.extend(
                dict(
                    condition=condition,
                    gene=g,
                    mean=float(mu),
                    sd=float(s),
                    minimum=float(lo),
                    maximum=float(hi),
                    negative_fraction=float(ne / n),
                    fraction_at_ten=float(c / n),
                )
                for g, mu, s, lo, hi, ne, c in zip(
                    names, means, sd, minimum, maximum, negative, cap
                )
            )
    pd.DataFrame(genes).to_csv(PAPER / "soskic_input_gene_statistics.csv", index=False)
    pd.DataFrame(summaries).to_csv(PAPER / "soskic_input_space.csv", index=False)
    report = dict(
        inputs=summaries,
        source_methods="https://www.nature.com/articles/s41588-022-01066-3",
        source_section=(
            "Exploratory data analysis and removal of cellular contaminations;"
            " Identification of a lowly active T cell subpopulation"
        ),
        interpretation=(
            "Supplied condition-group-processed, covariate-regressed/scaled HVG"
            " coordinates; not a common unregressed log-count space."
        ),
        permitted_claim=(
            "Conditional donor transfer in the supplied coordinates. Fold-local"
            " re-standardization checks only the benchmark-added affine step."
        ),
        excluded_claim=(
            "Strictly inductive from raw cells, absolute activation magnitude, or"
            " biological enrichment inferred from pooled T2 coordinates."
        ),
        raw_access=(
            "Source study raw scRNA-seq: EGA EGAD00001008197; genotype access is a"
            " different dataset."
        ),
    )
    (PAPER / "soskic_input_space_provenance.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(pd.DataFrame(summaries).drop(columns="sha256").to_string(index=False))


if __name__ == "__main__":
    main()
