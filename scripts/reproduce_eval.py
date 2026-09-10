#!/usr/bin/env python3
"""Reproduce the evaluation (predictions -> metrics -> result rows) WITHOUT raw data, model checkpoints, or a GPU.

GPU-free verification entry point for the deposited prediction layer. Each deposited prediction file is a
self-describing .npz (the `ivcbench` prediction-bundle format, one per cluster x model x split):

    pred_cells       float32 [n_test, n_genes]   model-predicted per-cell expression on the test fold
    test_cells       float32 [n_test, n_genes]   observed (ground-truth) per-cell expression
    cell_strata      object  [n_test]            stratum label per test cell (perturbation / lineage / donor ...)
    control_mean     float32 [n_genes]           matched control mean (the Delta baseline)
    genes            object  [n_genes]           gene (HVG) names
    exclude_gene_idx int     [k]   (optional)    leak-safe metric exclusions (on-target gene / response panel)
    pca_components,  float32                      train-cloud PCA-50 basis (optional) -> energy distance is exact
    pca_mean         float32
    # uns metadata:  cluster, model, split

(A compact alternative stores per-stratum pred_means/obs_means instead of per-cell arrays -> Pearson-Δ only.)

    python scripts/reproduce_eval.py 'predictions/**/*.npz' -o reproduced_results.csv

recomputes per-(cluster, model, split) Pearson-Delta with the SAME frozen metric code used for the paper
(ivcbench.metrics); energy distance requires per-cell arrays AND a stored training-fold PCA basis.
The deposited compact mean bundles return NaN for energy distance (Pearson-Delta only).
This low-level command scores the files requested; the final 47-cell census additionally
requires the inclusion manifest and aggregation in scripts/assemble_cross_cluster.py.
scripts/check_consistency.py verifies the census, not arbitrary NPZ globs.
"""
from __future__ import annotations
import argparse, csv, glob, sys
from pathlib import Path

# the writer/reader live in the installed package so the runner and every model script share one format
from ivcbench.eval.bundle import (
    save_bundle,
    score_bundle,
)  # noqa: F401  (save_bundle re-exported for tests)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="GPU-free predictions -> metrics reproduction"
    )
    ap.add_argument(
        "bundles",
        nargs="*",
        help="prediction .npz files (globs ok); not necessarily census inputs",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        help="score only the current census manifest's selected bundle paths",
    )
    ap.add_argument("-o", "--out", default=None, help="write reproduced rows to CSV")
    a = ap.parse_args(argv)
    import os

    if a.manifest and a.bundles:
        ap.error("Choose --manifest or explicit files, not both")
    if a.manifest:
        root = Path(__file__).resolve().parents[1]
        with a.manifest.open() as handle:
            files = sorted(
                {str(root / r["bundle_path"]) for r in csv.DictReader(handle)}
            )
    else:
        if not a.bundles:
            ap.error("Provide --manifest or one or more bundle files")
        files = sorted(
            {f for g in a.bundles for f in glob.glob(g, recursive=True)} or a.bundles
        )
    # predictions/example holds 4 format-demo toy bundles (a copied LOCT fold); they document the
    # .npz layout but are NOT census members. Keep them out of the reproduced rows.
    files = [f for f in files if os.sep + "example" + os.sep not in f]
    rows = [score_bundle(f) for f in files]
    cols = [
        "cluster",
        "model",
        "split",
        "dataset",
        "n_test_strata",
        "pearson_delta",
        "e_distance",
    ]
    w = csv.DictWriter(
        a.out and open(a.out, "w", newline="") or sys.stdout, fieldnames=cols
    )
    w.writeheader()
    for r in rows:
        w.writerow(r)
    if a.out:
        print(f"wrote {len(rows)} reproduced rows -> {a.out}", file=sys.stderr)
    return rows


if __name__ == "__main__":
    main()
