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

recomputes per-(cluster, model, split) Pearson-Delta and energy distance with the SAME frozen metric code used
for the paper (ivcbench.metrics). Compare against the deposited results_raw.csv to confirm every headline number.
"""
from __future__ import annotations
import argparse, csv, glob, sys

# the writer/reader live in the installed package so the runner and every model script share one format
from ivcbench.eval.bundle import save_bundle, score_bundle  # noqa: F401  (save_bundle re-exported for tests)


def main(argv=None):
    ap = argparse.ArgumentParser(description="GPU-free predictions -> metrics reproduction")
    ap.add_argument("bundles", nargs="+", help="prediction .npz files (globs ok)")
    ap.add_argument("-o", "--out", default=None, help="write reproduced rows to CSV")
    a = ap.parse_args(argv)
    files = sorted({f for g in a.bundles for f in glob.glob(g, recursive=True)} or a.bundles)
    rows = [score_bundle(f) for f in files]
    cols = ["cluster", "model", "split", "n_test_strata", "pearson_delta", "e_distance"]
    w = csv.DictWriter(a.out and open(a.out, "w", newline="") or sys.stdout, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    if a.out:
        print(f"wrote {len(rows)} reproduced rows -> {a.out}", file=sys.stderr)
    return rows


if __name__ == "__main__":
    main()
