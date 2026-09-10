#!/usr/bin/env python3
"""Cache observed rank-program readouts for the exact T3/T5c census targets.

CPU-only reanalysis of existing cells; this does not fit or invoke any model.
Each loader/split must reproduce the genes, labels, control mean and observed
expression means in the selected floor bundle before an output can be written.
The cache contains aggregate program scores, not individual-cell expression.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from assemble_cross_cluster import ROOT
from ivcbench.clusters import c3, c5
from ivcbench.data.loaders import chen, mccutcheon, op3, schmidt, shifrut
from ivcbench.metrics.program import aucell
from ivcbench.splits.builder import build_split

PAPER = Path(ROOT) / "results/_paper"
OUT = PAPER / "program_observations"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell_scores(X, genes):
    return np.concatenate(
        [aucell(X[start : start + 2048], genes) for start in range(0, len(X), 2048)]
    )


def build_one(cs, spec, programs, row):
    split = build_split(cs, spec)
    path = Path(ROOT) / row.bundle_path
    with np.load(path, allow_pickle=True) as data:
        genes = data["genes"].astype(str)
        strata = data["strata"].astype(str)
        if not np.array_equal(genes, np.asarray(cs.var_names, str)):
            raise ValueError(f"{row.unit}: loader genes differ from the census")
        if set(strata) != set(split.test_strata.astype(str)):
            raise ValueError(f"{row.unit}: loader held strata differ from the census")
        test = cs.X[split.test_idx]
        observed = np.vstack([test[split.test_strata == s].mean(0) for s in strata])
        observed_diff = float(np.abs(observed - data["obs_means"]).max())
        controls = split.inference_input_idx
        if not len(controls):
            controls = split.train_idx[
                cs.obs.iloc[split.train_idx].is_control.to_numpy()
            ]
        control = cs.X[controls]
        control_diff = float(np.abs(control.mean(0) - data["control_mean"]).max())
        if max(observed_diff, control_diff) > 0.0001:
            raise ValueError(
                f"{row.unit}: targets differ: obs={observed_diff}, ctrl={control_diff}"
            )
    names = list(programs)
    means, zeros, ctrl_means, measured, coverage = [], [], [], [], []
    for name, members in programs.items():
        idx = cs.gene_index(members)
        scores = cell_scores(test, idx)
        ctrl_scores = cell_scores(control, idx)
        means.append([scores[split.test_strata == s].mean() for s in strata])
        zeros.append([np.mean(scores[split.test_strata == s] == 0) for s in strata])
        ctrl_means.append(float(ctrl_scores.mean()))
        measured.append(len(idx))
        coverage.append(";".join(genes[idx]))
    record = dict(
        task_key=row.task_key,
        unit=str(row.unit),
        floor_bundle=row.bundle_path,
        floor_sha256=sha(path),
        observed_max_difference=observed_diff,
        control_max_difference=control_diff,
        n_cells=len(test),
        n_controls=len(control),
        n_strata=len(strata),
        n_genes=len(genes),
        method=(
            "mean per-cell AUCell-like score, top 5%; original argsort tie convention"
        ),
        targets="exact census held observations; no model refitting",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{row.task_key}__{row.unit}"
    np.savez_compressed(
        stem.with_suffix(".npz"),
        genes=genes,
        strata=strata,
        programs=np.asarray(names),
        obs_program_means=np.asarray(means).T,
        obs_zero_fraction=np.asarray(zeros).T,
        control_program_means=np.asarray(ctrl_means),
        n_measured=np.asarray(measured),
        measured_genes=np.asarray(coverage),
        n_cells=np.asarray([np.sum(split.test_strata == s) for s in strata]),
    )
    record["cache_sha256"] = sha(stem.with_suffix(".npz"))
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "shifrut",
            "schmidt",
            "mccutcheon_CRISPRi",
            "mccutcheon_CRISPRa",
            "chen",
            "op3",
        ],
    )
    args = parser.parse_args()
    units = pd.read_csv(PAPER / "census_unit_scores.csv")
    loaders = dict(
        shifrut=shifrut.load,
        schmidt=schmidt.load,
        chen=chen.load,
        mccutcheon_CRISPRi=lambda: mccutcheon.load(modality="CRISPRi"),
        mccutcheon_CRISPRa=lambda: mccutcheon.load(modality="CRISPRa"),
        op3=op3.load,
    )
    for dataset in args.datasets:
        print(f"Loading {dataset}: observed-program audit only", flush=True)
        cs = loaders[dataset]()
        task = "T5c" if dataset == "op3" else "T3"
        rows = units[(units.task_key == task) & (units.model == "cell-mean")]
        if task == "T3":
            rows = rows[rows.unit == dataset]
        for row in rows.itertuples():
            if task == "T3":
                held = c3.held_gene_fraction(cs.uns["genes_perturbed"], 0.1, seed=0)
                spec, programs = c3.true_lo_gene(held, "10"), c3.C3_PROGRAMS
            else:
                lineage = next(
                    (
                        x
                        for x in cs.obs.cell_type_coarse.unique()
                        if str(x).replace(" ", "_") == row.unit
                    ),
                    None,
                )
                if lineage is None:
                    raise ValueError(f"Unknown OP3 census lineage: {row.unit}")
                spec, programs = c5.cross_celltype_loct(lineage), c5.C5_PROGRAMS
            build_one(cs, spec, programs, row)


if __name__ == "__main__":
    main()
