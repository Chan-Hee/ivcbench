#!/usr/bin/env python3
"""Independent-control repeatability for the exact census targets and masks.

No model is fitted. Each loader/split must reproduce the stored observations,
controls, genes and strata before resampling. Treated AND control cells are
partitioned into disjoint halves. Shared-control correlations are computed on
the same treated halves solely as a paired diagnostic. Neither statistic is a
prediction ceiling, a biological-replicate estimate, or an attenuation correction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from assemble_cross_cluster import ROOT
from ivcbench.clusters import c1, c3, c4, c5
from ivcbench.splits.builder import build_split
from ivcbench.splits.spec import SplitSpec

PAPER = Path(ROOT) / "results/_paper"


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else np.nan


def disjoint_halves(n, rng):
    if n < 6:
        raise ValueError("At least six cells are needed for disjoint half-samples")
    order = rng.permutation(n)
    h = n // 2
    a, b = order[:h], order[h : 2 * h]
    assert not np.intersect1d(a, b).size
    return a, b


def held_label(stratum):
    """Decode a stored perturbation stratum without changing compound names."""
    value = str(stratum).split("|", 1)[0]
    if not value.startswith("perturbation="):
        raise ValueError(f"Expected named perturbation stratum: {stratum}")
    return value.removeprefix("perturbation=")


def evaluate_split(cs, spec, row, partitions):
    split = build_split(cs, spec)
    path = Path(ROOT) / row.bundle_path
    with np.load(path, allow_pickle=True) as data:
        genes, strata = data["genes"].astype(str), data["strata"].astype(str)
        if not np.array_equal(genes, np.asarray(cs.var_names, str)):
            raise ValueError(f"{row.task_key}/{row.unit}: different feature panel")
        if set(strata) != set(split.test_strata.astype(str)):
            raise ValueError(f"{row.task_key}/{row.unit}: different held strata")
        treated, controls = cs.X[split.test_idx], cs.X[split.inference_input_idx]
        observed = np.vstack([treated[split.test_strata == s].mean(0) for s in strata])
        obs_error = float(np.max(np.abs(observed - data["obs_means"])))
        ctrl_error = float(np.max(np.abs(controls.mean(0) - data["control_mean"])))
        if max(obs_error, ctrl_error) > 1e-4:
            raise ValueError(
                f"{row.task_key}/{row.unit}: target mismatch ({obs_error},"
                f" {ctrl_error})"
            )
        keep = np.ones(len(genes), bool)
        if "exclude_gene_idx" in data.files:
            keep[np.asarray(data["exclude_gene_idx"], int)] = False
    seed = int(
        hashlib.sha256(
            f"{row.task_key}/{row.unit}/independent-control-v1".encode()
        ).hexdigest()[:8],
        16,
    )
    rng = np.random.default_rng(seed)
    ctrl = np.asarray(controls[:, keep], float)
    full_control = ctrl.mean(0)
    control_halves = []
    for _ in range(partitions):
        a, b = disjoint_halves(len(ctrl), rng)
        control_halves.append((ctrl[a].mean(0), ctrl[b].mean(0)))
    results, draws = [], []
    for s in strata:
        values = np.asarray(treated[split.test_strata == s][:, keep], float)
        unit = s.split("|")[0] if row.task_key == "T5u" else str(row.unit)
        record = dict(
            task_key=row.task_key,
            unit=unit,
            stratum=s,
            n_treated=len(values),
            n_controls=len(ctrl),
            n_scored_genes=int(keep.sum()),
            partitions=partitions,
            seed=seed,
            bundle_path=row.bundle_path,
        )
        independent, shared = [], []
        if len(values) >= 6:
            for p, (ca, cb) in enumerate(control_halves):
                a, b = disjoint_halves(len(values), rng)
                ta, tb = values[a].mean(0), values[b].mean(0)
                ri = correlation(ta - ca, tb - cb)
                rs = correlation(ta - full_control, tb - full_control)
                independent.append(ri)
                shared.append(rs)
                draws.append(
                    dict(
                        task_key=row.task_key,
                        unit=unit,
                        stratum=s,
                        partition=p,
                        independent_control_r=ri,
                        shared_control_r=rs,
                    )
                )
        has_estimate = bool(np.isfinite(independent).any())
        record.update(
            independent_control_r=(
                float(np.nanmean(independent)) if has_estimate else np.nan
            ),
            shared_control_r=(
                float(np.nanmean(shared)) if np.isfinite(shared).any() else np.nan
            ),
            status=(
                "estimable"
                if has_estimate
                else (
                    "fewer than six treated cells"
                    if len(values) < 6
                    else "constant half-sample response"
                )
            ),
        )
        results.append(record)
    provenance = dict(
        task_key=row.task_key,
        unit=str(row.unit),
        bundle=row.bundle_path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        observed_max_difference=obs_error,
        control_max_difference=ctrl_error,
        feature_mask_sha256=hashlib.sha256(keep.tobytes()).hexdigest(),
        n_strata=len(strata),
        partitions=partitions,
        independent_control_halves=True,
    )
    return results, draws, provenance


def main():
    # Raw-cell dependencies are optional; summary replay and unit tests need none.
    from ivcbench.data.loaders import (
        kang,
        soskic,
        shifrut,
        schmidt,
        mccutcheon,
        chen,
        frangieh,
        op3,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "kang",
            "soskic",
            "shifrut",
            "schmidt",
            "mccutcheon_CRISPRi",
            "mccutcheon_CRISPRa",
            "chen",
            "frangieh",
            "op3",
        ],
    )
    parser.add_argument("--partitions", type=int, default=100)
    args = parser.parse_args()
    frame = pd.read_csv(PAPER / "census_unit_scores.csv", dtype={"unit": str})
    refs = frame[frame.model == "cell-mean"].drop_duplicates(
        ["task_key", "bundle_path"]
    )
    loaders = dict(
        kang=kang.load,
        soskic=soskic.load,
        shifrut=shifrut.load,
        schmidt=schmidt.load,
        chen=chen.load,
        frangieh=frangieh.load,
        mccutcheon_CRISPRi=lambda: mccutcheon.load(modality="CRISPRi"),
        mccutcheon_CRISPRa=lambda: mccutcheon.load(modality="CRISPRa"),
        op3=op3.load,
    )
    task_of = dict(kang="T1", soskic="T2", frangieh="T4", op3="T5")
    for dataset in args.datasets:
        started = time.perf_counter()
        print(f"Reading {dataset}; no model fitting", flush=True)
        cs = loaders[dataset]()
        task = task_of.get(dataset, "T3")
        rows = refs[refs.task_key.str.startswith(task)]
        if task == "T3":
            rows = rows[rows.unit == dataset]
        records, partitions, provenance = [], [], []
        for row in rows.itertuples():
            if task == "T1":
                lineage = next(
                    x
                    for x in cs.obs.cell_type_coarse.unique()
                    if str(x).replace(" ", "_") == row.unit
                )
                spec = c1.coarse_loct(lineage)
            elif task == "T2":
                spec = SplitSpec(
                    name=f"repeatability_{row.unit}",
                    cluster="C2",
                    key_col="donor_id",
                    held_values=[row.unit],
                    control_inference_only=True,
                    strata_cols=["cell_type_coarse"],
                    registry_task="C2",
                )
            elif task == "T3":
                spec = c3.true_lo_gene(
                    c3.held_gene_fraction(cs.uns["genes_perturbed"], 0.1, seed=0), "10"
                )
            elif task == "T4":
                with np.load(Path(ROOT) / row.bundle_path, allow_pickle=True) as b:
                    held = [held_label(s) for s in b["strata"].astype(str)]
                spec = c4.modality_lo_ko(held, row.unit)
            elif row.task_key == "T5c":
                lineage = next(
                    x
                    for x in cs.obs.cell_type_coarse.unique()
                    if str(x).replace(" ", "_") == row.unit
                )
                spec = c5.cross_celltype_loct(lineage)
            else:
                held = [
                    held_label(s)
                    for s in frame[
                        (frame.task_key == "T5u") & (frame.model == "cell-mean")
                    ].unit
                ]
                spec = c5.global_compound_holdout(held)
            values, draws, source = evaluate_split(cs, spec, row, args.partitions)
            records.extend(values)
            partitions.extend(draws)
            provenance.append(source)
        out = PAPER / "target_repeatability"
        out.mkdir(exist_ok=True)
        pd.DataFrame(records).to_csv(out / f"{dataset}.csv", index=False)
        pd.DataFrame(partitions).to_csv(out / f"{dataset}_partitions.csv", index=False)
        (out / f"{dataset}.json").write_text(
            json.dumps(
                dict(
                    dataset=dataset,
                    inputs=provenance,
                    elapsed_seconds=time.perf_counter() - started,
                    scope=(
                        "cell-sampling repeatability on exact held targets, not"
                        " biological replication or an accuracy bound"
                    ),
                ),
                indent=2,
            )
            + "\n"
        )
        print(
            f"{dataset}: {len(records)} target strata;"
            f" {time.perf_counter()-started:.1f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
