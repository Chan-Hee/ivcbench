#!/usr/bin/env python3
"""Measured CPU replay and explicitly bounded training-time records.

Never turn elapsed process time into measured GPU utilization, invent a
universal minimum VRAM requirement, or sum duplicate/restarted folds as one run.
The archived revision-wide ledger includes excluded runs and partial timings;
it is not a complete cost estimate for retraining the final census.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results/_paper"
RAW = ROOT.parent / "benchmark/outputs/additional_models"
REV = ROOT.parent / "revision_BIB-26-1553/03_etc"
if (ROOT / "results/provenance_inputs/compute").is_dir():
    RAW = ROOT / "results/provenance_inputs/compute"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure-replay", action="store_true")
    args = parser.parse_args()
    if args.measure_replay:
        start = time.perf_counter()
        run = subprocess.run(
            [sys.executable, "scripts/check_consistency.py"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        measurement = dict(
            command="python scripts/check_consistency.py",
            elapsed_seconds=time.perf_counter() - start,
            maximum_resident_MiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            / 1024,
            stdout=run.stdout,
            stderr=run.stderr,
            threads_available=64,
            physical_cores=32,
            cpu="2 x Intel Xeon Silver 4314 @ 2.40 GHz",
            gpu_required=False,
            scope=(
                "fresh re-scoring, matched-target checks, census, source hashes and"
                " uncertainty"
            ),
        )
        (PAPER / "cpu_replay_measurement.json").write_text(
            json.dumps(measurement, indent=2) + "\n"
        )
        print(json.dumps(measurement, indent=2), flush=True)
    measurement = json.loads((PAPER / "cpu_replay_measurement.json").read_text())
    # The four seed-0 shards cover 86 donors. The base file supplies the first
    # recorded run for the remaining donors; later restarted/smoke rows are not
    # substituted for a full-budget fold. This selection concerns timing only.
    paths = [RAW / f"cellot_soskic_shard{i}_timing.json" for i in range(4)] + [
        RAW / "cellot_soskic_timing.json"
    ]
    records = []
    for path in paths:
        for position, r in enumerate(json.loads(path.read_text())):
            if r["seed"] == 0:
                records.append(
                    {
                        **r,
                        "source": str(path.relative_to(ROOT.parent)),
                        "source_row": position,
                    }
                )
    timed = pd.DataFrame(records).drop_duplicates(["donor", "seed"], keep="first")
    if timed.donor.nunique() != 106 or timed.sec.min() < 600:
        raise ValueError("Incomplete or non-full-budget CellOT timing selection")
    timed.to_csv(PAPER / "compute_cellot_donor_records.csv", index=False)
    canon = json.loads((PAPER / "CANONICAL_NUMBERS.json").read_text())
    rows = [
        dict(
            stage="CPU census verification",
            hardware="32 physical cores / 64 threads; no GPU",
            measured_units=canon["census_cells"],
            unit="census entries",
            seconds=measurement["elapsed_seconds"],
            elapsed_hours=measurement["elapsed_seconds"] / 3600,
            median_seconds=np.nan,
            range_min_seconds=np.nan,
            range_max_seconds=np.nan,
            host_RAM_MiB=measurement["maximum_resident_MiB"],
            scope=(
                f"{canon['census_bundles']:,} selected mean-profile bundles; complete"
                " consistency check; single measured invocation"
            ),
        ),
        dict(
            stage="CellOT T2, seed 0",
            hardware="one NVIDIA L40 (48 GB) per fit",
            measured_units=106,
            unit="donor fits",
            seconds=timed.sec.sum(),
            elapsed_hours=timed.sec.sum() / 3600,
            median_seconds=timed.sec.median(),
            range_min_seconds=timed.sec.min(),
            range_max_seconds=timed.sec.max(),
            host_RAM_MiB=np.nan,
            scope=(
                "selected full-budget run records; 12,000 AE + 8,000 transport"
                " iterations; serial sum"
            ),
        ),
    ]
    ledgerpath = (
        RAW / "compute_job_ledger.csv"
        if (RAW / "compute_job_ledger.csv").exists()
        else REV / "09_draft/tables/compute_job_ledger.csv"
    )
    ledger = pd.read_csv(ledgerpath)
    for model in ["scGPT", "scFoundation"]:
        frame = ledger[
            (ledger.wave == "2 foundation")
            & (ledger.model == model)
            & (ledger.task == "T2")
        ]
        units = frame.note.str.extract(r"^(\d+) units")[0].astype(int).sum()
        rows.append(
            dict(
                stage=f"{model} T2 recorded chunks",
                hardware="one NVIDIA L40 (48 GB) per fit",
                measured_units=int(units),
                unit="donor fits",
                seconds=frame.sec.sum(),
                elapsed_hours=frame.sec.sum() / 3600,
                median_seconds=np.nan,
                range_min_seconds=np.nan,
                range_max_seconds=np.nan,
                host_RAM_MiB=np.nan,
                scope=(
                    f"{len(frame)} recorded chunks; partial if fewer than 106 donors;"
                    " optimizer scope in training table"
                ),
            )
        )
    for model in ["STATE", "scPRAM"]:
        # outputs/ is gitignored in its entirety, so this died with FileNotFoundError on a fresh
        # clone and took two of make summaries' eight steps with it. The timing records are
        # deposited under provenance/timing/, which is tracked; the working copy is the fallback.
        path = ROOT / f"provenance/timing/{model.lower()}_soskic_timing.json"
        if not path.is_file():
            path = ROOT / f"outputs/additional_models/{model.lower()}_soskic_timing.json"
        frame = pd.DataFrame(json.loads(path.read_text()))
        rows.append(
            dict(
                stage=f"{model} T2 timing sample",
                hardware="one NVIDIA L40 (48 GB) per fit",
                measured_units=len(frame),
                unit="donor fits",
                seconds=frame.sec.sum(),
                elapsed_hours=frame.sec.sum() / 3600,
                median_seconds=frame.sec.median(),
                range_min_seconds=frame.sec.min(),
                range_max_seconds=frame.sec.max(),
                host_RAM_MiB=np.nan,
                scope=(
                    "two-donor pilot timing only; not a full-task total or a memory"
                    " measurement"
                ),
            )
        )
        paths.append(path)
    curves = [
        ROOT / f"results/newdata/scgpt_donor_curve_shard{i}_timing.json"
        for i in range(3)
    ]
    frame = pd.DataFrame([r for p in curves for r in json.loads(p.read_text())])
    if (
        len(frame) != 50
        or frame.duplicated(["eval_donor", "n_train_donors", "seed"]).any()
    ):
        raise ValueError("Unexpected final learning-curve timing coverage")
    rows.append(
        dict(
            stage="scGPT matched learning curve",
            hardware="one NVIDIA L40 (48 GB) per fit",
            measured_units=50,
            unit="donor/grid fits",
            seconds=frame.sec.sum(),
            elapsed_hours=frame.sec.sum() / 3600,
            median_seconds=frame.sec.median(),
            range_min_seconds=frame.sec.min(),
            range_max_seconds=frame.sec.max(),
            host_RAM_MiB=np.nan,
            scope="final seed-0 shards only; ten epochs, 8,000 stimulated-cell cap",
        )
    )
    paths += [ledgerpath, *curves]
    summary = pd.DataFrame(rows)
    summary.to_csv(PAPER / "compute_evidence_summary.csv", index=False)
    (PAPER / "compute_evidence_provenance.json").write_text(
        json.dumps(
            dict(
                sources=[
                    dict(
                        path=str(p.relative_to(ROOT.parent)),
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                    )
                    for p in paths
                ],
                gpu_memory=(
                    "48-GB L40 worker configuration, not a measured minimum; no uniform"
                    " peak-VRAM audit exists"
                ),
                time=(
                    "elapsed seconds, not utilization-integrated GPU-hours; parallel"
                    " execution changes calendar time"
                ),
                completeness=(
                    "partial rerun estimates only; no defensible all-census retraining"
                    " total"
                ),
                revision_ledger_elapsed_hours=float(ledger.sec.sum() / 3600),
                ledger_scope=(
                    "includes excluded/failed/repeated jobs and partial timings; not"
                    " added to model-specific estimates"
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
