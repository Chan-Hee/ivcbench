#!/usr/bin/env python3
"""Assemble Supplementary Table S8 from deposited per-task result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "_paper" / "Supplementary_Table_S8_energy_distance.csv"

SOURCES = (
    ROOT / "results" / "C1" / "results_raw.csv",
    ROOT / "results" / "C3" / "results_raw.csv",
    ROOT / "results" / "C4" / "results_raw.csv",
    ROOT / "results" / "C5" / "results_raw.csv",
)

OUT_COLUMNS = (
    "cluster",
    "dataset",
    "modality",
    "split",
    "model",
    "family",
    "response_direction_pearson_delta",
    "distributional_energy_distance",
)


def read_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ran" in df.columns:
        df = df[df["ran"].fillna(False).astype(bool)].copy()

    out = pd.DataFrame(
        {
            "cluster": df["cluster"],
            "dataset": df["dataset"],
            "modality": df.get("modality", pd.Series(index=df.index, dtype=object)),
            "split": df["split"],
            "model": df["baseline"],
            "family": df["family"],
            "response_direction_pearson_delta": df["pearson_delta"],
            "distributional_energy_distance": df["e_distance"],
        }
    )
    out["modality"] = out["modality"].fillna("RNA")
    return out


def assemble() -> pd.DataFrame:
    missing = [path for path in SOURCES if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing S12 source file(s): " + ", ".join(map(str, missing)))

    table = pd.concat([read_source(path) for path in SOURCES], ignore_index=True)
    # S12 reports the scored per-task distributional fidelity. For C1 (T1) the scored
    # task is leave-one-cell-type-out (cell-context); the auxiliary C1 leave-one-donor-out
    # and random-split (optimism-control) folds are not the T1 task and are excluded so the
    # per-task summary and the deposited table are consistent.
    aux_c1 = (table["cluster"] == "C1") & (~table["split"].astype(str).str.startswith("C1_loct"))
    table = table[~aux_c1]
    # Normalise the family label for the deterministic knockout-embedding shift to match the
    # method survey / census (Supplementary Tables S2a, S2b); results_raw carries a stale "latent".
    table.loc[table["model"] == "linear-shift-KOemb", "family"] = "Deterministic shift"
    table = table.drop_duplicates().sort_values(
        ["cluster", "dataset", "modality", "split", "family", "model"],
        kind="mergesort",
    )
    return table.loc[:, OUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV output path")
    args = parser.parse_args()

    table = assemble()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, float_format="%.10g")
    print(f"wrote {args.out} ({len(table)} rows)")


if __name__ == "__main__":
    main()
