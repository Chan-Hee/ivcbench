#!/usr/bin/env python3
"""Stage the immune-program reanalysis inputs out of this archive.

The reanalysis was written against a frozen workspace that carried its own copies of five inputs.
Every one of them is already here, byte for byte: the 200 prediction bundles under predictions/,
the two cluster modules that define the program membership, and the two census tables. The only
file that was not is the bundle manifest, and it is a one-line filter of the deposited census
manifest -- `task in {T1, T3, T5c}` -- with two columns that restate `bundle_path` and `sha256`.

So nothing has to be added to the deposit for the analysis to run: this module rebuilds that
workspace layout from what the archive already ships, and the analysis reads it unchanged. Keeping
the shape identical is deliberate. The scoring code, its masks, its support rules, its NA
conventions and its tolerances are the approved ones, untouched, and the per-file sha256 check the
analysis already performed still runs -- now against the census manifest's own column.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

TASKS = ("T1", "T3", "T5c")
EXPECTED_ROWS = 200


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_manifest(root):
    """The 200 bundles the program reanalysis scores, from the deposited census manifest."""
    root = Path(root)
    census = pd.read_csv(root / "results/_paper/census_bundle_manifest.csv", keep_default_na=False)
    sel = census[census.task.isin(TASKS)].reset_index(drop=True).copy()
    if len(sel) != EXPECTED_ROWS:
        raise SystemExit(f"expected {EXPECTED_ROWS} T1/T3/T5c bundles, found {len(sel)}")
    # the frozen workspace kept its own copy of each bundle; here the archive's own copy IS it
    sel["snapshot_path"] = sel["bundle_path"]
    sel["snapshot_sha256"] = sel["sha256"]
    return sel


def stage(root, into):
    """Materialise the derived bundle manifest and return the directory holding it.

    The other four inputs are read from their own places in the archive, so the provenance record
    names src/ivcbench/clusters/c3.py rather than a copy of it.
    """
    root, into = Path(root), Path(into)
    inputs = into / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    selected_manifest(root).to_csv(inputs / "selected_bundle_manifest.csv", index=False)
    return inputs
