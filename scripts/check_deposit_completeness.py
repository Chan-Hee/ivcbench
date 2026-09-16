#!/usr/bin/env python3
"""Every bundle the census names must be in the repository a reader clones.

The availability statement promises that re-scoring the deposited bundles reproduces the reported
census. That is only true if the bundles ship. .gitignore ignores *.npz and allowlists directories
by name, and it named only predictions/C1 through C5 -- so when this revision's re-runs began
depositing into predictions/v2_native, predictions/C2_metric_aligned and the predictions/ root, a
fresh clone was missing 567 of the 1,401 bundles census_bundle_manifest.csv points at. Nothing
noticed, because every check ran in the working tree, where the files exist.

This checks what git TRACKS, not what is on disk, which is the only question a reader's clone asks.

    python scripts/check_deposit_completeness.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/_paper/census_bundle_manifest.csv"


def main() -> int:
    with MANIFEST.open(encoding="utf-8") as fh:
        want = [r["bundle_path"] for r in csv.DictReader(fh)]
    tracked = set(subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "predictions", "results"],
        capture_output=True, text=True, check=True).stdout.split("\n"))
    missing = sorted(p for p in want if p not in tracked)
    print(f"census manifest: {len(want)} bundles; untracked: {len(missing)}")
    if missing:
        import collections
        by = collections.Counter(str(Path(p).parent) for p in missing)
        for d, n in by.most_common():
            print(f"  {n:5d}  {d}")
        print("\nA clone would not carry these, so the availability statement would be false.")
        print("Add the directory to .gitignore's allowlist, or explain why the census reads a")
        print("bundle the release does not ship.")
        return 1
    print("every census bundle is tracked; a clone can re-score the panel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
