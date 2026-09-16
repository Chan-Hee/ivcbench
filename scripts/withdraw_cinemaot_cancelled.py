#!/usr/bin/env python
"""Take the cancelled CINEMA-OT re-run back out of the current-execution store.

NATIVE_102_FINAL_REVIEW.md puts CINEMA-OT at X in all six cells (F07, M05) and says what to do
with the numbers already on disk: keep them as historical, and separate the historical path from
the current run. predictions/v2_native/ IS the current-run path, so no CINEMA-OT bundle belongs
in it. The three jobs that wrote them -- cinemaot_T1_v2, cinemaot_T2_v2, cinemaot_T5c_v2 -- carry
"CANCELLED by the final 102 ruling" in their own status files, and T5c was stopped after three of
its four units.

114 of those bundles were already withdrawn on 2026-09-14. That withdrawal stopped binding.
_is_withdrawn pins a decision to the file's sha256 -- "still the exact execution that was
withdrawn" -- and the cancelled re-run then overwrote 107 of those same paths, so the recorded
sha no longer matched anything on disk and the bundles quietly became eligible again. They could
not reach a reported cell, since CINEMA-OT is in no roster, but they were being re-scored and
carried into the diagnostic rows.

This re-pins every CINEMA-OT bundle under v2_native at the sha it actually has now, and adds the
three C5_LOCT units that were never registered at all. Nothing is deleted. The historical results
stay where they are, under predictions/C1, C2 and C5.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "results" / "_paper" / "withdrawn_bundles.csv"
PAT = "predictions/v2_native/*CINEMA-OT*.npz"
REASON = (
    "withdrawn 2026-09-16: CINEMA-OT is X in all six cells of NATIVE_102_FINAL_REVIEW.md "
    "(F07, M05) and is excluded from the current execution and inference family; the jobs that "
    "deposited this file (cinemaot_T1_v2 / T2_v2 / T5c_v2) were cancelled under that ruling, "
    "T5c after three of its four units. Re-pinned at the sha this cancelled run left behind: the "
    "2026-09-14 withdrawal was pinned to the earlier execution's sha and stopped binding when the "
    "file was overwritten. The historical CINEMA-OT numbers are kept under predictions/C1, C2, C5."
)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    apply = "--apply" in sys.argv
    rows = list(csv.DictReader(open(REG)))
    by_path = {r["bundle_path"]: r for r in rows}

    fresh, repinned, added = [], 0, 0
    for p in sorted(ROOT.glob(PAT)):
        rel = str(p.relative_to(ROOT))
        got = sha256(p)
        old = by_path.get(rel)
        if old is None:
            added += 1
        elif old["sha256"] == got:
            continue                      # already binding, leave the existing reason alone
        else:
            repinned += 1
        fresh.append({"bundle_path": rel, "sha256": got, "reason": REASON})

    print(f"{len(list(ROOT.glob(PAT)))} CINEMA-OT bundle(s) under predictions/v2_native/")
    print(f"  {repinned} withdrawal(s) re-pinned to the sha now on disk")
    print(f"  {added} never registered (the cancelled T5c units)")
    if not fresh:
        print("nothing to do")
        return
    if not apply:
        print("\n(report only; pass --apply to write the registry)")
        return

    touched = {r["bundle_path"] for r in fresh}
    out = [r for r in rows if r["bundle_path"] not in touched] + fresh
    with open(REG, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["bundle_path", "sha256", "reason"])
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {REG} ({len(out)} entries)")


if __name__ == "__main__":
    main()
