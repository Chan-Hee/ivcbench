#!/usr/bin/env python
"""Withdraw the partial CINEMA-OT T5c deposit, and leave the completed T1/T2 re-run alone.

cinemaot_T5c_v2 stopped after three of its four units -- its log records B, Mono and NK and no
T_cells -- and its status file says "CANCELLED by the final 102 ruling", which puts CINEMA-OT at
X in all six cells (F07, M05). Those three bundles were never registered anywhere, so the store
held them alongside the complete four-unit historical deposit under predictions/C5/. A partial run
must not stand as the record for a cell when a complete one does, and the ruling's instruction is
to keep the existing numbers as historical, so the three partial bundles are the ones to withdraw.

Not the T1 and T2 bundles in the same directory. Their withdrawal on 2026-09-14 was about a
DEFECT in the run before them -- the reference arm was the held unit's controls alone, collapsing
the prediction to the training treated mean, measured at 0.995 correlation across the eight held
lineages -- and it names its own remedy: "superseded by the re-run whose reference also carries
the training controls". cinemaot_T1_v2 and cinemaot_T2_v2 ARE that re-run, they finished every
unit, and they wrote over the bundles they replaced. The lapse of those 114 entries is that
supersession working, not a decision being undone. CINEMA-OT is still X in all six cells; the
deposit keeps the numbers as the historical record the ruling asks for, and Table S15b counts
them among the executions the deposit preserves but the census does not.

    python scripts/withdraw_cinemaot_cancelled.py [--apply]     (default: report only)
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "results" / "_paper" / "withdrawn_bundles.csv"
PARTIAL = sorted((ROOT / "predictions" / "v2_native").glob("C5_LOCT__CINEMA-OT__C5_loct_*.npz"))
REASON = (
    "withdrawn 2026-09-16: partial output of cinemaot_T5c_v2, which stopped after three of its "
    "four units and whose status records \"CANCELLED by the final 102 ruling\" -- CINEMA-OT is X "
    "in all six cells (NATIVE_102_FINAL_REVIEW.md F07, M05). The complete four-unit deposit under "
    "predictions/C5/ remains as the historical record for this cell."
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

    fresh = []
    for p in PARTIAL:
        rel = str(p.relative_to(ROOT))
        got = sha256(p)
        old = by_path.get(rel)
        if old is not None and old["sha256"] == got and old["reason"] == REASON:
            continue
        fresh.append({"bundle_path": rel, "sha256": got, "reason": REASON})

    print(f"{len(PARTIAL)} partial CINEMA-OT T5c bundle(s); {len(fresh)} to register")
    for f in fresh:
        print(f"  - {f['bundle_path']}")
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
