#!/usr/bin/env python3
"""Put the hand-driven scFoundation re-runs into the compute ledger.

append_run_ledger.py reads runs/*.status, which the dispatcher writes. The scFoundation unseen-gene
and unseen-knockout cells were re-run from wrapper scripts in a screen session after the queued
attempts were held and killed at batch_size=2 (runs/scf_T4_u0.status "would launch at batch_size=2,
which never completed a unit"; runs/scf_T4_u1.status "KILLED 2026-09-15 by author"). Those wrappers
wrote no status file, so the ledger never saw them:

  - T3 was absent entirely, which build_cell_ledger.py reports as "cells with NO measurement".
  - T4 carried a 535 s row from the 2026-09-02 foundation wave, not the run whose bundles are
    deposited. Two rows for one cell would double-count it in Table S16, so the stale row is
    replaced rather than kept beside the new one.

Each timing is the wrapper's own START/END pair, matched to the deposited bundle by its mtime.
That is the same evidentiary standard as the rows already in the ledger sourced from a log rather
than a status file (raw/foundation/c4_scf.log, raw/admission/sweep.log).

Discarded attempts are not added. The first shifrut run was killed by the eight-hour adapter budget
at 2,952 s (rc=137) and the cache-off retry took 4,509 s and was superseded by the cache-on run
that deposited; neither produced a bundle, and the ledger counts what the census reports.

    python scripts/record_handrun_timings.py [--apply]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results/provenance_inputs/compute/compute_job_ledger.csv"
WAVE = "4 native coverage (2026-09-14)"

# (task, seconds, the wrapper log the timing is read from, which unit it produced)
RUNS = [
    ("T3", 10740, "logs/scf_tail_driver.log", "chunk 0 of 5 (chen)"),
    ("T3", 4334, "logs/scf_real.log", "chunk 1 of 5"),
    ("T3", 4942, "logs/scf_real.log", "chunk 2 of 5"),
    ("T3", 5681, "logs/scf_real.log", "chunk 3 of 5"),
    ("T3", 5350, "logs/scf_shifrut_driver.log", "chunk 4 of 5 (shifrut), encoder cache on"),
    ("T4", 2892, "logs/scf_real.log", "chunk 0 of 2 (25% holdout)"),
    ("T4", 2599, "logs/scf_real.log", "chunk 1 of 2 (50% holdout)"),
]
# the row this supersedes: same cell, earlier wave, not the execution that deposited
STALE = {("scFoundation", "T4", "raw/foundation/c4_scf.log")}


def main() -> None:
    apply = "--apply" in sys.argv
    rows = list(csv.DictReader(open(LEDGER)))
    fields = list(rows[0])

    dropped = [r for r in rows
               if (r["model"], r["task"], r["source"]) in STALE]
    kept = [r for r in rows if r not in dropped]

    have = {(r["model"], r["task"], r["source"], r["sec"]) for r in kept}
    add = []
    for task, sec, log, what in RUNS:
        key = ("scFoundation", task, log, f"{float(sec)}")
        if key in have:
            continue
        add.append({"wave": WAVE, "model": "scFoundation", "task": task,
                    "sec": f"{float(sec)}", "hours": f"{sec / 3600:.3f}",
                    "source": log,
                    "note": f"hand-driven re-run, {what}; wall clock from the wrapper's START/END"})

    print(f"ledger: {len(rows)} rows")
    print(f"  superseded (earlier wave, not the deposited execution): {len(dropped)}")
    for d in dropped:
        print(f"    - {d['model']} {d['task']} {d['sec']}s  {d['source']}")
    print(f"  to add: {len(add)}  ({sum(float(a['hours']) for a in add):.2f} GPU-hours)")
    for a in add:
        print(f"    + {a['model']} {a['task']} {float(a['hours']):.3f} h  {a['note'][:60]}")
    if not apply:
        print("\n(report only; pass --apply to write)")
        return
    with open(LEDGER, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fields} for r in kept + add])
    print(f"\nwrote {LEDGER} ({len(kept) + len(add)} rows)")


if __name__ == "__main__":
    main()
