#!/usr/bin/env python
"""Verify a finished job actually produced something the census can use.

  runs/postflight.py <job-id> <deposit-dir> [result-csv ...]

Exit 0 = usable. Non-zero = the job "succeeded" but its output cannot be reported.
rc=0 from the script is NOT evidence of a filled cell -- these all happened on 2026-09-14:

  * PertAdapt T4 exited 0 with ran=False on every unit (a missing env var, which reads exactly
    like a model limitation in the CSV).
  * Eight runs exited 0 and deposited no bundle at all.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: postflight.py <job-id> <deposit-dir> [result-csv ...]", file=sys.stderr)
        sys.exit(2)
    job_id, dep = sys.argv[1], Path(sys.argv[2])
    csvs = [Path(a) for a in sys.argv[3:]]
    started = float(Path(f"{dep}/.{job_id}.t0").read_text()) if Path(f"{dep}/.{job_id}.t0").exists() else 0.0
    problems: list[str] = []

    fresh = [p for p in dep.glob("*.npz") if p.stat().st_mtime >= started]
    if not fresh:
        problems.append(
            f"no bundle was deposited into {dep} during this job -- the census is assembled from "
            "predictions/**/*.npz, so nothing this job computed can be reported"
        )

    # With several jobs depositing into one directory, `fresh` also catches bundles another job
    # wrote during this window, and one of them may be half written at this instant. Reading it
    # must not fail THIS job: the pass/fail decision below rests on the job's own result CSV.
    for f in fresh:
        try:
            z = np.load(f, allow_pickle=True)
        except Exception as e:  # noqa: BLE001 -- a concurrent writer, not this job's problem
            print(f"  (skipped {f.name}: {e.__class__.__name__}; likely still being written)")
            continue
        if "pred_means" not in z.files:
            print(f"  (skipped {f.name}: no pred_means yet)")
            continue
        n_dec = int(z["n_declined_cells"]) if "n_declined_cells" in z.files else 0
        dec = [str(s) for s in z["declined_strata"]] if "declined_strata" in z.files else []
        if dec:
            print(f"  coverage {f.name}: {len(dec)} stratum/strata declined "
                  f"({n_dec} cells): {', '.join(dec[:6])}{' ...' if len(dec) > 6 else ''}")

    for c in csvs:
        if not c.exists():
            problems.append(f"result CSV missing: {c}")
            continue
        rows = list(csv.DictReader(open(c)))
        if not rows:
            problems.append(f"{c.name}: no rows")
            continue
        # Two result schemas reach here. run_obligation.py writes one row per unit with `ran` and
        # `leak_free`; the per-task drivers (state_soskic.py and friends) write one row per
        # (unit, metric) with `leak_free` and no `ran` at all. Reading the second with the first's
        # rule says "every unit has ran=False" about a job that completed every donor -- which is
        # exactly what it did to state_T2_batch_3, 13 donors and all.
        cols = set(rows[0])
        if "ran" in cols:
            ran = [r for r in rows if str(r.get("ran", "")).lower() == "true"]
            if not ran:
                why = next((r.get("error", "") for r in rows if r.get("error")), "")
                # Say which KIND of failure. A timeout and a model that cannot express the task
                # look identical in this CSV -- both are ran=False -- and reading the second for
                # the first is how scFoundation T3/T4 burned four hours a unit, five units deep,
                # before anyone opened the error column.
                kind = ("the per-unit TIME BUDGET ran out (TimeoutExpired), which is a scheduling "
                        "failure, NOT a model limitation: raise the adapter's timeout_s and/or run "
                        "it sharded with --chunk" if "Timeout" in why else
                        "this is a FAILED job, not a filled cell")
                problems.append(f"{c.name}: every unit has ran=False -- {kind}"
                                + (f" | {why[:200]}" if why else ""))
            elif len(ran) < len(rows):
                problems.append(f"{c.name}: {len(rows) - len(ran)} of {len(rows)} units did not run")
        else:
            ran = rows
            print(f"  ({c.name}: no `ran` column -- per-task driver schema; checking leak_free "
                  f"over {len(rows)} row(s))")
        leaky = [r for r in ran if str(r.get("leak_free", "true")).lower() == "false"]
        if leaky:
            problems.append(f"{c.name}: {len(leaky)} unit(s) reported leak_free=False")

    if problems:
        print(f"POSTFLIGHT FAILED [{job_id}]", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(4)
    print(f"postflight: {job_id} OK ({len(fresh)} bundle(s) deposited into the shared directory "
          "during this job -- concurrent jobs' bundles are listed too)")


if __name__ == "__main__":
    main()
