#!/usr/bin/env python
"""Add this revision's native-coverage runs to the compute ledger, from the run records themselves.

WHY. Supplementary Table S16 answers "what did this cost" (reviewer 2 comment 8) by reading
results/provenance_inputs/compute/compute_job_ledger.csv and matching each job to the reported
panel. Every run of 2026-09-14 -- the re-runs under the corrected code and the newly opened cells --
is absent from it, so the table would report the cost of the SUBMITTED census while the paper
reports a different one. The numbers here are not estimated: each job's elapsed time comes from its
own runs/<id>.status file, which launch.sh stamps at start and at completion.

  python scripts/append_run_ledger.py            # report what would be added
  python scripts/append_run_ledger.py --apply    # write it
"""
from __future__ import annotations

import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LEDGER = ROOT / "results/provenance_inputs/compute/compute_job_ledger.csv"
WAVE = "4 native coverage (2026-09-14)"
TS = "%Y-%m-%dT%H:%M:%SZ"

# how a job id maps onto (model, task) when the command does not name them outright
CLUSTER_TASK = {"C1": "T1", "C2": "T2", "C3": "T3", "C4": "T4"}


from entrypoints import resolve as parse, shard_of  # ONE copy of the mapping


def main() -> None:
    apply = "--apply" in sys.argv
    rows = list(csv.DictReader(open(LEDGER)))
    fields = list(rows[0])
    have = {(r["model"], r["task"], r["sec"]) for r in rows}
    # The elapsed-seconds key only catches a job whose time reproduces exactly, and the ledger
    # already records which JOB each row came from, so key on that too. Without it a second pass
    # of finish_census.sh re-adds a job the first pass had dropped as superseded -- the by_cell
    # rule below picks the latest run of a cell out of `add` alone, and once the winner is in
    # `rows` the loser is no longer competing with anything. finish_census.sh is meant to be
    # re-runnable, so a second pass must not inflate the compute total in Table S16.
    have_source = {r.get("source", "") for r in rows}

    def _end_of(job: str) -> float:
        """UTC end stamp of a finished job, from its status file."""
        st = RUNS / f"{job}.status"
        if not st.is_file():
            return 0.0
        m = re.search(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)", st.read_text())
        if not m:
            return 0.0
        return datetime.strptime(m.group(1), TS).replace(tzinfo=timezone.utc).timestamp()

    def _shard_of_job(job: str) -> str:
        cmd = RUNS / f"{job}.cmd"
        sh = shard_of(cmd.read_text().strip()) if cmd.is_file() else None
        return f"{sh[0]}/{sh[1]}" if sh else ""

    # The latest run ALREADY RECORDED for each cell. The by_cell rule below keeps only the last
    # run of a cell, but it compares candidates against each other, not against the ledger -- so
    # once the winner is written, a discarded earlier attempt for the same cell stops competing
    # with anything and is added on the next pass. state_C5 (05:30Z) did exactly that against the
    # v2_state_C5 row (07:51Z) it had already lost to.
    recorded_end = {}
    for r in rows:
        src = r.get("source", "")
        if not src.startswith("runs/") or not src.endswith(".status"):
            continue
        job = src[len("runs/"):-len(".status")]
        cell = (r["model"], r["task"], _shard_of_job(job))
        recorded_end[cell] = max(recorded_end.get(cell, 0.0), _end_of(job))

    add, skipped, superseded = [], [], []
    for st in sorted(RUNS.glob("*.status")):
        text = st.read_text().strip()
        if not text.startswith("DONE:0"):
            skipped.append((st.stem, text.split()[0]))
            continue
        cmd_file = RUNS / f"{st.stem}.cmd"
        if not cmd_file.exists():
            skipped.append((st.stem, "no .cmd"))
            continue
        cmd = cmd_file.read_text()
        target = parse(cmd)
        if target is None:
            skipped.append((st.stem, "cannot resolve model/task"))
            continue
        # The RUNNING stamp is overwritten on completion, so the start time comes from the wrapper
        # launch.sh writes at dispatch: runs/<id>.run.sh is created once, at t0, and never touched
        # again. The end time is the UTC stamp in the DONE line. A deposit-side .t0 marker exists
        # for later jobs and is preferred when present.
        m = re.search(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)", text)
        wrapper = RUNS / f"{st.stem}.run.sh"
        if not (m and wrapper.exists()):
            skipped.append((st.stem, "no timestamps"))
            continue
        # the stamp is UTC; without tzinfo Python reads it as local time and the
        # elapsed time comes out 9 h short here, i.e. negative for every short job
        end = datetime.strptime(m.group(1), TS).replace(tzinfo=timezone.utc).timestamp()
        t0 = Path(str(ROOT / "predictions/v2_native") + f"/.{st.stem}.t0")
        start = float(t0.read_text().strip()) if t0.exists() else wrapper.stat().st_mtime
        sec = round(max(end - start, 0.0), 1)
        if sec <= 0:
            skipped.append((st.stem, "nonpositive elapsed"))
            continue
        key = (target[0], target[1], f"{sec}")
        if key in have or f"runs/{st.stem}.status" in have_source:
            continue
        # A cell re-run under the corrected code appears twice in runs/: the discarded first
        # attempt (no bundle deposited, pre-fix code) and the run that produced the deposit.
        # Only the second is a cost of the reported census -- counting both would inflate S16.
        superseded.append((st.stem, target)) if False else None
        _sh = shard_of(cmd)
        _cell = (target[0], target[1], f"{_sh[0]}/{_sh[1]}" if _sh else "")
        if end <= recorded_end.get(_cell, 0.0):
            skipped.append((st.stem, "an earlier attempt at a cell the ledger already records"))
            continue
        add.append({
            "wave": WAVE, "model": target[0], "task": target[1],
            "sec": f"{sec}", "hours": f"{sec / 3600:.3f}",
            "source": f"runs/{st.stem}.status", "note": f"job {st.stem}; rc=0",
            "_shard": f"{_sh[0]}/{_sh[1]}" if _sh else "",
            "_end": end,
        })

    # Keep only the LAST run of each cell: earlier attempts were discarded and left no deposit, so
    # they are not a cost of the census the paper reports.
    #
    # A SHARD is not an earlier attempt. Cells too slow to run in one job are split with
    # --chunk i n, and the shards are complementary: STATE T2 is eight of them, CellOT T5c four,
    # scFoundation T3 five. Keying on (model, task) alone would have kept ONE and thrown the rest
    # away, reporting an eighth of the donor cell's cost. Key on the shard as well, and pick the
    # latest END TIME rather than the lexicographically largest job id -- the id ordering put
    # "v2_" first by luck, and the new ids (state_T2b_*, cellot_T5c_b*) do not carry it at all.
    by_cell = {}
    for a in add:
        by_cell.setdefault((a["model"], a["task"], a["_shard"]), []).append(a)
    dropped = []
    keep = []
    for cell, jobs in by_cell.items():
        chosen = max(jobs, key=lambda a: a["_end"])
        keep.append(chosen)
        dropped += [j for j in jobs if j is not chosen]
    add = keep
    for a in add + dropped:
        a.pop("_shard", None), a.pop("_end", None)

    print(f"compute ledger: {len(rows)} existing rows")
    if dropped:
        print(f"{len(dropped)} superseded attempt(s) excluded (no deposit, pre-fix code):")
        for d in dropped:
            print(f"  - {d['model']} {d['task']}  {float(d['hours']):.2f} h  {d['note']}")
    print(f"{len(add)} job(s) to add ({sum(float(a['hours']) for a in add):.2f} GPU-hours)")
    for a in sorted(add, key=lambda a: -float(a["hours"])):
        print(f"  {a['model']:<20} {a['task']:<5} {float(a['hours']):>7.2f} h   {a['note']}")
    if skipped:
        print(f"\n{len(skipped)} run record(s) not added:")
        for name, why in skipped:
            print(f"  - {name}: {why}")
    if not apply:
        print("\n(report only; pass --apply to write)")
        return
    with open(LEDGER, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows + [{k: a.get(k, "") for k in fields} for a in add])
    print(f"\nwrote {LEDGER} ({len(rows) + len(add)} rows)")


if __name__ == "__main__":
    main()
