#!/usr/bin/env python
"""Every bundle of a completed cell must come from the job that completed it.

THE HOLE THIS CLOSES. A re-run deposits under the SAME filename as the run it replaces, which is
what makes supersede_reruns.py necessary. It also means a re-run that dies half way leaves the cell
holding a mixture: five lineages from the new runner, three from the old one still sitting at the
same paths. Nothing downstream notices -- the assembler counts units and checks sha256 against the
withdrawal registry, and a mixed set passes both. The scores would then be a blend of two different
data contracts, which is exactly the defect the re-run was for.

THE CHECK. runs/launch.sh stamps <deposit>/.<job>.t0 with the job's start time. For each job the
manifest calls complete, every bundle of that job's cell must be newer than that stamp. A bundle
older than it was written by something else.

  python scripts/verify_cell_provenance.py            report
  python scripts/verify_cell_provenance.py --strict   exit 9 if any cell is mixed
"""
from __future__ import annotations

import argparse
import csv
import json
import glob
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CEN_CLUSTER = {
    "T1": ("C1_LOCT",), "T2": ("C2", "C2_LODO"), "T3": ("C3_LO_gene",),
    "T4": ("C4_Axis2", "C4"), "T5c": ("C5_LOCT",), "T5u": ("C5_unseen_cpd",),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--manifest", default=str(ROOT / "results/_paper/run_manifest.csv"))
    a = ap.parse_args()

    dump = os.environ.get("IVCBENCH_PRED_DUMP") or str(ROOT / "predictions/v2_native")

    # model -> the bundles that exist for it, per cluster
    bundles = []
    for p in glob.glob(str(ROOT / "predictions" / "**" / "*.npz"), recursive=True):
        try:
            z = np.load(p, allow_pickle=True)
        except Exception:
            continue
        if "pred_means" not in z.files:
            continue
        bundles.append((
            str(z["cluster"]) if "cluster" in z.files else "",
            str(z["model"]) if "model" in z.files else "",
            p, os.path.getmtime(p),
        ))

    # A SHARDED cell is completed by several jobs that start minutes apart on purpose, and each
    # writes its own units. Comparing every bundle of the cell against ONE shard's stamp calls the
    # units its siblings wrote "older than the job" -- STATE T2 has eight shards spread over
    # thirteen minutes and every one of them flagged the other seven's output. The run that matters
    # is the whole shard set, so the bound is the EARLIEST stamp among the jobs credited with the
    # cell. build_cell_ledger.py already treats a shard set as one unit of completion; this did not.
    by_cell = {}
    unstamped = []
    for r in csv.DictReader(open(a.manifest)):
        if r.get("complete") != "yes":
            continue
        t0p = Path(dump) / f".{r['job']}.t0"
        if not t0p.exists():
            # A historical job, from before the stamp existed. Skipping it is right -- there is
            # nothing to compare against -- but skipping it SILENTLY was not: five jobs have no
            # stamp, and one whole cell (scPRAM x T1, eight census bundles) is covered by nothing
            # else, so the summary printed "0 cell(s)" while never looking at it. Say so.
            unstamped.append((r["job"], r.get("model", "?"), r.get("task", "?")))
            continue
        try:
            t0 = float(t0p.read_text())
        except Exception:
            unstamped.append((r["job"], r.get("model", "?"), r.get("task", "?")))
            continue
        key = (r["model"], r["task"])
        cur = by_cell.get(key)
        if cur is None or t0 < cur[0]:
            by_cell[key] = (t0, r["job"], set())
        by_cell[key][2].add(r["job"])

    # A bundle rewritten by the panel-mask migration carries the migration's mtime, not the run's.
    # Where the migration recorded the original we use it; where it did not (the first migration
    # predates that record) the bundle cannot be checked this way and must not be counted as
    # passing.
    migrated, recovered = set(), {}
    mig = Path("results/_paper/panel_mask_migration.json")
    if mig.is_file():
        for e in json.loads(mig.read_text()):
            full = str(Path(e["bundle"]).resolve())
            migrated.add(full)
            if "mtime_before" in e:
                recovered[full] = float(e["mtime_before"])

    problems = []
    unverifiable = []
    for (model, task), (t0, job, jobs) in sorted(by_cell.items()):
        cl = CEN_CLUSTER.get(task, ())
        mine = [b for b in bundles if b[0] in cl and b[1] == model
                and Path(b[2]).parent.resolve() == Path(dump).resolve()]
        opaque = [b for b in mine
                  if str(Path(b[2]).resolve()) in migrated
                  and str(Path(b[2]).resolve()) not in recovered]
        if opaque:
            unverifiable.append((model, task, len(opaque), len(mine)))
        mtime = lambda b: recovered.get(str(Path(b[2]).resolve()), b[3])
        stale = [b for b in mine if b not in opaque and mtime(b) < t0]
        if stale:
            label = job if len(jobs) == 1 else f"{len(jobs)} shards, earliest {job}"
            problems.append((label, model, task, len(stale), len(mine),
                             sorted(Path(b[2]).name for b in stale)[:4]))

    checked = set(by_cell)
    blind = sorted({(m, t) for _, m, t in unstamped} - checked)
    print(f"verify_cell_provenance: {len(problems)} cell(s) hold bundles older than the job that "
          f"was credited with completing them; {len(checked)} cell(s) checked")
    if unstamped:
        print(f"  {len(unstamped)} job(s) carry no start stamp and were not compared: "
              + ", ".join(sorted(j for j, _, _ in unstamped)))
    for m, t in blind:
        print(f"  NOT CHECKED: {m} x {t} -- every job credited with this cell is unstamped, so "
              "the 0 above says nothing about it")
    for m, t, no, n in sorted(unverifiable):
        print(f"  NOT CHECKED: {m} x {t} -- {no} of {n} bundles were rewritten by the panel-mask "
              "migration before it recorded the original mtime, so their age proves nothing")
    for job, m, t, ns, n, names in problems:
        print(f"  {m} x {t} (job {job}): {ns} of {n} bundles predate the job -- {names}")
        print("     the cell is a MIXTURE of two runs; re-run it whole or withdraw the old paths")
    if problems and a.strict:
        raise SystemExit(9)


if __name__ == "__main__":
    main()
