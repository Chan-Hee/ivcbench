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

    problems = []
    for r in csv.DictReader(open(a.manifest)):
        if r.get("complete") != "yes":
            continue
        t0p = Path(dump) / f".{r['job']}.t0"
        if not t0p.exists():
            continue                      # a historical job, before the stamp existed
        try:
            t0 = float(t0p.read_text())
        except Exception:
            continue
        cl = CEN_CLUSTER.get(r["task"], ())
        mine = [b for b in bundles if b[0] in cl and b[1] == r["model"]
                and Path(b[2]).parent.resolve() == Path(dump).resolve()]
        stale = [b for b in mine if b[3] < t0]
        if stale:
            problems.append((r["job"], r["model"], r["task"], len(stale), len(mine),
                             sorted(Path(b[2]).name for b in stale)[:4]))

    print(f"verify_cell_provenance: {len(problems)} cell(s) hold bundles older than the job that "
          "was credited with completing them")
    for job, m, t, ns, n, names in problems:
        print(f"  {m} x {t} (job {job}): {ns} of {n} bundles predate the job -- {names}")
        print("     the cell is a MIXTURE of two runs; re-run it whole or withdraw the old paths")
    if problems and a.strict:
        raise SystemExit(9)


if __name__ == "__main__":
    main()
