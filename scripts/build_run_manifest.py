#!/usr/bin/env python
"""The selected-run manifest: what actually produced each census cell, from execution evidence.

WHY. Completion was being inferred from two things that are not evidence:
  * a bundle exists under predictions/v2_native/   -> "done"
  * the B3 correction moves the score by < 1e-4    -> "keep"
Neither says a model ran. PertAdapt T4 was counted done while its own CSV recorded
action=declined, ran=False on both folds; CPA T5c was counted done while the queue actually
dispatched the historical fingerprint->latent Ridge path.

This walks the run records instead: for every completed job it reads the job's own result CSV
(ran / leak_free / action per unit), the command that produced it (which model, task, runner and
environment), and the bundles it deposited. A cell is COMPLETE only when every unit of the split
ran, the run is leak-free, and a bundle exists for each unit.

  python scripts/build_run_manifest.py [--csv results/_paper/run_manifest.csv]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
# How many biological units a finished cell must carry, from the same numbers the census assembler
# enforces (assemble_cross_cluster.CELLS n_unit). Without this the denominator came from the job's
# own output, so `ran == units` was a tautology: a job that emitted 7 of 106 donors scored
# units=7, ran=7, complete=yes.
# T5u is ONE unit, not 28: the unseen-compound split is a single held-out set whose 28 compounds
# are strata inside one bundle, which is why assemble_cross_cluster.py:694 expects 1 there.
EXPECTED_UNITS = {"T1": 8, "T2": 106, "T3": 5, "T4": 2, "T5c": 4, "T5u": 1}


def _same_model(row_name: str, model: str) -> bool:
    """Does this result row belong to `model`?

    Exact string equality was too strict: the chemCPA summary writes
    "chemCPA (native, use_rdkit_embeddings)" and the census calls the same cell chemCPA, so two
    finished cells read as "no row names chemCPA". Compare on a normalised stem instead, while
    still refusing a row that names a DIFFERENT model -- which is the thing this check exists for.
    """
    # One census model, several spellings: the chemCPA T5c job is invoked as --model chemCPA and
    # its adapter reports baseline "CPA", because the census treats them as one group.
    GROUP = {"cpa": "cpa", "chemcpa": "cpa", "cpa / chemcpa": "cpa",
             "cpa/chemcpa (chemcpa)": "cpa", "biolord": "biolord"}

    def norm(x):
        x = (x or "").strip().lower().split("(")[0].strip()
        return GROUP.get(x, x)

    a, b = norm(row_name), norm(model)
    if not a:
        return False
    return a == b
RUNS = ROOT / "runs"
CLUSTER_TASK = {"C1": "T1", "C2": "T2", "C3": "T3", "C4": "T4"}
# One cluster run can cover two census columns: --cluster C5 holds out cell types (T5c) and
# compounds (T5u) in the same pass, and the rows say which is which. Attributing the whole job
# to a single task hid a finished T5u behind a T5c row.
REGISTRY_TASK = {
    "C1_LOCT": "T1", "C2_LODonor": "T2", "C3_LO_gene": "T3", "C4_LO_KO": "T4",
    "C5_LOCT": "T5c", "C5_unseen_cpd": "T5u",
}


from entrypoints import resolve as target_of, shard_of  # ONE copy of the mapping


def out_csv_of(cmd: str):
    """run_obligation writes --out <file>; run_cluster writes --outdir <dir>/<cluster>/results.csv."""
    m = re.search(r"--out\s+(\S+)", cmd)
    if m:
        return m.group(1)
    m = re.search(r"--outdir\s+(\S+)", cmd)
    if m:
        # results_raw.csv carries ran / leak_free per unit; results.csv does not
        hits = sorted(glob.glob(os.path.join(ROOT, m.group(1), "*", "results_raw.csv"))) or \
               sorted(glob.glob(os.path.join(ROOT, m.group(1), "*", "results.csv")))
        if hits:
            return os.path.relpath(hits[0], ROOT)
    # the chemCPA wrapper writes its summary under the out_dir it is given
    if "chemcpa_t5u" in cmd:
        cand = "outputs/native_rerun/chemcpa_t5u_shim/chemcpa_op3_unseen_compound_summary.csv"
        if (ROOT / cand).exists():
            return cand
    # run_c4_conditioned writes a fixed location
    if "run_c4_conditioned" in cmd:
        for cand in ("results/C4/conditioned_rows.json", "results/C4/results.csv"):
            if (ROOT / cand).exists():
                return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results/_paper/run_manifest.csv"))
    a = ap.parse_args()

    rows = []
    for st in sorted(RUNS.glob("*.status")):
        job = st.stem
        state = st.read_text().strip()
        cmd_file = RUNS / f"{job}.cmd"
        if not cmd_file.exists():
            continue
        # The status line is the author's disposition of the job, and only a clean finish may
        # evidence a cell. Everything else -- SUPERSEDED, CANCELLED, KILLED, UNUSABLE, DEFERRED,
        # STOPPED, DISCARDED, RUNNING, and DIAGNOSTIC-ONLY (a run kept for the record whose value
        # is not adopted) -- left a complete-looking output CSV behind, and this loop used to read
        # it anyway. Two STATE T2 shards from the pre-split runner were certifying that cell as
        # 2/8 complete while their bundles sat quarantined.
        if not state.startswith("DONE:0"):
            continue
        cmd = cmd_file.read_text().strip()
        _t = target_of(cmd)      # shared resolver returns None, not (None, None)
        if _t is None:
            continue
        model, task = _t
        oc = out_csv_of(cmd)
        units = ran = leak_ok = declined = 0
        detail = ""
        groups = {}

        def _t(v):
            return v is True or str(v).strip().lower() == "true"

        if oc and (ROOT / oc).exists():
            if oc.endswith(".json"):
                import json
                try:
                    raw = json.loads((ROOT / oc).read_text())
                except Exception:
                    raw = []      # being written, or not the shape we expect
                recs = raw if isinstance(raw, list) else raw.get("rows", [])
            else:
                recs = list(csv.DictReader(open(ROOT / oc)))
            # run_cluster writes a wide table; run_obligation writes one row per unit
            # `or recs` used to fall back to EVERY row when none named this model, which certified
            # a cell from somebody else's results: linear-shift-KOemb x T4 read complete=yes off
            # four STATE rows, because a STATE-only re-run had overwritten the shared C4 file.
            named = [r for r in recs if _same_model(r.get("baseline") or r.get("model"), model)]
            if named:
                recs = named
            elif any(r.get("baseline") or r.get("model") for r in recs):
                # the file names models, and none of them is this one
                recs = []
                detail = f"no row names {model}; this job does not evidence that cell"
            groups = {}
            for r in recs:
                groups.setdefault(REGISTRY_TASK.get(str(r.get("registry_task", "")), task), []).append(r)
            units = len(recs)
            # A per-task driver's summary has no `ran` column -- one row, an n_units count and a
            # score. Reading the run_obligation schema onto it reported ran=0 for a job that had
            # just deposited its bundle. Translate it instead.
            if recs and "ran" not in recs[0]:
                # Row count is not unit count here: the donor driver writes one row per
                # (donor, metric), so 13 donors arrive as 39 rows. Prefer an explicit count, then
                # the number of distinct biological units, and only then the row count.
                n = next((int(float(r[k])) for r in recs for k in ("n_units", "n_unit")
                          if r.get(k)), None)
                if n is None:
                    for key in ("donor", "unit", "split"):
                        if key in recs[0]:
                            n = len({r[key] for r in recs})
                            break
                if n is None:
                    n = len(recs)
                scored = any(
                    str(r.get(k, "")).strip() not in ("", "nan")
                    for r in recs for k in r if "score" in k.lower()
                )
                for g in groups.values():
                    for r in g:
                        r["ran"] = "true" if scored else "false"
                        r.setdefault("leak_free", "true")
                        r["_n_units"] = n
                units = n
            ran = sum(_t(r.get("ran")) for r in recs)
            leak_ok = sum(_t(r.get("leak_free")) for r in recs)
            declined = sum(str(r.get("action", "")).lower() in ("declined", "failed") for r in recs)
            if declined:
                detail = f"{declined} unit(s) declined/failed"
        runner = re.search(r"(\w+_runner\.py)", cmd)
        sh = shard_of(cmd)
        if not groups:
            groups = {task: []}
        for gtask, recs_g in sorted(groups.items()):
            if recs_g:
                units = int(recs_g[0].get("_n_units") or len(recs_g))
                ran = units if all(_t(r.get("ran")) for r in recs_g) else sum(_t(r.get("ran")) for r in recs_g)
                leak_ok = units if all(_t(r.get("leak_free", "true")) for r in recs_g) else sum(_t(r.get("leak_free")) for r in recs_g)
                declined = sum(str(r.get("action", "")).lower() in ("declined", "failed") for r in recs_g)
                detail = f"{declined} unit(s) declined/failed" if declined else ""
            exp = EXPECTED_UNITS.get(gtask)
            if exp is not None and sh:
                lo = exp // sh[1]                      # this shard's share, allowing the remainder
                exp = lo if exp % sh[1] == 0 else lo
            short = "" if exp is None else ("" if units >= exp else
                                            f"{units} of {exp} expected unit(s)")
            rows.append(dict(
                job=job, model=model, task=gtask, state=state.split()[0],
                shard=(f"{sh[0]}/{sh[1]}" if sh else ""),
                units=units, expected=("" if exp is None else exp),
                ran=ran, leak_free=leak_ok, declined=declined,
                result_csv=oc or "", runner=runner.group(1) if runner else "",
                complete=("yes" if (state.startswith("DONE:0") and units and ran == units
                                    and leak_ok == units and not declined and not short)
                          else "no"),
                note="; ".join(x for x in (detail, short) if x), command=cmd,
            ))

    rows.sort(key=lambda r: (r["model"], r["task"], r["job"]))
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["complete"] == "yes"]
    cells = {(r["model"], r["task"]) for r in ok}
    print(f"run manifest -> {a.csv}   ({len(rows)} job records)")
    print(f"jobs that completed every unit, leak-free, nothing declined: {len(ok)}")
    print(f"distinct (model, task) cells backed by such a job: {len(cells)}\n")
    bad = [r for r in rows if r["state"] == "DONE:0" and r["complete"] == "no"]
    if bad:
        print(f"{len(bad)} job(s) exited DONE:0 but are NOT complete:")
        for r in bad:
            print(f"  {r['model']:<14} {r['task']:<5} {r['job']:<20} "
                  f"ran {r['ran']}/{r['units']}  {r['note']}")


if __name__ == "__main__":
    main()
