#!/usr/bin/env python
"""Catch the two ways a long run wastes a day: it stops progressing, or it progresses wrongly.

Both have already happened here. scFoundation spent 4, 4.4 and 8 GPU-hours on units that never
finished, and nobody could see it because the adapter captured the runner's stderr into a pipe
that is only readable after exit. STATE T1, T5c and T2 finished cleanly and deposited values that
were wrong -- held controls in the fit, seen compounds withheld, a lineage collapsed to a constant.
A job that exits 0 is not evidence; a job that is running is not progress.

Run it repeatedly. It keeps its own state in runs/.healthcheck.json and reports only CHANGES,
so it is safe to poll. Exit code 1 if anything needs attention.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
STATE = RUNS / ".healthcheck.json"
DEPOSIT = Path(os.environ.get("IVCBENCH_PRED_DUMP", ROOT / "predictions/v2_native"))
# How long a job may legitimately be quiet is a property of the job, not of the machine. One
# STATE donor is ~6 min and one CellOT compound ~1.5 min, but ONE PERTURBNET DONOR IS 53 MIN and a
# scFoundation unit prints nothing between epochs that can take longer still. A single global
# threshold would cry wolf on exactly the two longest jobs, and an alarm that fires on healthy runs
# is worse than no alarm: it trains you to ignore the real one.
STALL_MIN_DEFAULT = int(os.environ.get("IVCBENCH_STALL_MIN", "45"))
STALL_MIN_BY_JOB = {
    "pn_": 90,      # PerturbNet T2: 53 min per donor measured, so 90 is comfortably past one
    "perturbnet": 90,
    "scf_": 120,    # scFoundation at the published batch: 9,090 steps through a frozen encoder
    "cellot": 60,   # 2000 iters per compound, and it slows under CPU contention
}


def stall_minutes(job: str) -> int:
    for pre, val in STALL_MIN_BY_JOB.items():
        if job.startswith(pre):
            return val
    return STALL_MIN_DEFAULT


def running_jobs():
    for f in sorted(RUNS.glob("*.status")):
        try:
            if f.read_text().startswith("RUNNING"):
                yield f.stem
        except OSError:
            continue


def _descendant_pids(job: str) -> set[int]:
    """Every pid under this job's run.sh, so a per-job artefact can be told from a shared one."""
    try:
        roots = subprocess.run(["pgrep", "-f", f"{job}.run.sh"], capture_output=True,
                               text=True, timeout=20).stdout.split()
    except Exception:
        return set()
    out, stack = set(), [int(r) for r in roots]
    while stack:
        pid = stack.pop()
        if pid in out:
            continue
        out.add(pid)
        try:
            kids = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True,
                                  text=True, timeout=20).stdout.split()
            stack += [int(k) for k in kids]
        except Exception:
            pass
    return out


def _cellot_lineage(job: str) -> str | None:
    """chunk i of n -> the i-th lineage, the order run_obligation shards them in."""
    cmd = RUNS / f"{job}.cmd"
    if not cmd.exists():
        return None
    m = re.search(r"--chunk\s+(\d+)\s+(\d+)", cmd.read_text())
    prog = RUNS / "cellot_progress.tsv"
    if not m or not prog.exists():
        return None
    lineages = sorted({l.split("\t", 1)[0] for l in prog.read_text().splitlines() if "\t" in l})
    i = int(m.group(1))
    return lineages[i] if i < len(lineages) else None


def progress_signal(job: str) -> tuple[str, int]:
    """A number that must go up while the job is healthy, plus how it was measured.

    It has to be THIS job's number. Three CellOT lineages share one progress file and four
    PertAdapt units write into the same log directory; counting the shared artefact lets a healthy
    sibling hide a stalled job, which is the exact failure this script exists to catch.
    """
    log = RUNS / f"{job}.log"
    # CellOT: count only the lines this job's own lineage wrote, after its own START
    if job.startswith("cellot"):
        lin = _cellot_lineage(job)
        prog = RUNS / "cellot_progress.tsv"
        if lin and prog.exists():
            n, started = 0, False
            for ln in prog.read_text().splitlines():
                if not ln.startswith(lin + "\t"):
                    continue
                if "\tSTART" in ln:
                    n, started = 0, True
                elif started:
                    n += 1
            return (f"{lin} compounds", n)
    # STATE donor shards and per-task drivers count finished units in their own log
    if log.exists():
        try:
            txt = log.read_text(errors="replace")
        except OSError:
            txt = ""
        done = len(re.findall(r"donor \S+ done", txt))
        if done:
            return ("donors finished", done)
    # subprocess runners: the log the adapter tees is named with the run_obligation pid
    mine = _descendant_pids(job)
    if mine:
        best, bytes_ = None, 0
        for f in glob.glob(str(ROOT / "logs/runners" / "*.log")):
            m = re.search(r"_(\d+)\.log$", f)
            if m and int(m.group(1)) in mine:
                sz = os.path.getsize(f)
                if best is None or sz > bytes_:
                    best, bytes_ = f, sz
        if best is not None:
            return ("runner log bytes", bytes_)
    return ("job log bytes", log.stat().st_size if log.exists() else 0)


def gpu_busy() -> dict[int, int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return {int(a): int(b) for a, b in (l.split(", ") for l in out.strip().splitlines())}
    except Exception:
        return {}


def scan_new_bundles(since: float) -> list[str]:
    """Defects that a clean exit does not catch, checked on bundles written since the last run."""
    bad = []
    for p in glob.glob(str(DEPOSIT / "*.npz")):
        try:
            if os.path.getmtime(p) <= since:
                continue
            z = np.load(p, allow_pickle=True)
        except Exception:
            continue
        if "pred_means" not in z.files or "control_mean" not in z.files:
            continue
        P = np.atleast_2d(np.asarray(z["pred_means"], np.float64))
        C = np.asarray(z["control_mean"], np.float64).ravel()
        if P.shape[1] != C.shape[0]:
            continue
        name = os.path.basename(p)
        n = P.shape[0]
        if not np.isfinite(P).all():
            bad.append(f"{name}: prediction holds NaN or inf")
            continue
        same = int(sum(np.array_equal(P[i], C) for i in range(n)))
        if n and same == n:
            bad.append(f"{name}: every one of {n} held units is EXACTLY the control mean "
                       f"-- the cell is empty, not scored")
        elif n >= 4 and same > 0.5 * n:
            bad.append(f"{name}: {same}/{n} held units are exactly the control mean")
        if n >= 4:
            D = P - C
            keep = np.linalg.norm(D, axis=1) > 0
            if keep.sum() >= 4:
                u = len(np.unique(D[keep].round(9), axis=0))
                if u == 1:
                    bad.append(f"{name}: one profile repeated over {int(keep.sum())} held units "
                               f"-- degenerate, the same answer for every target")
    return bad


def main() -> int:
    prev = {}
    if STATE.exists():
        try:
            prev = json.loads(STATE.read_text())
        except Exception:
            prev = {}
    now = time.time()
    seen = prev.get("jobs", {})
    alarms, lines = [], []
    cur = {}
    for job in running_jobs():
        how, val = progress_signal(job)
        was = seen.get(job)
        if was and was["value"] == val:
            quiet = (now - was["since"]) / 60
            cur[job] = {"value": val, "since": was["since"], "how": how}
            limit = stall_minutes(job)
            if quiet >= limit:
                alarms.append(f"STALLED {job}: {how} stuck at {val} for {quiet:.0f} min "
                              f"(limit {limit}m)")
            else:
                lines.append(f"  {job}: {how}={val} (quiet {quiet:.0f}m)")
        else:
            cur[job] = {"value": val, "since": now, "how": how}
            lines.append(f"  {job}: {how}={val} advancing")
    util = gpu_busy()
    idle = [g for g, u in util.items() if u < 5]
    if idle and cur:
        lines.append(f"  GPUs under 5% SM: {idle}")
    bad = scan_new_bundles(prev.get("checked_at", now - 3600))
    alarms += [f"DEFECT {b}" for b in bad]
    STATE.write_text(json.dumps({"jobs": cur, "checked_at": now}, indent=1))
    for a in alarms:
        print(a, flush=True)
    if os.environ.get("IVCBENCH_HEALTH_VERBOSE"):
        print(f"[{time.strftime('%H:%M:%S')}] {len(cur)} running", flush=True)
        for l in lines:
            print(l, flush=True)
    return 1 if alarms else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # piping into head closes stdout early; that is not a health problem
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.exit(0)
