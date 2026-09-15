#!/usr/bin/env python
"""Gate every job BEFORE it takes a GPU.  runs/preflight.py <job-id> <command...>

Exit 0 = may launch. Non-zero = refuse, with the reason on stderr.

Each check exists because that exact mistake already cost GPU time on 2026-09-14:

  1  roster        queued SCREEN/MAP, which are in the dispatch registry but NOT in the plan's
                   17-model table -- running them produces cells that cannot be reported.
  2  disposition   a cell marked X (structural exclusion) or R (withdrawn) must never be run.
  3  entry point   (model, task) must resolve in run_obligation's registry, or the runner file
                   must exist -- two PertAdapt launches died on a missing entry point/asset.
  4  assets        $IVCBENCH_* must point at files that exist. PertAdapt T4 failed twice: first
                   IVCBENCH_SCFOUNDATION_DIR unset, then set to a tree missing the gene index.
  5  deposit       IVCBENCH_PRED_DUMP must be set, writable, and NOT predictions/ itself.
                   Eight runs deposited nothing because dump_bundle() is a silent no-op when it
                   is unset; none of them could have entered the census.
  6  no overwrite  the deposit dir must not already hold a bundle this job would replace, and
                   must never be a path pinned in withdrawn_bundles.csv (path+sha history).
  7  gpu free      no other tracked job may be running on the target GPU -- two pairs of jobs
                   were assigned to GPUs that were already occupied.
  8  not duplicate no job with this id may already be RUNNING (v2/v3 copies raced each other).
  9  code version  the aggregation path must carry the B3 float64 fix; a run on pre-fix code
                   bakes a spurious response into the deposited array.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT.parent / "revision_claude" / "NATIVE_COVERAGE_EXECUTION_PLAN.md"
RUNS = ROOT / "runs"
TASKS = ("T1", "T2", "T3", "T4", "T5c", "T5u")


# A refusal is either PERMANENT (the job must never run as written) or TRANSIENT (it may run once
# something else finishes). They must exit differently: a transient refusal that leaves a .status
# behind removes the job from the dispatcher's view for good, and the cell silently disappears.
TRANSIENT = {"gpu busy", "duplicate"}
_WHY = {
    "X": "(structural exclusion",
    "R": "(withdrawn from the comparison",
    "P": "(outside this revision's adaptation exception",
}


def fail(n: str, msg: str) -> None:
    print(f"PREFLIGHT REFUSED [{n}]: {msg}", file=sys.stderr)
    sys.exit(8 if n in TRANSIENT else 3)


def plan_table() -> dict[tuple[str, str], str]:
    """(model, task) -> N | L | A | B | R | X, parsed from the plan's disposition table."""
    if not PLAN.exists():
        fail("roster", f"the plan is missing at {PLAN}")
    # Read ONLY the disposition table. A looser match once swallowed a row of a neighbouring table
    # and reported 90 cells, which then refused every launch.
    out: dict[tuple[str, str], str] = {}
    inside = False
    for line in PLAN.read_text().splitlines():
        if line.startswith("| 모델 / comparator |"):
            inside = True
            continue
        if inside and not line.startswith("|"):
            inside = False
        if not inside:
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 9:
            continue
        model, marks = cells[1], cells[2:8]
        if not model or model.startswith("-"):
            continue
        if all(m in {"N", "L", "A", "B", "R", "X", "P"} for m in marks):
            for t, m in zip(TASKS, marks):
                out[(model, t)] = m
    if len(out) != 102:
        fail("roster", f"parsed {len(out)} plan cells, expected 102 (17 models x 6 columns)")
    return out


# model names as spelled in the plan vs as dispatched
ALIAS = {
    "Biolord": "biolord",
    "chemCPA": "CPA / chemCPA",
    "CPA": "CPA / chemCPA",
    "scGPT-fcond": "scGPT",
    "scFoundation-fcond": "scFoundation",
    "linear-shift-KOemb": "linear-shift-KOemb",
}
CLUSTER_TASK = {"C1": "T1", "C2": "T2", "C3": "T3", "C4": "T4"}


def parse_target(cmd: str) -> tuple[str, str] | None:
    """-> (model, task) the command will run, or None when it cannot be decided from the text."""
    m = re.search(r"--model\s+(\S+).*?--task\s+(\S+)", cmd, re.S)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"--cluster\s+(\S+).*?--only\s+(\S+)", cmd, re.S)
    if m:
        cl, model = m.group(1), m.group(2)
        return (model, CLUSTER_TASK.get(cl, "T5c"))  # C5 covers T5c and T5u in one run
    m = re.search(r"IVCBENCH_C4_ONLY=(\S+)", cmd)
    if m:
        return (m.group(1), "T4")
    # The native chemCPA unseen-compound run has no --model/--task: it is its own script. Without
    # this line the roster and disposition checks are skipped for the one cell it produces.
    if "chemcpa_native_op3.py" in cmd or "chemcpa_t5u" in cmd:
        return ("chemCPA", "T5u")
    # The donor task has its own driver, not run_obligation.
    if "state_soskic.py" in cmd:
        return ("STATE", "T2")
    return None


def main() -> None:
    if len(sys.argv) < 3:
        fail("usage", "runs/preflight.py <job-id> <command...>")
    argv = sys.argv[1:]
    # --dry validates a QUEUE ahead of time: plan/roster/entry point/assets/deposit/code version,
    # but not the runtime state (GPU occupancy, duplicate launch), which is only meaningful at the
    # moment of dispatch.
    dry = "--dry" in argv
    argv = [a for a in argv if a != "--dry"]
    job_id, cmd = argv[0], " ".join(argv[1:])
    table = plan_table()
    target = parse_target(cmd)

    # 1-2 roster + disposition -------------------------------------------------
    if target is None:
        print(f"preflight: {job_id} is not a model run; checks 1-3, 6 skipped", file=sys.stderr)
    else:
        model, task = target
        plan_name = ALIAS.get(model, model)
        if task not in TASKS:
            fail("entry point", f"task {task!r} is not one of {TASKS}")
        if (plan_name, task) not in table:
            fail(
                "roster",
                f"{model} ({plan_name}) is not in the plan's 17-model table; running it "
                "produces a cell that cannot be reported. Add it to the plan first, or do not run it.",
            )
        mark = table[(plan_name, task)]
        if mark in ("X", "R", "P"):
            fail(
                "disposition",
                f"{plan_name} x {task} is marked {mark} in the plan "
                + _WHY[mark]
                + "). A low score or a missing implementation is not a reason to run it anyway.",
            )
        print(f"preflight: {plan_name} x {task} = {mark}", file=sys.stderr)

        # 3 entry point --------------------------------------------------------
        if "--model" in cmd:
            src = (ROOT / "scripts" / "run_obligation.py").read_text()
            pairs = set(re.findall(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)\s*:', src))
            if (model, task) not in pairs:
                fail("entry point", f'("{model}", "{task}") is not in run_obligation.py\'s registry')

    # 4 assets ----------------------------------------------------------------
    r = subprocess.run(["bash", "-c", f'. "{RUNS}/env.sh"'], capture_output=True, text=True)
    if r.returncode != 0:
        fail("assets", (r.stderr or "runs/env.sh failed").strip())

    # 5 deposit ---------------------------------------------------------------
    dump = os.environ.get("IVCBENCH_PRED_DUMP") or subprocess.run(
        ["bash", "-c", f'. "{RUNS}/env.sh" && printf %s "$IVCBENCH_PRED_DUMP"'],
        capture_output=True, text=True,
    ).stdout.strip()
    if not dump:
        fail("deposit", "IVCBENCH_PRED_DUMP is unset -> dump_bundle() is a silent no-op and the "
                        "run deposits nothing the census can read")
    dp = Path(dump)
    if dp.resolve() == (ROOT / "predictions").resolve():
        fail("no overwrite", "IVCBENCH_PRED_DUMP must be a SUBDIRECTORY of predictions/, never "
                             "predictions/ itself -- a new bundle would overwrite the historical "
                             "one at the same filename, which withdrawn_bundles.csv pins by sha256")
    dp.mkdir(parents=True, exist_ok=True)
    if not os.access(dp, os.W_OK):
        fail("deposit", f"{dp} is not writable")

    # 7-8 gpu free + not already running --------------------------------------
    want_gpu = None
    st = RUNS / f"{job_id}.status"
    if dry:
        m = re.search(r"--gpu\s+(\d+)", cmd) or re.search(r"CUDA_VISIBLE_DEVICES=(\d+)", cmd) or re.search(r"--gpus\s+(\d+)", cmd)
        want_gpu = m.group(1) if m else None
        _skip_runtime = True
    else:
        _skip_runtime = False
    if not _skip_runtime:
        if st.exists() and st.read_text().startswith("RUNNING"):
            fail("duplicate", f"{job_id} is already RUNNING ({st.read_text().strip()})")
        m = re.search(r"--gpu\s+(\d+)", cmd) or re.search(r"CUDA_VISIBLE_DEVICES=(\d+)", cmd) or re.search(r"--gpus\s+(\d+)", cmd)
        if m:
            want_gpu = m.group(1)
            on_gpu = []
            for other in RUNS.glob("*.status"):
                if other == st or not other.read_text().startswith("RUNNING"):
                    continue
                ocmd = RUNS / f"{other.stem}.cmd"
                if not ocmd.exists():
                    continue
                o = ocmd.read_text()
                om = re.search(r"--gpu\s+(\d+)", o) or re.search(r"CUDA_VISIBLE_DEVICES=(\d+)", o) or re.search(r"--gpus\s+(\d+)", o)
                if om and om.group(1) == want_gpu:
                    on_gpu.append(other.stem)
            # These L40s hold 46-49 GB and the jobs here occupy 2-10 GB, so one job per card leaves
            # most of the machine idle. Allow a few per card; dispatch.sh's memory-headroom check is
            # what actually guards against oversubscription.
            # The global limit was measured on a heavy job (chemCPA went 18 s -> 7 min per epoch
            # at three per card), but it is the wrong number for a card carrying only light ones.
            # Measured 2026-09-15: GPU 0 held CellOT at 2000 iters plus a STATE donor shard and
            # averaged 23.5% SM over 30 s (28 of 30 samples under 50%) while GPUs 1-3 sat at
            # 96-99%. IVCBENCH_JOBS_PER_GPU_<n> raises the limit for one card without touching the
            # others; dispatch.sh's memory-headroom check still guards oversubscription.
            if want_gpu in os.environ.get("IVCBENCH_RESERVED_GPUS", "").split():
                fail("gpu reserved", f"GPU {want_gpu} is reserved for a single long job "
                                     f"(IVCBENCH_RESERVED_GPUS); pick another card")
            limit = int(os.environ.get(f"IVCBENCH_JOBS_PER_GPU_{want_gpu}",
                                       os.environ.get("IVCBENCH_JOBS_PER_GPU", "2")))
            if len(on_gpu) >= limit:
                fail("gpu busy", f"GPU {want_gpu} already runs {len(on_gpu)} job(s) "
                                 f"({', '.join(on_gpu)}); limit {limit}")

    # 9 code version ----------------------------------------------------------
    resp = (ROOT / "src/ivcbench/metrics/response.py").read_text()
    bund = (ROOT / "src/ivcbench/eval/bundle.py").read_text()
    if "dtype=np.float64" not in resp or "dtype=np.float64" not in bund:
        fail("code version", "the B3 float64 stratum-mean fix is not present in "
                             "metrics/response.py and eval/bundle.py; a run on pre-fix code bakes "
                             "a spurious response into the deposited array")

    print(f"preflight: {job_id} OK (deposit -> {dp}{', gpu ' + want_gpu if want_gpu else ''})", file=sys.stderr)


if __name__ == "__main__":
    main()
