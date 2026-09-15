"""Which (model, task) a launch command produces — ONE copy of the mapping.

There were three: runs/preflight.py decided what a job was allowed to run, scripts/
build_run_manifest.py decided whether a cell had been run, and scripts/append_run_ledger.py
decided what it cost. They drifted, and each drift is silent in a different way:

  * preflight knew scripts/state_soskic.py, the manifest did not -- so eight completed, deposited,
    scored STATE T2 shards produced no manifest row and the cell read as unrun;
  * preflight knew the chemCPA wrapper, the ledger did not -- so a 2 h re-training did not appear
    in the compute table;
  * the manifest learned both, and the ledger still had neither.

Anything that adds an entry point adds it HERE.
"""
from __future__ import annotations

import re

CLUSTER_TASK = {"C1": "T1", "C2": "T2", "C3": "T3", "C4": "T4"}

# a per-task driver script -> the cell it produces
DRIVERS = {
    "state_soskic.py": ("STATE", "T2"),
    "scpram_soskic.py": ("scPRAM", "T2"),
    "cpa_soskic.py": ("CPA", "T2"),
    "pertadapt_soskic.py": ("PertAdapt", "T2"),
    # foundation_c2_donor.py is NOT one cell: it takes --model scGPT|scFoundation. Mapping the
    # script to a single model recorded every scGPT T2 run as scFoundation T2. Handled below.
    "foundation_c2_donor.py": (None, "T2"),
    "state_kang.py": ("STATE", "T1"),
    "cellot_kang.py": ("CellOT", "T1"),
    "scpram_kang.py": ("scPRAM", "T1"),
    "scpram_frangieh.py": ("scPRAM", "T4"),
    "cellot_frangieh.py": ("CellOT", "T4"),
    "chemcpa_native_op3.py": ("chemCPA", "T5u"),
    "chemcpa_t5u": ("chemCPA", "T5u"),          # the wrapper shell script
}


def resolve(cmd: str) -> tuple[str, str] | None:
    """-> (model, task), or None when the command does not name a cell."""
    m = re.search(r"--model\s+(\S+).*?--task\s+(\S+)", cmd, re.S)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"--cluster\s+(\S+).*?--only\s+(\S+)", cmd, re.S)
    if m:
        # C5 covers T5c and T5u in one run; callers that can tell them apart from the result rows
        # should do so, this is the coarse answer.
        return m.group(2), CLUSTER_TASK.get(m.group(1), "T5c")
    m = re.search(r"IVCBENCH_C4_ONLY=(\S+)", cmd)
    if m:
        return m.group(1), "T4"
    for needle, cell in DRIVERS.items():
        if needle not in cmd:
            continue
        model, task = cell
        if model is None:                      # multi-model driver: the model is an argument
            m = re.search(r"--model\s+(\S+)", cmd)
            if not m:
                return None
            return (m.group(1), task)
        return cell
    return None


def shard_of(cmd: str) -> tuple[int, int] | None:
    """-> (i, n) when the command runs shard i of n, else None."""
    m = re.search(r"--chunk\s+(\d+)\s+(\d+)", cmd)
    return (int(m.group(1)), int(m.group(2))) if m else None
