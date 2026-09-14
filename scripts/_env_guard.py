"""Refuse to run an ivcbench entry point against the stale package copy.

Two trees exist: ivcbench/ (current; 42 runners, STATE output-selection fix) and benchmark/ (kept
as historical evidence; 20 runners, no fix). benchmark/.venv installs ivcbench editable from
benchmark/src, so `benchmark/.venv/bin/python ivcbench/scripts/run_cluster.py` silently runs the
HISTORICAL runners: STATE recovers adata_real.h5ad and reproduces the archived numbers exactly, and
the newer runners are simply absent. A full round of GPU jobs was spent before that was noticed.

Import this first in any script under ivcbench/scripts/.
"""
from pathlib import Path


def assert_current_package():
    import ivcbench

    want = Path(__file__).resolve().parents[1] / "src" / "ivcbench"
    got = Path(ivcbench.__file__).resolve().parent
    if got != want:
        raise SystemExit(
            f"Wrong ivcbench package.\n"
            f"  imported : {got}\n"
            f"  expected : {want}\n"
            f"Run with ivcbench/.venv/bin/python (benchmark/.venv resolves the historical copy)."
        )


assert_current_package()
