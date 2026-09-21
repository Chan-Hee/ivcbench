"""The immune-program analysis resolves its inputs from this archive, not from a frozen workspace.

The analysis itself is a `make check` gate, because it takes twenty seconds and re-scores 200
bundles. What belongs here is the one piece of logic the port added: the bundle manifest the
frozen workspace carried is rebuilt from the deposited census manifest. If that derivation ever
drifts, the gate would still pass on whatever subset it produced, so it is pinned separately.
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _program_inputs import EXPECTED_ROWS, TASKS, selected_manifest


def test_manifest_is_the_census_restricted_to_the_program_tasks():
    sel = selected_manifest(ROOT)
    census = pd.read_csv(ROOT / "results/_paper/census_bundle_manifest.csv", keep_default_na=False)
    want = census[census.task.isin(TASKS)].reset_index(drop=True)
    assert len(sel) == EXPECTED_ROWS == len(want)
    for col in want.columns:
        assert sel[col].tolist() == want[col].tolist(), col
    # the two columns the frozen workspace added named its own copy of each bundle; here the
    # archive's own copy is the bundle, so they must restate it rather than point elsewhere
    assert sel.snapshot_path.tolist() == sel.bundle_path.tolist()
    assert sel.snapshot_sha256.tolist() == sel.sha256.tolist()


def test_every_bundle_the_analysis_reads_is_in_the_archive():
    sel = selected_manifest(ROOT)
    missing = [p for p in sel.bundle_path if not (ROOT / p).is_file()]
    assert not missing, f"{len(missing)} bundle(s) absent from the archive, e.g. {missing[:3]}"


def test_the_metadata_inputs_are_read_from_the_archive_itself():
    for rel in ("src/ivcbench/clusters/c3.py", "src/ivcbench/clusters/c5.py",
                "results/_paper/census_uncertainty.csv", "results/_paper/census_unit_scores.csv"):
        assert (ROOT / rel).is_file(), rel
    src = (ROOT / "scripts/program_analysis.py").read_text(encoding="utf-8")
    # the port must not have left a read pointing at the frozen workspace
    assert "inputs/metadata" not in src
    assert 'BASE / "src/ivcbench/clusters"' in src
