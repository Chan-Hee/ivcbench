"""biolord's untrained residual latent is switched off on the GENETIC path only — K3 regression.

biolord/_module.py builds the residual ("unknown attribute") latent as
    RegularizedEmbedding(n_input=n_samples, ...)  ->  nn.Embedding, ONE ROW PER CELL
and `forward` ends in `x_ = x_ * self.embed`. A held unit's cells are the `ood` split and never
receive a gradient, so with the default unknown_attributes=True their row is still the N(0, 1)
initialisation at prediction time: the held identity contributes nothing and a random vector of
norm ~sqrt(n_latent) contributes a lot, seed-dependently.

This file used to require the override on EVERY task. The ruling went the other way: switching
that latent off to rescue a context cell runs a different method, so it is set only where the
biolord authors themselves set it -- the genetic configuration, adamson_config_optimal.py:25,
which the C3 runner serves for T3 and T4. On the cell-context and seen-compound tasks the default
stands and the cell is declared not-reported instead, with a reason in Supplementary Table S15b.
The runners were edited to comply and this test was the last artefact still asserting the old
answer.

So the expectation is derived from the census and S15b rather than typed here: re-admitting
biolord on T1 fails because S15b no longer excuses it, and a new runner fails the task map.
"""

import sys
from pathlib import Path

import csv
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ROOT / "model_runners"
sys.path.insert(0, str(ROOT / "scripts"))

# Which census cells each runner serves. Globbed against the directory below, so a runner added
# without an entry here fails rather than going unchecked.
SERVES = {
    "biolord_c1_runner.py": [("T1", "cell-context (LOCT)"), ("T2", "donor (LODO)")],
    "biolord_c3_runner.py": [("T3", "unseen-gene"), ("T4", "unseen-knockout")],
    "biolord_c5_runner.py": [("T5", "cell-context (LOCT)"), ("T5", "unseen-compound")],
}
# The one path where the flag belongs: the authors' own genetic configuration sets it.
OFF_ON = "biolord_c3_runner.py"


def _census_cells():
    from assemble_cross_cluster import census_metadata_rows

    return {(r["task_id"], r["split"]) for r in census_metadata_rows()
            if r["model"].lower() == "biolord"}


def test_the_runner_roster_is_complete():
    found = {p.name for p in RUNNERS.glob("biolord_*_runner.py")}
    assert found == set(SERVES), f"runner roster changed: {found ^ set(SERVES)}"


def test_the_genetic_runner_switches_the_residual_off():
    src = (RUNNERS / OFF_ON).read_text()
    assert '"unknown_attributes"' in src, f"{OFF_ON} leaves biolord's default (True) in place"
    i = src.index('"unknown_attributes"')
    assert "False" in src[i : i + 200], f"{OFF_ON} names the flag but does not set it False"
    assert "adamson_config_optimal" in src, (
        "the override is allowed here because the biolord authors set it in the genetic "
        "configuration; the runner must cite that, or it is our choice and not theirs"
    )


@pytest.mark.parametrize("name", [n for n in SERVES if n != OFF_ON])
def test_the_other_runners_leave_it_at_biolord_s_default(name):
    src = (RUNNERS / name).read_text()
    assert '"unknown_attributes"' not in src, (
        f"{name} disables biolord's residual latent. Outside the genetic configuration that "
        "runs a different method; the ruling declines the cell instead."
    )


@pytest.mark.parametrize("name", [n for n in SERVES if n != OFF_ON])
def test_what_those_runners_cannot_serve_is_declared_not_reported(name):
    """Declining a cell is only honest if the paper says so. Every task these runners serve that
    the census does not report must carry a reason in Supplementary Table S15b."""
    reported = _census_cells()
    with (ROOT / "results/_paper/supplementary_tables/Supplementary_Table_S15b.csv").open() as fh:
        excused = {r["Task"] for r in csv.DictReader(fh) if r["Model"].lower() == "biolord"}
    for task_id, split in SERVES[name]:
        if (task_id, split) in reported:
            continue
        key = "T5c" if (task_id, split) == ("T5", "cell-context (LOCT)") else task_id
        assert key in excused, (
            f"biolord {key} is neither reported by the census nor excused in Table S15b"
        )


def test_the_upstream_default_is_still_true():
    """If upstream ever flips the default, the genetic runner no longer needs the override --
    but we must find out by this test failing, not by silently relying on it."""
    mod = Path(
        "/data1/home/chlee/miniconda3/envs/ctx-biolord/lib/python3.9/site-packages/biolord/_module.py"
    )
    if not mod.exists():
        pytest.skip("ctx-biolord env is not present")
    src = mod.read_text()
    assert "unknown_attributes: bool = True" in src
    assert "x_ = x_ * self.embed" in src        # how False switches it off
    assert "n_input=n_samples" in src           # why a held cell's row is untrained
