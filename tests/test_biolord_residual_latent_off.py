"""biolord's untrained residual latent must be switched off on every task — K3 regression.

biolord/_module.py builds the residual ("unknown attribute") latent as
    RegularizedEmbedding(n_input=n_samples, ...)  ->  nn.Embedding, ONE ROW PER CELL
and `forward` ends in `x_ = x_ * self.embed`. A held unit's cells are the `ood` split and never
receive a gradient, so with the default unknown_attributes=True their row is still the N(0, 1)
initialisation at prediction time: the held identity contributes nothing and a random vector of
norm ~sqrt(n_latent) contributes a lot, seed-dependently.
"""

from pathlib import Path

import pytest

RUNNERS = Path(__file__).resolve().parents[1] / "model_runners"
BIOLORD = ["biolord_c1_runner.py", "biolord_c3_runner.py", "biolord_c5_runner.py"]


@pytest.mark.parametrize("name", BIOLORD)
def test_every_biolord_runner_switches_the_residual_off(name):
    src = (RUNNERS / name).read_text()
    assert '"unknown_attributes"' in src, f"{name} leaves biolord's default (True) in place"
    # the value must be False, not merely mentioned
    i = src.index('"unknown_attributes"')
    assert "False" in src[i : i + 200], f"{name} names the flag but does not set it False"


def test_the_upstream_default_is_still_true():
    """If upstream ever flips the default, these runners no longer need the override --
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
