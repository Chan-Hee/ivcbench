"""STATE's T2 runner must reach the held donor's own basal state, and must not be scored in a
coordinate half of which its output cannot represent — S4 regression.

Two independent defects, both traced by the audit:
  * the held donor's own 0h cells were added ONLY under the held stim label, so state's random
    basal mapper -- which draws from the CONTROL-labelled pool of the same cell type -- never saw
    them; the basal cell came from other donors and the held donor contributed nothing.
  * the Soskic coordinate is signed (52% of observed values are negative) while state's prediction
    path ends in a ReLU and a [0, 14] clip, so half the target space is unreachable by construction.
"""

from pathlib import Path

import numpy as np

SRC = (Path(__file__).resolve().parents[1] / "model_runners" / "state_soskic_runner.py").read_text()


def test_held_cells_are_also_emitted_as_controls():
    # the same X_ctrl_inf block must be appended twice: once under the held label, once as control
    assert SRC.count("blk_X.append(X_ctrl_inf[idx])") == 2, (
        "the held donor's own cells must be added under the held stim label AND as `control`, "
        "or state's basal mapper cannot draw them"
    )
    assert 'blk_pert.append(np.array(["control"] * len(idx), dtype=object))' in SRC


def test_the_coordinate_shift_is_applied_and_inverted():
    """The shift is OPT-IN now, and whichever way it is set it must be undone before returning.

    It used to be unconditional; the same commit that added this test made it opt-in behind
    IVCBENCH_STATE_BOXMAP (default "none") after measuring that it changed nothing. So asserting
    that it is applied unconditionally asserts the opposite of what the runner does. What still
    has to hold is that the default is off, and that any shift applied is inverted on the way out
    -- an applied-but-not-removed shift would move every returned profile.
    """
    assert 'os.environ.get("IVCBENCH_STATE_BOXMAP", "none")' in SRC, "the default must be off"
    assert "STATE_SHIFT, STATE_SCALE = 0.0, 1.0" in SRC, "the off case must be the identity"
    assert "Xall = (Xall + STATE_SHIFT) * STATE_SCALE" in SRC
    assert "/ STATE_SCALE - STATE_SHIFT" in SRC, (
        "the shift must be removed before the profiles are returned"
    )


def test_a_constant_shift_cannot_change_the_metric():
    """Whatever the shift is, Δ = pred − control is unchanged, so the score is untouched."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ivcbench.metrics.response import pearson_delta

    rng = np.random.default_rng(0)
    ctrl = rng.normal(size=200).astype(np.float32)
    pred = rng.normal(size=(30, 200)).astype(np.float32)
    obs = rng.normal(size=(30, 200)).astype(np.float32)
    strata = np.array(["cell_type_coarse=CD4_Naive"] * 30)
    c = float(max(0.0, -min(ctrl.min(), pred.min())))
    assert c > 0
    a = pearson_delta(pred, obs, ctrl, strata)["macro"]
    b = pearson_delta(
        (pred + np.float32(c)).astype(np.float32),
        obs,
        (ctrl + np.float32(c)).astype(np.float32),
        strata,
    )["macro"]
    # Exact in real arithmetic; in float32 the translation costs one rounding. Measured at the
    # real T2 scale this is ~5e-10 on the score. The bound below is deliberately far tighter than
    # B3's damage (5.4e-5 absolute -> +0.036 on the score) and than the 1e-5 control-equality
    # guard, so this translation can never manufacture a response the way B3's aggregation did.
    assert abs(a - b) < 1e-8, abs(a - b)
