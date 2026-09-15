"""A prediction that is one profile repeated must score exactly zero — B3 regression.

Before the fix, stratum means were accumulated in float32. The mean of ONE profile repeated n
times then drifts from that profile (5.4e-5 at n=1032), lands outside the 1e-5 control-equality
guard, and scores as a real response: the Chen FHIP1A stratum read +0.0356 for seven models AND
for the ctrl-pred diagnostic, which is the control by construction and must read 0.
"""

import numpy as np
import pytest

from ivcbench.eval.bundle import _cells_to_means
from ivcbench.metrics.response import pearson_delta


@pytest.mark.parametrize("n", [60, 1032, 4096])
def test_repeated_control_has_no_response(n):
    rng = np.random.default_rng(0)
    ctrl = rng.normal(size=2000).astype(np.float32)
    pred = np.repeat(ctrl[None, :], n, axis=0)                       # the fallback: control, n times
    obs = (ctrl + rng.normal(scale=0.5, size=(n, 2000))).astype(np.float32)
    strata = np.array(["perturbation=FHIP1A"] * n)

    assert pearson_delta(pred, obs, ctrl, strata)["macro"] == 0.0

    means = _cells_to_means(
        dict(pred_cells=pred, test_cells=obs, cell_strata=strata,
             control_mean=ctrl, genes=np.arange(2000).astype(str))
    )
    # the DEPOSITED array must collapse back to the control exactly; no later re-score can undo it
    assert np.abs(means["pred_means"][0] - ctrl).max() == 0.0
