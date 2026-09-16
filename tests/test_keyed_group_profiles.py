"""A held-group runner's keyed profiles must reach their own strata — S3 regression.

PertAdapt (T2) returns one profile per '<donor>::<lineage>' and STATE (T1) one per
'stim::<donor>'. The generic adapter averaged every returned profile and tiled the blend over all
test rows, so neither profile survived and the model was scored on something it never predicted.
"""

import numpy as np
import pytest

from ivcbench.baselines.heavy import SubprocessAdapter


class _Fake(SubprocessAdapter):
    name, pred_key_is_group = "Fake", True

    def __init__(self, returned):
        self._returned = returned
        self.ctrl = np.zeros(4, dtype=np.float32)

    def _invoke(self, cs, split, side_info, test_perts_override=None):
        return self._returned


class _Split:
    def __init__(self, strata):
        self.test_strata = np.asarray(strata)
        self.test_idx = np.arange(len(strata))


class _CS:
    def __init__(self, strata):
        import pandas as pd

        self.obs = pd.DataFrame({"perturbation": ["stim"] * len(strata)})


def _run(returned, strata):
    sp = _Split(strata)
    return _Fake(returned).predict(_CS(strata), sp)


def test_each_stratum_gets_its_own_profile():
    a = np.array([1, 1, 1, 1], dtype=np.float32)
    b = np.array([9, 9, 9, 9], dtype=np.float32)
    r = _run({"stim::1256": a, "stim::101": b}, ["donor_id=1256", "donor_id=101", "donor_id=1256"])
    assert np.array_equal(r.pred_cells[0], a)
    assert np.array_equal(r.pred_cells[1], b)
    assert np.array_equal(r.pred_cells[2], a)
    assert not r.declined.any()
    # the old behaviour would have put the blend (5,5,5,5) in every row
    assert not np.allclose(r.pred_cells[0], (a + b) / 2)


def test_lineage_keys_work_too():
    a = np.array([2, 2, 2, 2], dtype=np.float32)
    b = np.array([4, 4, 4, 4], dtype=np.float32)
    r = _run(
        {"D348::CD4_Naive": a, "D348::CD4_Memory": b},
        ["cell_type_coarse=CD4_Naive", "cell_type_coarse=CD4_Memory"],
    )
    assert np.array_equal(r.pred_cells[0], a) and np.array_equal(r.pred_cells[1], b)


def test_a_single_profile_is_still_tiled():
    a = np.array([3, 3, 3, 3], dtype=np.float32)
    r = _run({"stim::whole-group": a}, ["donor_id=1", "donor_id=2"])
    assert np.array_equal(r.pred_cells[0], a) and np.array_equal(r.pred_cells[1], a)
    assert not r.declined.any()


def test_an_unmatched_stratum_is_declined_not_blended():
    a = np.array([1, 1, 1, 1], dtype=np.float32)
    b = np.array([9, 9, 9, 9], dtype=np.float32)
    r = _run({"stim::1256": a, "stim::101": b}, ["donor_id=1256", "donor_id=999"])
    assert np.array_equal(r.pred_cells[0], a)
    assert r.declined.tolist() == [False, True]
    assert np.array_equal(r.pred_cells[1], np.zeros(4, dtype=np.float32))  # the control


def test_no_key_matches_at_all_is_an_error_not_a_blend():
    a = np.array([1, 1, 1, 1], dtype=np.float32)
    b = np.array([9, 9, 9, 9], dtype=np.float32)
    with pytest.raises(RuntimeError, match="never predicted"):
        _run({"stim::aaa": a, "stim::bbb": b}, ["donor_id=1", "donor_id=2"])


def test_one_profile_over_many_compounds_is_still_refused():
    """The guard's real target: a pooled map reported per compound.

    Tiling is permitted on a held-GROUP split because its strata are a nuisance axis and the
    held unit is the whole prediction. When the strata are the perturbation axis they ARE the
    held entity, so one profile means the rest went unpredicted.
    """
    a = np.array([5, 5, 5, 5], dtype=np.float32)
    with pytest.raises(RuntimeError, match="never made"):
        _run({"loct::whole-group": a}, ["perturbation=Atorvastatin", "perturbation=Vorinostat"])


def test_an_unrecognised_stratum_axis_is_refused_rather_than_assumed():
    a = np.array([5, 5, 5, 5], dtype=np.float32)
    with pytest.raises(RuntimeError, match="never made"):
        _run({"x::whole-group": a}, ["something_new=1", "something_new=2"])
