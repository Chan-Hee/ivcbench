"""GPU-free predictions->metrics reproduction wrapper: round-trip correctness."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from reproduce_eval import save_bundle, score_bundle

def test_mean_profile_pearson_roundtrip(tmp_path):
    rng = np.random.default_rng(0); G, S = 60, 8
    ctrl = rng.normal(0, 1, G); obs = ctrl + rng.normal(0, 0.5, (S, G))
    p = save_bundle(str(tmp_path/'perfect.npz'), pred_means=obs.copy(), obs_means=obs, control_mean=ctrl,
                    strata=[f's{i}' for i in range(S)], genes=[f'g{i}' for i in range(G)], cluster='T', model='perfect', split='d')
    c = save_bundle(str(tmp_path/'ctrl.npz'), pred_means=np.tile(ctrl,(S,1)), obs_means=obs, control_mean=ctrl,
                    strata=[f's{i}' for i in range(S)], genes=[f'g{i}' for i in range(G)], cluster='T', model='ctrl', split='d')
    assert abs(score_bundle(p)['pearson_delta'] - 1.0) < 1e-9
    assert abs(score_bundle(c)['pearson_delta']) < 1e-9

def test_percell_enables_energy_distance(tmp_path):
    rng = np.random.default_rng(1); G, S, n = 40, 4, 30
    ctrl = rng.normal(0, 1, G)
    strata = np.repeat(np.arange(S), n)
    obs = np.vstack([ctrl + rng.normal(s*0.3, 0.4, (n, G)) for s in range(S)])
    p = save_bundle(str(tmp_path/'pc.npz'), pred_cells=obs.copy(), test_cells=obs, cell_strata=strata,
                    control_mean=ctrl, strata=[str(s) for s in range(S)], genes=[f'g{i}' for i in range(G)],
                    pred_means=np.zeros((1,G)), obs_means=np.zeros((1,G)), cluster='T', model='perfect', split='d')
    r = score_bundle(p)
    assert abs(r['pearson_delta'] - 1.0) < 1e-9 and np.isfinite(r['e_distance'])  # per-cell -> e_distance finite

def test_bespoke_pattern_matches_direct_score(tmp_path):
    """The exact shape the heavy-model scripts dump: per-cell pred + exclude_gene_idx + a custom training-fold
    fit_on basis. A bundle scored by reproduce_eval must equal a direct pearson_delta/e_distance call to
    float32 precision — this is the static guarantee that each bespoke dump reproduces its deposited number."""
    from ivcbench.metrics.response import pearson_delta
    from ivcbench.metrics.distribution import e_distance
    rng = np.random.default_rng(2); G, S, n = 50, 5, 24
    ctrl = rng.normal(0, 1, G)
    strata = np.repeat(np.arange(S), n)
    train = ctrl + rng.normal(0, 1.0, (300, G))                     # the e_distance basis (ed_basis)
    obs = np.vstack([ctrl + rng.normal(0.5 + s * 0.2, 0.4, (n, G)) for s in range(S)])
    pred = obs + rng.normal(0, 0.15, obs.shape)                     # an imperfect predictor
    excl = np.array([3, 7, 11])                                     # leak-safe exclusions (rg / excl_idx)
    pe = pearson_delta(pred, obs, ctrl, strata, excl)['macro']      # exactly as a bespoke script scores
    ed = e_distance(pred, obs, strata, fit_on=train)['macro']
    p = save_bundle(str(tmp_path / 'bespoke.npz'), pred_cells=pred, test_cells=obs, cell_strata=strata,
                    control_mean=ctrl, genes=[f'g{i}' for i in range(G)], exclude_gene_idx=excl, fit_on=train,
                    cluster='C2', model='scPRAM-like', split='LODO')
    r = score_bundle(p)
    assert abs(r['pearson_delta'] - pe) / (abs(pe) + 1e-9) < 1e-4   # float32 deposit precision
    assert abs(r['e_distance'] - ed) / (abs(ed) + 1e-9) < 1e-4
