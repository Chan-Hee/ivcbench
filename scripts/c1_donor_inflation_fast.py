#!/usr/bin/env python
"""FAST seed-aware bootstrap of the Kang donor random-split inflation (reqs 5, 10).

Lighter than c1_donor_inflation_seeds.py: computes ONLY pearson_delta per (baseline, fold, seed) via
build_split + baseline.fit/predict + the framework's pearson_delta — skips e_distance, bootstrap CIs and
program corrs (the heavy parts of run_job that made the 20-seed run slow). Writes the CSV incrementally
after every seed so partial results are never lost. LODO folds are fixed; only the random-cell folds
vary by seed. Also records each held-donor's dominant lineage and type-I IFN AUCell-Δ for stratification.
CPU only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivcbench.baselines.simple import SIMPLE_BASELINES  # noqa: E402
from ivcbench.clusters import c1  # noqa: E402
from ivcbench.data.loaders import kang as kang_mod  # noqa: E402
from ivcbench.splits.builder import build_split  # noqa: E402
from ivcbench.splits.audit import audit_split  # noqa: E402
from ivcbench.metrics.response import pearson_delta  # noqa: E402

N_SEEDS = 12
IFN_GENES = ["ISG15", "IFI6", "MX1", "MX2", "OAS1", "OAS2", "IFIT1", "IFIT3", "ISG20",
             "STAT1", "IRF7", "IFI44", "IFI44L", "RSAD2", "USP18"]
OUT = ROOT / "results/C1/donor_inflation_seeds.csv"


def assign_rand_fold(cs, donors, seed):
    is_ctrl = cs.obs["is_control"].astype(bool)
    rng = np.random.default_rng(seed)
    rand_fold = np.array(["__none__"] * cs.n_cells, dtype=object)
    treated_idx = np.where(~is_ctrl.to_numpy())[0]
    ctrl_idx = np.where(is_ctrl.to_numpy())[0]
    perm_t = rng.permutation(treated_idx)
    perm_c = rng.permutation(ctrl_idx)
    ti = ci = 0
    for d in donors:
        nt = int(((cs.obs["donor_id"].astype(str) == d) & ~is_ctrl).sum())
        nc = max(1, len(ctrl_idx) // len(donors))
        fold = f"f{d}"
        rand_fold[perm_t[ti:ti + nt]] = fold; ti += nt
        rand_fold[perm_c[ci:ci + nc]] = fold; ci += nc
    return rand_fold


def score(cs, spec):
    """pearson_delta per simple baseline for one split spec (no e_dist/bootstrap)."""
    sp = build_split(cs, spec)
    audit = audit_split(cs, sp)
    test_X = cs.X[sp.test_idx]
    out = {}
    for B in SIMPLE_BASELINES:
        b = B()
        b.fit(cs, sp, side_info=cs.side_info)
        pred = b.predict(cs, sp, side_info=cs.side_info)
        resp = pearson_delta(pred.pred_cells, test_X, pred.control_mean, sp.test_strata, None)
        out[b.name] = (float(resp["macro"]), bool(audit["leak_free"]))
    return out


def main():
    cs = kang_mod.load()
    is_ctrl = cs.obs["is_control"].astype(bool)
    donors = sorted(cs.obs["donor_id"].astype(str).unique())
    donors = [d for d in donors
              if int(((cs.obs["donor_id"].astype(str) == d) & ~is_ctrl).sum()) >= 50]
    print("donors:", donors, flush=True)

    ct = cs.obs["cell_type_coarse"].astype(str)
    ifn_idx = np.asarray(cs.gene_index(IFN_GENES), dtype=int)
    X = cs.X

    def mean_ifn(mask):
        return float(np.asarray(X[mask][:, ifn_idx].mean())) if len(ifn_idx) else np.nan

    ctrl_mean_ifn = mean_ifn(is_ctrl.to_numpy())
    donor_lineage, donor_ifn = {}, {}
    for d in donors:
        m = (cs.obs["donor_id"].astype(str) == d).to_numpy()
        tm = m & ~is_ctrl.to_numpy()
        donor_lineage[d] = ct[tm].value_counts().idxmax()
        donor_ifn[d] = mean_ifn(tm) - ctrl_mean_ifn

    # fixed LODO
    lodo_vals = {}
    for d in donors:
        for bn, (v, lf) in score(cs, c1.donor_lodo(d)).items():
            lodo_vals[(bn, d)] = v
    print("LODO done", flush=True)

    rows = []
    for seed in range(N_SEEDS):
        cs.obs["_rand_fold"] = assign_rand_fold(cs, donors, seed)
        for d in donors:
            for bn, (v, lf) in score(cs, c1.random_cell_split(f"f{d}")).items():
                lo = lodo_vals.get((bn, d))
                if lo is None:
                    continue
                rows.append(dict(seed=seed, donor=d, baseline=bn, lineage=donor_lineage[d],
                                 ifn_delta=donor_ifn[d], rand=v, lodo=lo, infl=v - lo,
                                 leak_free=lf))
        pd.DataFrame(rows).to_csv(OUT, index=False)   # incremental write
        ps = pd.DataFrame(rows).query("seed==@seed and baseline!='ctrl-pred'").infl.mean()
        print(f"seed {seed} done (mean infl excl ctrl-pred = {ps:+.4f})", flush=True)

    df = pd.DataFrame(rows)
    sub = df[df.baseline != "ctrl-pred"]
    per_seed = sub.groupby("seed").infl.mean()
    print(f"\n=== {N_SEEDS} seeds x {len(donors)} donors (4 simple baselines; ctrl-pred excluded from grand mean) ===")
    print(f"per-seed mean inflation: mean={per_seed.mean():.4f}, sd={per_seed.std(ddof=1):.4f}, "
          f"min={per_seed.min():.4f}, max={per_seed.max():.4f}; positive seeds {int((per_seed>0).sum())}/{len(per_seed)}")
    rng = np.random.default_rng(0)
    boot = [per_seed.sample(len(per_seed), replace=True, random_state=int(s)).mean()
            for s in rng.integers(0, 1e6, 10000)]
    print(f"bootstrap-over-seeds grand-mean 95% CI [{np.percentile(boot,2.5):.4f}, {np.percentile(boot,97.5):.4f}]")
    bt = stats.binomtest(int((per_seed>0).sum()), len(per_seed), 0.5, alternative="greater")
    print(f"sign test per-seed means: {int((per_seed>0).sum())}/{len(per_seed)} positive, p={bt.pvalue:.2e}")
    print("\nper-baseline mean inflation:")
    print(df.groupby("baseline").infl.agg(["mean", "std", "count"]).round(4).to_string())
    print("\ninflation by held-donor dominant lineage (excl ctrl-pred):")
    print(sub.groupby("lineage").agg(infl=("infl","mean"), ifn=("ifn_delta","mean"), n=("infl","size")).round(4).to_string())
    g = sub.groupby("donor").agg(infl=("infl","mean"), ifn=("ifn_delta","mean"))
    if g.ifn.std() > 0:
        rho, p = stats.spearmanr(g.ifn, g.infl); pr, pp = stats.pearsonr(g.ifn, g.infl)
        print(f"\ninflation vs type-I IFN-Δ across {len(g)} donor folds: Spearman rho={rho:.3f} p={p:.3f}; Pearson r={pr:.3f} p={pp:.3f}")
    print(f"\nALL leak_free: {bool(df.leak_free.all())}")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
