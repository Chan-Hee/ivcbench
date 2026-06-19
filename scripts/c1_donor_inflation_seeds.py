#!/usr/bin/env python
"""Seed-aware bootstrap of the Kang donor random-split inflation (reviewer reqs 5, 10).

The published finding (+0.0143 mean inflation, random - LODO, Wilcoxon p=6.4e-4, n=48 baseline×donor)
rests on a SINGLE random partition (default_rng(0)). Here we re-draw the matched random-cell folds
across many seeds, re-running the (CPU, deterministic-given-split) simple baselines through the same
run_job path, to show the +0.014 inflation is a property of random-vs-LODO splitting, not of one lucky
partition. The LODO folds are fixed (donors are fixed); only the random folds vary by seed.

We also stratify the per-fold inflation by held lineage and by the held set's type-I IFN AUCell-Delta
(does the laundered signal concentrate in high-IFN monocytes vs T cells? req 10).

CPU only. Reuses the exact random-fold construction in clusters/spec.py (matched treated-count per
donor), parameterized by seed.
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
from ivcbench.runner.run import run_job  # noqa: E402

N_SEEDS = 20
IFN_GENES = ["ISG15", "IFI6", "MX1", "MX2", "OAS1", "OAS2", "IFIT1", "IFIT3", "ISG20",
             "STAT1", "IRF7", "IFI44", "IFI44L", "RSAD2", "USP18"]


def assign_rand_fold(cs, donors, seed):
    """Replicate clusters/spec.py random-fold construction for a given seed."""
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


def main():
    cs = kang_mod.load()
    is_ctrl = cs.obs["is_control"].astype(bool)
    donors = sorted(cs.obs["donor_id"].astype(str).unique())
    donors = [d for d in donors
              if int(((cs.obs["donor_id"].astype(str) == d) & ~is_ctrl).sum()) >= 50]
    print("donors:", donors, flush=True)

    # per-donor dominant lineage + held-set type-I IFN AUCell-Delta (for stratification, req 10)
    ct = cs.obs["cell_type_coarse"].astype(str)
    donor_lineage = {}
    donor_ifn = {}
    ifn_idx = np.asarray(cs.gene_index(IFN_GENES), dtype=int)
    X = cs.X

    def mean_ifn(rowmask):
        sub = X[rowmask][:, ifn_idx]
        return float(np.asarray(sub.mean())) if len(ifn_idx) else np.nan

    ctrl_mean_ifn = mean_ifn(is_ctrl.to_numpy())
    for d in donors:
        m = (cs.obs["donor_id"].astype(str) == d).to_numpy()
        treat_mask = m & ~is_ctrl.to_numpy()
        donor_lineage[d] = ct[treat_mask].value_counts().idxmax()
        donor_ifn[d] = mean_ifn(treat_mask) - ctrl_mean_ifn if len(ifn_idx) else np.nan

    # LODO baseline rows are FIXED (compute once per baseline×donor)
    lodo_vals = {}  # (baseline, donor) -> pearson_delta
    for d in donors:
        sp = c1.donor_lodo(d)
        for B in SIMPLE_BASELINES:
            r = run_job(cs, sp, B(), seed=0)
            if r.get("ran"):
                lodo_vals[(B().name, d)] = float(r["pearson_delta"])

    # random folds across seeds
    rows = []
    for seed in range(N_SEEDS):
        cs.obs["_rand_fold"] = assign_rand_fold(cs, donors, seed)
        for d in donors:
            sp = c1.random_cell_split(f"f{d}")
            for B in SIMPLE_BASELINES:
                r = run_job(cs, sp, B(), seed=0)
                if not r.get("ran"):
                    continue
                bn = B().name
                lo = lodo_vals.get((bn, d))
                if lo is None:
                    continue
                rows.append(dict(seed=seed, donor=d, baseline=bn,
                                 lineage=donor_lineage[d], ifn_delta=donor_ifn[d],
                                 rand=float(r["pearson_delta"]), lodo=lo,
                                 infl=float(r["pearson_delta"]) - lo))
        print(f"seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/C1/donor_inflation_seeds.csv", index=False)

    # --- summary ---
    print("\n=== Seed-aware inflation (random - LODO), simple baselines, "
          f"{N_SEEDS} seeds x {len(donors)} donors ===", flush=True)
    per_seed = df.groupby("seed").infl.mean()
    print(f"per-seed mean inflation: mean={per_seed.mean():.4f}, sd={per_seed.std(ddof=1):.4f}, "
          f"min={per_seed.min():.4f}, max={per_seed.max():.4f}", flush=True)
    print(f"fraction of seeds with positive mean inflation: {(per_seed > 0).mean():.3f}", flush=True)
    # bootstrap CI over seeds on the grand mean
    rng = np.random.default_rng(0)
    boot = [per_seed.sample(len(per_seed), replace=True, random_state=int(s)).mean()
            for s in rng.integers(0, 1e6, 10000)]
    print(f"bootstrap-over-seeds grand-mean 95% CI "
          f"[{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]", flush=True)
    # sign test on per-seed means
    k = int((per_seed > 0).sum())
    bt = stats.binomtest(k, len(per_seed), 0.5, alternative="greater")
    print(f"sign test on per-seed means: {k}/{len(per_seed)} positive, p={bt.pvalue:.2e}", flush=True)
    # per baseline
    print("\nper-baseline mean inflation (across all seeds×donors):", flush=True)
    print(df.groupby("baseline").infl.agg(["mean", "std", "count"]).round(4).to_string(), flush=True)
    # stratify by lineage (req 10)
    print("\ninflation by held-donor dominant lineage (excl ctrl-pred):", flush=True)
    sub = df[df.baseline != "ctrl-pred"]
    print(sub.groupby("lineage").agg(infl_mean=("infl", "mean"),
                                     ifn_delta=("ifn_delta", "mean"),
                                     n=("infl", "size")).round(4).to_string(), flush=True)
    # correlation inflation vs IFN-delta across donor folds (req 10)
    g = sub.groupby("donor").agg(infl=("infl", "mean"), ifn=("ifn_delta", "mean"))
    if g.ifn.notna().all() and g.ifn.std() > 0:
        rho, p = stats.spearmanr(g.ifn, g.infl)
        pr, pp = stats.pearsonr(g.ifn, g.infl)
        print(f"\ninflation vs type-I IFN-Delta across {len(g)} donor folds: "
              f"Spearman rho={rho:.3f} p={p:.3f}; Pearson r={pr:.3f} p={pp:.3f}", flush=True)
    print("\nwrote results/C1/donor_inflation_seeds.csv", flush=True)


if __name__ == "__main__":
    main()
