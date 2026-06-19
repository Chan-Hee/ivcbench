#!/usr/bin/env python
"""Complementary axis: the OBSERVED across-stratum program-shift signal (the learnable signal magnitude).

For each C5/C3 program we recompute, from the ACTUAL cells, the observed AUCell program-delta for every
perturbation stratum (vs the control mean), and report its dispersion across strata:
   obs_shift_sd  = std of obs program-delta across strata  (how much the program score actually MOVES
                   from perturbation to perturbation = the signal a recovery correlation can lock onto)
   obs_shift_mean= mean obs program-delta (overall program activation by the panel)
A recovery CORRELATION is only non-degenerate / learnable when obs_shift_sd is appreciable. This is the
honest mechanistic alternative to "low-rank-ness": recovery may track SIGNAL not DIMENSIONALITY.
Pure recompute via the same loaders + aucell as the deposited pipeline. CPU.
"""
from __future__ import annotations
import sys, importlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.clusters.c3 import C3_PROGRAMS
from ivcbench.clusters.c5 import C5_PROGRAMS
from ivcbench.metrics.program import aucell

OUT = ROOT / "results/_paper/program_shift_signal.csv"
C3_DATASETS = {
    "shifrut": ("ivcbench.data.loaders.shifrut", "load", {}),
    "schmidt": ("ivcbench.data.loaders.schmidt", "load", {}),
    "mccutcheon_CRISPRi": ("ivcbench.data.loaders.mccutcheon", "load", {"modality": "CRISPRi"}),
    "mccutcheon_CRISPRa": ("ivcbench.data.loaders.mccutcheon", "load", {"modality": "CRISPRa"}),
    "chen": ("ivcbench.data.loaders.chen", "load", {}),
}


def _load(mod, fn, kw):
    return getattr(importlib.import_module(mod), fn)(**kw)


def obs_shift(cs, genes):
    gi = cs.gene_index(genes)
    if len(gi) < 3:
        return np.nan, np.nan, 0
    ctrl_m = cs.obs["is_control"].astype(bool).to_numpy()
    if ctrl_m.sum() < 10:
        return np.nan, np.nan, 0
    ctrl_mean = float(aucell(cs.X[ctrl_m], gi).mean())
    pert = cs.obs["perturbation"].astype(str).to_numpy()
    strata = pert[~ctrl_m]
    Xp = cs.X[~ctrl_m]
    deltas = []
    for s in np.unique(strata):
        m = strata == s
        if m.sum() < 5:
            continue
        deltas.append(aucell(Xp[m], gi).mean() - ctrl_mean)
    deltas = np.array(deltas)
    if len(deltas) < 2:
        return np.nan, np.nan, len(deltas)
    return float(deltas.std()), float(deltas.mean()), len(deltas)


def main():
    rows = []
    # C5
    cs = _load("ivcbench.data.loaders.op3", "load", {})
    for name, genes in C5_PROGRAMS.items():
        sd, mu, ns = obs_shift(cs, genes)
        rows.append(dict(cluster="C5", program=name, obs_shift_sd=sd, obs_shift_mean=mu, n_strata=ns))
        print(f"C5 {name:20s} shift_sd={sd:.4f} shift_mean={mu:+.4f} n_strata={ns}", flush=True)
    del cs
    # C3 averaged over datasets
    acc = {p: {"sd": [], "mu": [], "w": []} for p in C3_PROGRAMS}
    for ds, (mod, fn, kw) in C3_DATASETS.items():
        cs = _load(mod, fn, kw)
        for name, genes in C3_PROGRAMS.items():
            sd, mu, ns = obs_shift(cs, genes)
            acc[name]["sd"].append(sd); acc[name]["mu"].append(mu); acc[name]["w"].append(ns)
            print(f"C3/{ds} {name:18s} shift_sd={sd if sd==sd else float('nan'):.4f} n_strata={ns}", flush=True)
        del cs
    for name in C3_PROGRAMS:
        sd = np.array(acc[name]["sd"]); mu = np.array(acc[name]["mu"]); w = np.array(acc[name]["w"], float)
        mk = np.isfinite(sd)
        sd_m = float(np.average(sd[mk], weights=w[mk])) if mk.any() else np.nan
        mu_m = float(np.average(mu[mk], weights=w[mk])) if mk.any() else np.nan
        rows.append(dict(cluster="C3", program=name, obs_shift_sd=sd_m, obs_shift_mean=mu_m,
                         n_strata=int(np.nansum(w))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print("\nWROTE", OUT)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
