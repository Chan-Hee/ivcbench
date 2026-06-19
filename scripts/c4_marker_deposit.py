#!/usr/bin/env python
"""C4 per-CITE-marker recovery DEPOSIT (reviewer reqs 10/18): serialize the per-surface-marker recovery
breakdown (already computed off-artifact by scripts/c4_per_marker.py) into a first-class deposited table
results/C4/cite_marker_recovery.csv, so the surface-marker immune claims (PD-1/CD279, HLA-E/HLA-A,
PD-L1/CD274) trace to a deposited artifact alongside results/C4/results_raw.csv. Per held-KO Δ recovery
by the best-simple (cell-mean) training shift, both holdout fractions. Leak-safe: shift learned on
train only. Columns: marker, alias, held_frac, n_held_KO, predDelta, obsDelta_mean, obsDelta_sd,
abs_err, sign_match.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from ivcbench.data.loaders.frangieh import load
from ivcbench.clusters import c4
from ivcbench.splits.builder import build_split

MARKER_ALIAS = {
    "CD274": "PD-L1 (CD274)", "CD279": "PD-1 (CD279)", "HLA_A": "HLA-A (class I)",
    "HLA_E": "HLA-E (class I)", "CD119": "IFNGR1 (CD119)", "CD58": "CD58 (LFA-3)",
    "CD47": "CD47", "CD59": "CD59", "CD44": "CD44", "CD29": "CD29 (ITGB1)",
}

cs = load(modality="protein")
markers = list(cs.var_names)
g = cs.uns["genes_perturbed"]

rows = []
for frac, lbl in [(0.25, "25"), (0.50, "50")]:
    held = c4.held_ko_fraction(g, frac, seed=0)
    spec = c4.modality_lo_ko(held, lbl)
    sp = build_split(cs, spec)
    tr = sp.train_idx
    is_ctrl_tr = cs.obs.iloc[tr]["is_control"].to_numpy()
    treated_mean = cs.X[tr[~is_ctrl_tr]].mean(0)
    ctrl_mean = (cs.X[sp.inference_input_idx].mean(0) if len(sp.inference_input_idx)
                 else cs.X[tr[is_ctrl_tr]].mean(0))
    pred_delta = treated_mean - ctrl_mean

    strata = sp.test_strata
    test_X = cs.X[sp.test_idx]
    uniq = np.unique(strata)
    obs_delta = np.vstack([test_X[strata == s].mean(0) - ctrl_mean for s in uniq])  # (n_KO, n_marker)

    for j, mk in enumerate(markers):
        o = obs_delta[:, j]
        pj = float(pred_delta[j])
        signmatch = float(np.mean(np.sign(o) == np.sign(pj))) if pj != 0 else np.nan
        rows.append({
            "marker": mk,
            "alias": MARKER_ALIAS.get(mk, mk),
            "held_frac_pct": int(lbl),
            "n_held_KO": int(len(uniq)),
            "predDelta": pj,
            "obsDelta_mean": float(o.mean()),
            "obsDelta_sd": float(o.std()),
            "abs_err": float(abs(o.mean() - pj)),
            "sign_match_frac": signmatch,
        })

df = pd.DataFrame(rows)
out = "results/C4/cite_marker_recovery.csv"
df.to_csv(out, index=False)
print(f"WROTE {out} ({len(df)} rows = {len(markers)} markers x 2 holdout fractions)")
# echo the named immune checkpoint / HLA markers for the log
key = df[df["marker"].isin(["CD279", "CD274", "HLA_E", "HLA_A"])]
print(key.to_string(index=False))
