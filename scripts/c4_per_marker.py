#!/usr/bin/env python
"""C4 per-surface-marker recovery (reviewer request 10/18): the 20-protein CITE Pearson-Δ aggregate is
decomposed into named immune markers (CD274/PD-L1, HLA-A, HLA-E, CD279/PD-1, ...). For the unseen-KO
split we score, per marker, how well the (best simple = cell-mean) training-shift prediction recovers
the across-KO Δ pattern. Method matches the framework Pearson-Δ: per held KO stratum, Δ = treated_mean
- control_mean; for each marker we correlate predicted-Δ vs observed-Δ across the held-KO strata
(so a per-marker Pearson across the held knockouts). Leak-safe: cell-mean shift learned on train only.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "src")
from ivcbench.data.loaders.frangieh import load
from ivcbench.clusters import c4
from ivcbench.splits.builder import build_split

# pretty names for the CITE markers
MARKER_ALIAS = {
    "CD274": "PD-L1 (CD274)", "CD279": "PD-1 (CD279)", "HLA_A": "HLA-A (class I)",
    "HLA_E": "HLA-E (class I)", "CD119": "IFNGR1 (CD119)", "CD58": "CD58 (LFA-3)",
    "CD47": "CD47", "CD59": "CD59", "CD44": "CD44", "CD29": "CD29 (ITGB1)",
}

cs = load(modality="protein")
markers = list(cs.var_names)
print("CITE markers:", markers)
g = cs.uns["genes_perturbed"]

for frac, lbl in [(0.25, "25"), (0.50, "50")]:
    held = c4.held_ko_fraction(g, frac, seed=0)
    spec = c4.modality_lo_ko(held, lbl)
    sp = build_split(cs, spec)
    tr = sp.train_idx
    is_ctrl_tr = cs.obs.iloc[tr]["is_control"].to_numpy()
    # best-simple = cell-mean: predict every held KO with the training treated mean
    treated_mean = cs.X[tr[~is_ctrl_tr]].mean(0)
    ctrl_mean = cs.X[sp.inference_input_idx].mean(0) if len(sp.inference_input_idx) else cs.X[tr[is_ctrl_tr]].mean(0)
    pred_delta = treated_mean - ctrl_mean          # one global predicted Δ (cell-mean shift)

    # per held-KO observed Δ (test strata = held KO genes)
    strata = sp.test_strata
    test_X = cs.X[sp.test_idx]
    uniq = np.unique(strata)
    obs_delta = np.vstack([test_X[strata == s].mean(0) - ctrl_mean for s in uniq])  # (n_held_KO, n_marker)
    pred_delta_mat = np.tile(pred_delta, (len(uniq), 1))                            # cell-mean is constant

    # Per-marker recovery: cell-mean predicts ONE global Δ for every held KO. We report, per marker:
    #   obsΔ_mean = mean across held KOs of (treated - control)  [is the marker modulated by KO?]
    #   predΔ     = the cell-mean predicted Δ for that marker     [constant across KOs]
    #   recovery  = 1 - |obsΔ_mean - predΔ| / (|obsΔ_mean| + eps)  bounded recovery of the mean shift
    #   r_acrossKO= Pearson of obs Δ vs the global predicted-direction proxy is degenerate (pred const),
    #               so we instead report whether obs Δ direction matches the predicted sign (sign-match
    #               frac across held KOs) — an honest per-marker readout given a constant predictor.
    print(f"\n=== held {lbl}% ({len(uniq)} KO strata) — per-marker recovery (cell-mean shift) ===")
    print(f"{'marker':18s} {'predΔ':>8s} {'obsΔ_mean':>10s} {'obsΔ_sd':>9s} {'|err|':>7s} {'signmatch':>9s}")
    rows = []
    eps = 1e-9
    for j, mk in enumerate(markers):
        o = obs_delta[:, j]
        pj = float(pred_delta[j])
        err = abs(o.mean() - pj)
        signmatch = float(np.mean(np.sign(o) == np.sign(pj))) if pj != 0 else float("nan")
        rows.append((mk, pj, o.mean(), o.std(), err, signmatch))
    for mk, pj, om, osd, err, sm in sorted(rows, key=lambda x: -abs(x[3])):
        alias = MARKER_ALIAS.get(mk, mk)
        print(f"{alias:18s} {pj:8.3f} {om:10.3f} {osd:9.3f} {err:7.3f} {sm:9.2f}")
