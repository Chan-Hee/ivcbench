#!/usr/bin/env python
"""C5 program-vs-bulk dissociation — compound-label-shuffle permutation null (reviewer req 12).

The C5 LOCT cell-context finding is that conditioned models recover the type-I IFN program broadly
(observed AUCell-Δ recovery ~0.75-0.79) even where bulk magnitude is lost. A reviewer can ask whether
that ~0.77 IFN recovery is real conditioning signal or an AUCell artifact. We test it with a label-
shuffle null that MIRRORS the C3 degenerate-zero null: the IFN recovery is corr(pred_Δ, obs_Δ) across
the held lineage's compound strata; under the null the model carries NO compound-specific information,
so its predicted per-compound IFN-Δ vector is exchangeable across compounds. We therefore permute the
predicted per-compound IFN-Δ relative to the observed per-compound IFN-Δ WITHIN each held lineage
(= shuffle which compound's prediction is matched to which compound's truth) and recompute the
correlation, 2000 permutations per lineage. If the observed recovery exceeds the shuffled-null band,
the IFN recovery is compound-resolved signal, not a marginal AUCell effect.

Model: FP-ridge (the only conditioned model that is fully CPU + deterministic, and the one whose IFN
recovery 0.770 also converts to bulk magnitude). Leak-safe via the framework build_split + audit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivcbench.data.loaders import op3  # noqa: E402
from ivcbench.clusters import c5  # noqa: E402
from ivcbench.clusters.c5 import C5_PROGRAMS  # noqa: E402
from ivcbench.baselines.chemistry import FPRidge  # noqa: E402
from ivcbench.splits.builder import build_split  # noqa: E402
from ivcbench.splits.audit import audit_split  # noqa: E402
from ivcbench.metrics.program import aucell  # noqa: E402

N_PERM = 2000
RNG = np.random.default_rng(0)


def per_compound_ifn_delta(pred_cells, test_cells, control_cells, gs_idx, strata):
    """Return (obs_d, pred_d) vectors over compound strata for the type-I IFN gene set."""
    ctrl = float(aucell(control_cells, gs_idx).mean())
    uniq = np.unique(strata)
    obs_d, pred_d = [], []
    for s in uniq:
        m = strata == s
        obs_d.append(aucell(test_cells[m], gs_idx).mean() - ctrl)
        pred_d.append(aucell(pred_cells[m], gs_idx).mean() - ctrl)
    return np.array(obs_d), np.array(pred_d), uniq


def main():
    cs = op3.load()
    gs_idx = cs.gene_index(C5_PROGRAMS["type_I_IFN"])
    # held-lineage label as it appears in cs.obs['cell_type_coarse']; "T cells" has a space (the
    # split DISPLAY name C5_loct_T_cells underscores it). Map display->held for the CSV row.
    lineages = [("B", "B"), ("Mono", "Mono"), ("NK", "NK"), ("T_cells", "T cells")]
    rows = []
    for disp, lin in lineages:
        spec = c5.cross_celltype_loct(held_lineage=lin)
        split = build_split(cs, spec)
        audit = audit_split(cs, split)
        assert audit["leak_free"], f"LEAK on {lin}"
        adapter = FPRidge()
        adapter.fit(cs, split, side_info=cs.side_info)
        pred = adapter.predict(cs, split, side_info=cs.side_info)
        test_X = cs.X[split.test_idx]
        ctrl_cells = cs.X[split.inference_input_idx] if len(split.inference_input_idx) else test_X
        obs_d, pred_d, uniq = per_compound_ifn_delta(pred.pred_cells, test_X, ctrl_cells,
                                                     gs_idx, split.test_strata)
        # observed recovery (same formula as aucell_delta_corr)
        if obs_d.std() < 1e-12 or pred_d.std() < 1e-12:
            obs_corr = 0.0
        else:
            obs_corr = float(np.corrcoef(pred_d, obs_d)[0, 1])
        # permutation null: shuffle which compound's prediction matches which compound's truth
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            perm = RNG.permutation(len(pred_d))
            pd_perm = pred_d[perm]
            if pd_perm.std() < 1e-12:
                null[b] = 0.0
            else:
                null[b] = np.corrcoef(pd_perm, obs_d)[0, 1]
        # one-sided p: P(null >= observed)
        p_perm = float((1 + np.sum(null >= obs_corr)) / (1 + N_PERM))
        null_mean = float(null.mean())
        null_hi95 = float(np.quantile(null, 0.95))
        null_lo95 = float(np.quantile(null, 0.05))
        z = (obs_corr - null_mean) / (null.std() + 1e-12)
        rows.append({
            "lineage": disp, "n_compound_strata": int(len(uniq)),
            "obs_IFN_recovery": round(obs_corr, 4),
            "null_mean": round(null_mean, 4),
            "null_5pct": round(null_lo95, 4), "null_95pct": round(null_hi95, 4),
            "null_z": round(z, 2), "perm_p_onesided": round(p_perm, 5), "n_perm": N_PERM,
        })
        print(f"{disp:8s} n_cpd={len(uniq):3d} obs={obs_corr:.4f} "
              f"null_mean={null_mean:+.4f} null95={null_hi95:.4f} z={z:.2f} p={p_perm:.4f}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "results/C5/ifn_shuffle_null.csv"
    df.to_csv(out, index=False)
    print(f"\nWROTE {out}")
    # combined Fisher p across lineages (independent permutation tests)
    from scipy.stats import combine_pvalues
    stat, fisher_p = combine_pvalues(df["perm_p_onesided"].clip(lower=1e-6).to_numpy(), method="fisher")
    print(f"mean obs IFN recovery = {df['obs_IFN_recovery'].mean():.4f}; "
          f"mean null = {df['null_mean'].mean():+.4f}; "
          f"Fisher combined one-sided p = {fisher_p:.2e} (chi2={stat:.2f})")


if __name__ == "__main__":
    main()
