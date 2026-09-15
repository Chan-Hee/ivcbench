#!/usr/bin/env python
"""Flag any census cell whose predictions do not distinguish the conditions it was asked about.

THE QUESTION A REVIEWER ASKS. On a task with many named conditions, did the model actually respond
to each one, or is it emitting one answer under many labels? A score alone cannot tell you: a
compound-blind predictor still scores, and on this benchmark several did.

THE MEASURE. Per bundle, the median pairwise Pearson correlation, across genes, of the predicted
delta (prediction - control) between different strata. 1.000 means every condition received the
same direction. The observed data's own value is printed beside it as the reference: conditions are
genuinely correlated with each other (~0.3 on T5c), so the target is not 0 -- it is "not pinned at
1 when the data is not".

WHAT IS LEGITIMATELY 1.000. Floors and diagnostics that are pooled BY DEFINITION -- cell-mean,
donor-shift, linear-PCA, ctrl-pred, CINEMA-OT -- and any task whose stratification is not by
condition at all (T1/T2 hold a group, and the single stimulus is the same for every stratum). Those
are listed as expected, not flagged.

This is a provenance check, not a quality gate: a model that is genuinely condition-blind is a
result. What it catches is a WIRING defect that makes a model look condition-blind when the
published method is not -- which is what happened to scPRAM (140 of 141 compounds got the control
mean), CellOT (one pooled transport broadcast to all) and the foundation C1 adapters (all compounds
pooled into one exposure).

  python scripts/audit_condition_separation.py [--task T5c] [--strict]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

# pooled by construction: these are the reference floors and the explicitly pooled diagnostics
EXPECTED_POOLED = {"cell-mean", "donor-shift", "linear-PCA", "ctrl-pred", "CINEMA-OT"}
# tasks whose strata are groups, not conditions: one stimulus, so one direction is correct
GROUP_TASKS = ("C1_loct", "C2_lodo", "C2_soskic")
BLIND = 0.999


def median_pair_r(M: np.ndarray) -> float:
    A = M - M.mean(1, keepdims=True)
    n = np.linalg.norm(A, axis=1)
    keep = n > 1e-12
    A, n = A[keep], n[keep]
    if len(A) < 2:
        return float("nan")
    R = (A @ A.T) / np.outer(n, n)
    iu = np.triu_indices(len(A), 1)
    return float(np.median(R[iu]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="C5_loct", help="substring of the split to audit")
    ap.add_argument("--strict", action="store_true", help="exit 6 if an unexpected cell is blind")
    a = ap.parse_args()
    from assemble_cross_cluster import eligible_bundle

    per_model: dict[str, list[tuple[str, float, float]]] = {}
    for f in glob.glob(os.path.join(ROOT, "predictions", "**", "*.npz"), recursive=True):
        if a.match not in f or not eligible_bundle(f):
            continue
        z = np.load(f, allow_pickle=True)
        if "pred_means" not in z.files or z["pred_means"].shape[0] < 2:
            continue
        C = z["control_mean"].astype(np.float64)
        pr = median_pair_r(z["pred_means"].astype(np.float64) - C)
        ob = median_pair_r(z["obs_means"].astype(np.float64) - C)
        model = str(z["model"]) if "model" in z.files else "?"
        per_model.setdefault(model, []).append((os.path.basename(f), pr, ob))

    if not per_model:
        print(f"no eligible bundles matched {a.match!r}")
        return
    is_group_task = any(t in a.match for t in GROUP_TASKS)
    flagged = []
    print(f"condition separation for {a.match!r} — median pairwise Pearson of the predicted Δ")
    print(f"  (1.000 = one direction for every condition; the observed Δ value is the reference)\n")
    for model, rows in sorted(per_model.items(), key=lambda kv: -np.nanmean([r[1] for r in kv[1]])):
        vals = [r[1] for r in rows]
        obs = float(np.nanmean([r[2] for r in rows]))
        mean = float(np.nanmean(vals))
        if mean > BLIND and model not in EXPECTED_POOLED and not is_group_task:
            tag, why = "BLIND", "  <-- one direction for every condition; check the wiring"
            flagged.append(model)
        elif mean > BLIND:
            tag, why = "pooled", "  (pooled by construction)"
        else:
            tag, why = "ok", ""
        print(
            f"  [{tag:^6}] {model:<22} n={len(vals)} "
            f"pred {', '.join(f'{v:.3f}' for v in vals)} | obs {obs:.3f}{why}"
        )
    if flagged:
        print(f"\n{len(flagged)} model(s) flagged: {', '.join(flagged)}")
        if a.strict:
            sys.exit(6)
    else:
        print("\nno unexpected condition-blind cell")


if __name__ == "__main__":
    main()
