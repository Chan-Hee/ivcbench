#!/usr/bin/env python
"""C5 Tanimoto stratification precompute (for OnePager Figure 7 panels a–b).

Re-runs the C5 roster on the unseen-compound holdout split ONLY, and records, per (baseline, held
compound): the per-compound Pearson-Δ (mean over that compound's cell-type strata) and the Tanimoto
distance to the nearest TRAINING compound (1 − max Morgan-fingerprint Tanimoto similarity). The figure
then shows error-vs-Tanimoto (panel a) and the per-baseline Tanimoto robustness slope/R² (panel b).
Tanimoto is post-hoc (never a model input). Output: results/C5/tanimoto_percompound.csv.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ivcbench.clusters import c5
from ivcbench.clusters.spec import REGISTRY, _c5_held_compounds
from ivcbench.data.loaders.op3 import load
from ivcbench.metrics.response import pearson_delta
from ivcbench.runner.gating import Action, decide
from ivcbench.splits.audit import audit_split
from ivcbench.splits.builder import build_split


def _tanimoto(a, b):
    inter = float((a * b).sum())
    uni = float(a.sum() + b.sum() - inter)
    return inter / uni if uni > 0 else 0.0


def main():
    cs = load()
    spec = REGISTRY["C5"]
    held = _c5_held_compounds(cs)
    sp = c5.global_compound_holdout(held)
    split = build_split(cs, sp)
    audit_split(cs, split)                               # hard leak gate
    test_X = cs.X[split.test_idx]

    fp = {k: np.asarray(v, dtype=np.float32) for k, v in cs.side_info["fingerprint"].items()}
    held_fp = [h for h in held if h in fp]
    train_cpds = [c for c in cs.uns.get("compounds", []) if c in fp and c not in set(held)]
    tdist = {h: 1.0 - max(_tanimoto(fp[h], fp[t]) for t in train_cpds) for h in held_fp}
    print(f"[c5_tanimoto] held={len(held_fp)} train={len(train_cpds)} "
          f"Tanimoto-dist range {min(tdist.values()):.3f}-{max(tdist.values()):.3f}", flush=True)

    gpus = ["0", "1"]
    rows = []
    for i, B in enumerate(spec.baselines):
        name = getattr(B, "name", B.__name__)
        act = decide(name, sp.registry_task, True)
        if act is Action.SKIP:
            continue
        ad = B()
        if getattr(ad, "gpu", False):
            ad.cuda_device = gpus[i % len(gpus)]
        try:
            ad.fit(cs, split, side_info=cs.side_info)
            pred = ad.predict(cs, split, side_info=cs.side_info)
        except Exception as e:                           # non-fatal, like the sweep
            print(f"[c5_tanimoto] {name} FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        resp = pearson_delta(pred.pred_cells, test_X, pred.control_mean, split.test_strata, None)
        bycpd = defaultdict(list)
        for k, v in resp["per_stratum"].items():
            cpd = k.split("|")[0].split("=", 1)[1]       # "perturbation=<cpd>|cell_type_coarse=<lin>"
            bycpd[cpd].append(v)
        for cpd, vs in bycpd.items():
            if cpd in tdist:
                rows.append(dict(baseline=name, compound=cpd,
                                 pearson_delta=float(np.mean(vs)), tanimoto_dist=float(tdist[cpd]),
                                 action=act.value, headline_eligible=(act is Action.RUN_HEADLINE)))
        print(f"[c5_tanimoto] {name}: {len(bycpd)} compounds", flush=True)

    out = Path("results/C5/tanimoto_percompound.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[c5_tanimoto] wrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
