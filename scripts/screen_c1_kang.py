#!/usr/bin/env python
"""SCREEN on the T1 cell-context split (Kang leave-one-lineage-out)."""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from ivcbench.clusters.c1 import coarse_loct
from ivcbench.data.loaders import kang as kangmod
from ivcbench.baselines.heavy import ScreenC1
from ivcbench.runner.run import run_job

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default=None); ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
cs = kangmod.load()
lins = sorted(set(cs.obs["cell_type_coarse"]))
if a.limit: lins = lins[:a.limit]
print(f"[SCREEN] Kang {cs.X.shape[0]} cells x {cs.X.shape[1]} genes; lineages {lins}", flush=True)
rows = []
for lin in lins:
    t0 = time.time()
    ad = ScreenC1()
    if a.gpu is not None: ad.cuda_device = str(a.gpu)
    r = run_job(cs, coarse_loct(held_lineage=lin), ad, seed=0)
    rows.append({k: r.get(k) for k in ["baseline","family","split","action","ran","leak_free",
                                       "n_train","n_test","pearson_delta","e_distance"]})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"  {lin:16s} {time.time()-t0:6.0f}s ran={r.get('ran')} leak={r.get('leak_free')} "
          f"pearson_delta={r.get('pearson_delta')}", flush=True)
print(f"[done] -> {a.out}", flush=True)
