#!/usr/bin/env python
"""Foundation models on the T5 cell-context split (OP3 leave-one-cell-type-out).

Companion to scripts/foundation_c2_donor.py, answering the same reviewer point on the second
high-headroom task the foundation models were input-valid for but not reported on. The split,
floors, metric and seed policy are the framework's own (run_job), so the new cells are directly
comparable to the deposited C5 LOCT census rows.

    <python> scripts/foundation_c5_loct.py --model scGPT --gpu 0 --out <csv>
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ivcbench.data.loaders import op3 as op3mod
from ivcbench.clusters.c5 import cross_celltype_loct, C5_PROGRAMS
from ivcbench.baselines.heavy import ScGPTC1, ScFoundationC1
from ivcbench.runner.run import run_job

ADAPTERS = {"scGPT": ScGPTC1, "scFoundation": ScFoundationC1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cs = op3mod.load()  # the same coarse 4-class loader the census uses
    lineages = sorted(set(cs.obs["cell_type_coarse"]))
    print(
        f"[{args.model}] OP3 {cs.X.shape[0]} cells x {cs.X.shape[1]} genes; lineages"
        f" {lineages}",
        flush=True,
    )

    rows = []
    for lin in lineages:
        t0 = time.time()
        spec = cross_celltype_loct(held_lineage=lin)
        ad = ADAPTERS[args.model]()
        if args.gpu is not None:
            ad.cuda_device = str(args.gpu)
        r = run_job(cs, spec, ad, seed=0, immune_programs=dict(C5_PROGRAMS))
        rows.append(
            {
                k: r.get(k)
                for k in [
                    "baseline",
                    "family",
                    "split",
                    "action",
                    "ran",
                    "leak_free",
                    "n_train",
                    "n_test",
                    "n_test_strata",
                    "pearson_delta",
                    "pearson_delta_lo",
                    "pearson_delta_hi",
                    "e_distance",
                ]
            }
        )
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(
            f"  {lin:10s} {time.time()-t0:6.0f}s"
            f" ran={r.get('ran')} leak_free={r.get('leak_free')} "
            f"pearson_delta={r.get('pearson_delta')}",
            flush=True,
        )
    print(f"[done] {len(rows)} lineages -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
