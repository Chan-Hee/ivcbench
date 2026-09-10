#!/usr/bin/env python
"""Foundation models on the T4 unseen-KO modality split (Frangieh RNA).

Completes the foundation-model row of the coverage matrix. Unlike the T1/T2/T5 drivers, this task
is the models' OWN published interface -- a held-out gene knockout predicted from a perturbation
token -- so ScGPT/ScFoundation run natively here, exactly as they do on T3; no author-written task
interface is involved. The split, floors, metric and seed policy are the framework's own (run_job),
so the new cells are directly comparable to the deposited C4 census rows.

    <python> scripts/foundation_c4_modality.py --model scGPT --gpu 0 --out <csv>
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ivcbench.data.loaders.frangieh import load
from ivcbench.clusters import c4
from ivcbench.baselines.heavy import ScGPT, ScFoundation
from ivcbench.runner.run import run_job

ADAPTERS = {"scGPT": ScGPT, "scFoundation": ScFoundation}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cs = load(modality="rna")  # census deposits only the RNA modality bundles
    g = cs.uns["genes_perturbed"]
    print(
        f"[{args.model}] Frangieh RNA {cs.X.shape[0]} cells x {cs.X.shape[1]} genes; "
        f"{len(g)} perturbed genes",
        flush=True,
    )

    rows = []
    for frac, lbl in [(0.25, "25"), (0.50, "50")]:
        t0 = time.time()
        held = c4.held_ko_fraction(g, frac, seed=0)
        spec = c4.modality_lo_ko(held, lbl)
        ad = ADAPTERS[args.model]()
        if args.gpu is not None:
            ad.cuda_device = str(args.gpu)
        r = run_job(cs, spec, ad, seed=0, exclude_genes=list(spec.held_values))
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
            f"  frac{lbl:3s} {time.time()-t0:6.0f}s ran={r.get('ran')} "
            f"leak_free={r.get('leak_free')} pearson_delta={r.get('pearson_delta')}",
            flush=True,
        )
    print(f"[done] {len(rows)} folds -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
