#!/usr/bin/env python
"""Foundation models on the T2 donor split (Soskic leave-one-donor-out).

Reviewer 2 asked why the foundation models are reported only on T3, where the task has little
headroom, when their inputs are equally valid on the high-headroom tasks. This runs them on T2
through the same cell-context adapters already used for T1 and documented in Supplementary
Table S15: the stimulus is a single global condition flag and the held donor's identity is carried
by its own control cells, which are the inference input.

Identical split, floors, metric and seed policy as the deposited CellOT T2 run, so the new cells are
directly comparable to the census.

    <python> scripts/foundation_c2_donor.py --model scGPT --chunk 0 4 --out <csv>
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split
from ivcbench.metrics.response import pearson_delta
from ivcbench.baselines.simple import CellMean, DonorShift, LinearPCA
from ivcbench.baselines.heavy import ScGPTC1, ScFoundationC1
from ivcbench.eval.bundle import dump_bundle
from c2_soskic_donor import load_soskic_donor, lodo_spec, response_gene_idx

ADAPTERS = {"scGPT": ScGPTC1, "scFoundation": ScFoundationC1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--chunk", type=int, nargs=2, default=None, metavar=("I", "N"))
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cs = load_soskic_donor(args.cap)
    donors = sorted(cs.obs.donor_id.unique())
    if args.chunk:
        i, n = args.chunk
        donors = donors[i::n]
    print(
        f"[{args.model}] {cs.X.shape[0]} cells x {cs.X.shape[1]} genes;"
        f" {len(donors)} donors",
        flush=True,
    )

    rows = []
    out_path = Path(args.out)
    if out_path.exists():  # resume
        old = pd.read_csv(out_path)
        rows = old.to_dict("records")
        done = set(old["donor"].astype(str))
        donors = [d for d in donors if str(d) not in done]
        print(f"  resume: {len(done)} donors already scored", flush=True)

    for k, d in enumerate(donors):
        t0 = time.time()
        sp = build_split(cs, lodo_spec(d))
        assert audit_split(cs, sp)["leak_free"], f"LEAK {d}"
        test_X = cs.X[sp.test_idx]
        test_strata = sp.test_strata
        ctrl_idx = sp.inference_input_idx
        ctrl_mean = cs.X[ctrl_idx].mean(0)
        rg = response_gene_idx(cs, sp.train_idx)

        floors = {}
        for B in (CellMean, DonorShift, LinearPCA):
            b = B()
            b.fit(cs, sp, side_info=cs.side_info)
            floors[b.name] = float(
                pearson_delta(
                    b.predict(cs, sp, side_info=cs.side_info).pred_cells,
                    test_X,
                    ctrl_mean,
                    test_strata,
                    rg,
                )["macro"]
            )
        binding = max(floors["cell-mean"], floors["linear-PCA"])

        adapter = ADAPTERS[args.model]()
        if args.gpu is not None:
            adapter.cuda_device = str(args.gpu)
        adapter.fit(cs, sp, side_info=cs.side_info)
        pred = adapter.predict(cs, sp, side_info=cs.side_info)
        pe = float(
            pearson_delta(pred.pred_cells, test_X, ctrl_mean, test_strata, rg)["macro"]
        )

        dump_bundle(
            os.environ.get("IVCBENCH_PRED_DUMP"),
            cluster="C2",
            model=args.model,
            split=sp.spec.name,
            pred_cells=pred.pred_cells,
            test_cells=test_X,
            cell_strata=test_strata,
            control_mean=ctrl_mean,
            genes=cs.var_names,
            exclude_gene_idx=rg,
            n_pca=50,
        )

        rows.append(
            dict(
                donor=str(d),
                model=args.model,
                pearson_delta=round(pe, 4),
                cell_mean=round(floors["cell-mean"], 4),
                linear_PCA=round(floors["linear-PCA"], 4),
                donor_shift=round(floors["donor-shift"], 4),
                binding_floor=round(binding, 4),
                margin=round(pe - binding, 4),
                beats=bool(pe > binding),
                n_test=int(len(sp.test_idx)),
                n_ctrl=int(len(ctrl_idx)),
                leak_free=True,
            )
        )
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(
            f"  [{k+1}/{len(donors)} {d}] {time.time()-t0:.0f}s pearsonD={pe:.4f} "
            f"floor={binding:.4f} margin={pe-binding:+.4f}",
            flush=True,
        )
    print(f"[done] {len(rows)} donors -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
