#!/usr/bin/env python
"""Fold-local sensitivity for the T2 donor headline (Reviewer-facing robustness check).

The deposited census standardises the Soskic panel once over all 106 donors, so the held-out
donor's cells contribute to the per-gene mean and SD. That transform is unsupervised and is
applied identically to CellOT and to both universal-floor members, but a reviewer is entitled
to ask whether the donor result survives estimating it inside each fold.

Here the matrix is loaded without ADDITIONAL joint standardization. It is already
condition-group covariate-regressed, scaled and clipped in the source study; the
variable name `raw` below does not mean raw counts or unregressed expression.
For each fold, an additional per-gene mean/SD is fitted on training donors and
applied to both training and held cells. Caps, splits, seeds, budgets, floors and
metric match the primary run. This tests only the additional scaling choice, not
upstream feature selection, regression or clipping, and cannot establish a fully
inductive raw-cell evaluation.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ivcbench.data.schema import CellSet
from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split
from ivcbench.metrics.response import pearson_delta
from ivcbench.baselines.simple import CellMean, DonorShift, LinearPCA
import cellot_runner as R
from c2_soskic_donor import load_soskic_donor, lodo_spec, response_gene_idx


def fold_standardised(raw: CellSet, train_mask: np.ndarray) -> CellSet:
    """Additional train-fitted scaling of the already processed source matrix."""
    mu = raw.X[train_mask].mean(0, keepdims=True)
    sd = raw.X[train_mask].std(0, keepdims=True) + 1e-6
    return CellSet(
        X=((raw.X - mu) / sd).astype(np.float32),
        obs=raw.obs,
        var_names=raw.var_names,
        side_info=raw.side_info,
        uns=raw.uns,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, nargs=2, default=None, metavar=("I", "N"))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--ae-iters", type=int, default=12000)
    ap.add_argument("--cellot-iters", type=int, default=8000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = load_soskic_donor(args.cap, standardize="none")
    donors = sorted(raw.obs.donor_id.unique())
    if args.chunk:
        i, n = args.chunk
        donors = donors[i::n]
    print(
        f"[foldlocal] {raw.X.shape[0]} cells x {raw.X.shape[1]} genes;"
        f" {len(donors)} donors",
        flush=True,
    )

    don = raw.obs.donor_id.astype(str).to_numpy()
    rows = []
    for k, d in enumerate(donors):
        t0 = time.time()
        cs = fold_standardised(raw, don != str(d))  # standardise on the training donors alone, with the held donor removed
        sp = build_split(cs, lodo_spec(d))
        assert audit_split(cs, sp)["leak_free"], f"LEAK {d}"
        test_X = cs.X[sp.test_idx]
        test_strata = sp.test_strata
        ctrl_idx = sp.inference_input_idx
        ctrl_strata = np.array(
            [
                f"cell_type_coarse={cs.obs.iloc[i]['cell_type_coarse']}"
                for i in ctrl_idx
            ],
            dtype=object,
        )
        ctrl_mean = cs.X[ctrl_idx].mean(0)
        rg = response_gene_idx(cs, sp.train_idx)

        bp = {}
        for B in (CellMean, DonorShift, LinearPCA):
            b = B()
            b.fit(cs, sp, side_info=cs.side_info)
            bp[b.name] = float(
                pearson_delta(
                    b.predict(cs, sp, side_info=cs.side_info).pred_cells,
                    test_X,
                    ctrl_mean,
                    test_strata,
                    rg,
                )["macro"]
            )
        pes = []
        for seed in args.seeds:
            pred, _ = R.run_cellot_on_split(
                cs, sp, seed, args.ae_iters, args.cellot_iters
            )
            aligned = R.stratum_align(pred, ctrl_strata, test_strata)
            pes.append(
                float(
                    pearson_delta(aligned, test_X, ctrl_mean, test_strata, rg)["macro"]
                )
            )
        cellot = float(np.mean(pes))
        binding = max(
            bp["cell-mean"], bp["linear-PCA"]
        )  # the floor to clear is the stronger of the two simple references
        rows.append(
            dict(
                donor=str(d),
                cellot=round(cellot, 4),
                cell_mean=round(bp["cell-mean"], 4),
                linear_PCA=round(bp["linear-PCA"], 4),
                donor_shift=round(bp["donor-shift"], 4),
                binding_floor=round(binding, 4),
                margin=round(cellot - binding, 4),
                beats=bool(cellot > binding),
            )
        )
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(
            f"  [{k+1}/{len(donors)} {d}] {time.time()-t0:.0f}s cellot={cellot:.4f} "
            f"floor={binding:.4f} margin={cellot-binding:+.4f}",
            flush=True,
        )
    print(f"[done] {len(rows)} donors -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
