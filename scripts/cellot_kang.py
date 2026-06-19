#!/usr/bin/env python
"""EXPERIMENT A1: CellOT on Kang IFN-beta PBMC leave-one-lineage-out (LOLO).

Uses the EXACT repo split: src/ivcbench/clusters/c1.py:coarse_loct + dispatch _c1_splits (lineages with
>=50 treated cells). For each held lineage CellOT is trained (scgen AE + f/g ICNN) on all NON-held cells
and predicts the held lineage's IFN-beta response from its OWN control cells. Metrics = repo
pearson_delta / e_distance / type-I-IFN AUCell-delta. Primary baseline = matched simple context baseline (cell-mean / donor-shift; not a universal-floor member).
Strata = donor_id (matches coarse_loct strata_cols).
"""
from __future__ import annotations
import sys, time, argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ivcbench.data.loaders import kang as kang_mod
from ivcbench.clusters import c1
from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split
from ivcbench.metrics.response import pearson_delta
from ivcbench.metrics.distribution import e_distance
from ivcbench.metrics.program import aucell
from ivcbench.baselines.simple import CellMean, DonorShift, CtrlPred, LinearPCA

import cellot_runner as R


def lineages_for(cs):
    ct = cs.obs["cell_type_coarse"].astype(str)
    is_ctrl = cs.obs["is_control"].astype(bool)
    treated = ct[~is_ctrl]
    return [l for l in sorted(ct.unique()) if int((treated == l).sum()) >= 50]


def baseline_preds(cs, sp):
    """All 4 simple baselines' per-test predictions + control mean (for the primary + best-of-four)."""
    out = {}
    for B in (CtrlPred, CellMean, DonorShift, LinearPCA):
        b = B(); b.fit(cs, sp, side_info=cs.side_info); p = b.predict(cs, sp, side_info=cs.side_info)
        out[b.name] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineages", nargs="*", default=None, help="subset of lineages (default all)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--ae-iters", type=int, default=12000)
    ap.add_argument("--cellot-iters", type=int, default=8000)
    ap.add_argument("--out", default=str(ROOT / "outputs/additional_models/cellot_kang_raw.csv"))
    ap.add_argument("--timing-out", default=str(ROOT / "outputs/additional_models/cellot_kang_timing.json"))
    args = ap.parse_args()

    print(f"[device] {R.DEVICE}", flush=True)
    cs = kang_mod.load()
    gs_ifn = np.asarray(cs.gene_index(IFN := IFN_LIST), dtype=int)
    lins = args.lineages or lineages_for(cs)
    print(f"[kang] {cs.n_cells} cells x {cs.n_genes} genes; lineages={lins}", flush=True)

    rows, timing = [], []
    for lin in lins:
        sp = build_split(cs, c1.coarse_loct(lin))
        audit = audit_split(cs, sp)
        assert audit["leak_free"], f"LEAK {lin}"
        test_X = cs.X[sp.test_idx]
        test_strata = sp.test_strata
        ctrl_idx = sp.inference_input_idx
        ctrl_X = cs.X[ctrl_idx]
        ctrl_strata = np.array([f"donor_id={cs.obs.iloc[i]['donor_id']}" for i in ctrl_idx], dtype=object)
        ctrl_mean = ctrl_X.mean(0)
        ed_basis = cs.X[sp.train_idx]
        if len(ed_basis) > 5000:
            ed_basis = ed_basis[np.random.default_rng(0).choice(len(ed_basis), 5000, replace=False)]
        ctrl_auc = float(aucell(ctrl_X, gs_ifn).mean())

        # ---- baselines (computed once; deterministic) ----
        bpreds = baseline_preds(cs, sp)
        def b_pearson(name):
            p = bpreds[name]
            return float(pearson_delta(p.pred_cells, test_X, p.control_mean, test_strata)["macro"])
        def b_edist(name):
            p = bpreds[name]
            return float(e_distance(p.pred_cells, test_X, test_strata, fit_on=ed_basis)["macro"])
        def b_aucell(name):
            p = bpreds[name]
            # per-donor-stratum predicted IFN delta vs observed -> magnitude error proxy: use mean delta
            return float(aucell(p.pred_cells, gs_ifn).mean()) - ctrl_auc
        prim_pearson_name = max(("cell-mean", "donor-shift"), key=b_pearson)
        prim_edist_name = min(("cell-mean", "donor-shift"), key=b_edist)  # lower is better
        bof_pearson = max(("ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"), key=b_pearson)

        obs_ifn = float(aucell(test_X, gs_ifn).mean()) - ctrl_auc

        # ---- CellOT per seed ----
        seed_pearson, seed_edist, seed_aucell, seed_mmd = [], [], [], []
        for seed in args.seeds:
            t0 = time.time()
            pred_genes, best_mmd = R.run_cellot_on_split(cs, sp, seed, args.ae_iters, args.cellot_iters)
            dt = time.time() - t0
            aligned = R.stratum_align(pred_genes, ctrl_strata, test_strata)
            pe = float(pearson_delta(aligned, test_X, ctrl_mean, test_strata)["macro"])
            ed = R.edist_clouds(pred_genes, ctrl_strata, test_X, test_strata, ed_basis)
            au = float(aucell(pred_genes, gs_ifn).mean()) - ctrl_auc
            seed_pearson.append(pe); seed_edist.append(ed); seed_aucell.append(au); seed_mmd.append(best_mmd)
            timing.append(dict(lineage=lin, seed=seed, sec=round(dt, 1), best_mmd=round(best_mmd, 5),
                               n_train=int(len(sp.train_idx)), n_test=int(len(sp.test_idx)),
                               n_ctrl=int(len(ctrl_idx))))
            print(f"  [{lin} seed{seed}] {dt:.0f}s pearsonD={pe:.4f} eDist={ed:.4f} "
                  f"aucellD={au:+.4f} mmd={best_mmd:.4f}", flush=True)

        # collapse seeds within unit (mean) -- seeds are technical repeats
        cellot_pe = float(np.mean(seed_pearson)); cellot_ed = float(np.mean(seed_edist))
        cellot_au = float(np.mean(seed_aucell))
        for metric, cellot_score, prim_name, prim_score, bof_name, bof_score, orient in [
            ("pearson_delta", cellot_pe, prim_pearson_name, b_pearson(prim_pearson_name),
             bof_pearson, b_pearson(bof_pearson), +1),
            ("e_distance", cellot_ed, prim_edist_name, b_edist(prim_edist_name),
             prim_edist_name, b_edist(prim_edist_name), -1),
            ("aucell_ifn_delta", cellot_au, "cell-mean", b_aucell("cell-mean"),
             "cell-mean", b_aucell("cell-mean"), None),
        ]:
            if metric == "aucell_ifn_delta":
                # magnitude recovery: |pred_delta - obs| ; lower is better; oriented so positive favours cellot
                delta_prim = abs(prim_score - obs_ifn) - abs(cellot_score - obs_ifn)
                delta_bof = delta_prim
            elif orient > 0:
                delta_prim = cellot_score - prim_score
                delta_bof = cellot_score - bof_score
            else:  # lower-better
                delta_prim = prim_score - cellot_score
                delta_bof = bof_score - cellot_score
            rows.append(dict(
                lineage=lin, metric=metric, cellot_score=round(cellot_score, 4),
                primary_baseline=prim_name, baseline_score=round(prim_score, 4),
                delta_vs_primary=round(delta_prim, 4),
                bestof4_baseline=bof_name, bestof4_score=round(bof_score, 4),
                delta_vs_bestof4=round(delta_bof, 4),
                obs_ifn_delta=(round(obs_ifn, 4) if metric == "aucell_ifn_delta" else ""),
                seed_scores=json.dumps([round(x, 4) for x in
                                        (seed_pearson if metric == "pearson_delta" else
                                         seed_edist if metric == "e_distance" else seed_aucell)]),
                seeds=",".join(map(str, args.seeds)),
                n_test=int(len(sp.test_idx)), n_ctrl=int(len(ctrl_idx)),
                n_strata=int(len(np.unique(test_strata))), best_mmd=round(float(np.mean(seed_mmd)), 5),
                leak_free=bool(audit["leak_free"]),
            ))
        pd.DataFrame(rows).to_csv(args.out, index=False)
        json.dump(timing, open(args.timing_out, "w"), indent=2)
    print(f"\nWROTE {args.out} ({len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    pe = df[df.metric == "pearson_delta"]
    print("\n=== Kang Pearson-delta CellOT vs primary baseline ===")
    print(pe[["lineage", "cellot_score", "primary_baseline", "baseline_score",
              "delta_vs_primary", "delta_vs_bestof4"]].to_string(index=False))
    print(f"\nmean delta_vs_primary = {pe.delta_vs_primary.mean():+.4f}; "
          f"%positive = {100*(pe.delta_vs_primary>0).mean():.0f}%")
    print(f"mean delta_vs_bestof4 = {pe.delta_vs_bestof4.mean():+.4f}; "
          f"%positive = {100*(pe.delta_vs_bestof4>0).mean():.0f}%")


IFN_LIST = ["ISG15", "IFI6", "MX1", "MX2", "OAS1", "OAS2", "IFIT1", "IFIT3", "ISG20",
            "STAT1", "IRF7", "IFI44", "IFI44L", "RSAD2", "USP18"]

if __name__ == "__main__":
    main()
