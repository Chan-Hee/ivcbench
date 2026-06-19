#!/usr/bin/env python
"""EXPERIMENT A2: CellOT on Soskic CD4 activation leave-one-donor-out (LODO).

Reuses scripts/c2_soskic_donor.py:load_soskic_donor (same 106 donors, same caps, same shared-gene
re-standardization) and lodo_spec (same split). For each held donor CellOT (scgen AE + f/g ICNN) trains
on all NON-held donors (0h source / 16h target) and predicts the held donor's 16h response from its OWN
0h cells. Metrics = repo pearson_delta / e_distance (e_distance_basis builder) / SOSKIC AUCell programs.
Primary baseline = matched simple context baseline (cell-mean / donor-shift; not a universal-floor member). Strata = cell_type_coarse (CD4 Naive/Memory).

COMPUTE: full 106 = target. --test K runs the first K donors (sorted) as EXPLORATORY only.
"""
from __future__ import annotations
import sys, time, argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split
from ivcbench.metrics.response import pearson_delta
from ivcbench.metrics.distribution import e_distance
from ivcbench.metrics.program import aucell
from ivcbench.baselines.simple import CellMean, DonorShift, CtrlPred, LinearPCA

import cellot_runner as R
from c2_soskic_donor import (load_soskic_donor, lodo_spec, e_distance_basis,
                             response_gene_idx, program_delta_mae, SOSKIC_PROGRAMS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0, help="first K donors (EXPLORATORY subset)")
    ap.add_argument("--chunk", type=int, nargs=2, default=None, metavar=("I", "N"))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--ae-iters", type=int, default=12000)
    ap.add_argument("--cellot-iters", type=int, default=8000)
    ap.add_argument("--out", default=str(ROOT / "outputs/additional_models/cellot_soskic_raw.csv"))
    ap.add_argument("--timing-out", default=str(ROOT / "outputs/additional_models/cellot_soskic_timing.json"))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    print(f"[device] {R.DEVICE}", flush=True)
    cs = load_soskic_donor(args.cap)
    donors = sorted(cs.obs.donor_id.unique())
    if args.test:
        donors = donors[: args.test]
    if args.chunk:
        i, n = args.chunk; donors = donors[i::n]

    out_path = Path(args.out)
    rows, timing, done = [], [], set()
    if args.skip_existing and out_path.exists():
        old = pd.read_csv(out_path); rows = old.to_dict("records")
        done = set(old["donor"].astype(str).unique())
        donors = [d for d in donors if d not in done]
        print(f"resume: {len(old)} rows; skipping {len(done)} donors", flush=True)
    if (Path(args.timing_out).exists()):
        try:
            timing = json.load(open(args.timing_out))
        except Exception:
            timing = []
    print(f"[soskic] {cs.X.shape[0]} cells x {cs.X.shape[1]} genes; running {len(donors)} donors; "
          f"seeds={args.seeds}", flush=True)

    for k, d in enumerate(donors):
        sp = build_split(cs, lodo_spec(d))
        audit = audit_split(cs, sp)
        assert audit["leak_free"], f"LEAK {d}"
        test_X = cs.X[sp.test_idx]; test_strata = sp.test_strata
        ctrl_idx = sp.inference_input_idx; ctrl_X = cs.X[ctrl_idx]
        ctrl_strata = np.array([f"cell_type_coarse={cs.obs.iloc[i]['cell_type_coarse']}"
                                for i in ctrl_idx], dtype=object)
        ctrl_mean = ctrl_X.mean(0)
        rg = response_gene_idx(cs, sp.train_idx)
        ed_basis = e_distance_basis(cs, sp.train_idx)

        # ---- baselines ----
        bp = {}
        for B in (CtrlPred, CellMean, DonorShift, LinearPCA):
            b = B(); b.fit(cs, sp, side_info=cs.side_info); bp[b.name] = b.predict(cs, sp, side_info=cs.side_info)
        def b_pe(n): return float(pearson_delta(bp[n].pred_cells, test_X, bp[n].control_mean, test_strata, rg)["macro"])
        def b_ed(n): return float(e_distance(bp[n].pred_cells, test_X, test_strata, fit_on=ed_basis)["macro"])
        def b_au(n):
            ctrl_strat = cs.obs.iloc[ctrl_idx]["cell_type_coarse"].astype(str).to_numpy()
            return program_delta_mae(bp[n].pred_cells, test_X, ctrl_X, test_strata, ctrl_strat, cs)["aucell_delta_score"]
        prim_pe_name = max(("cell-mean", "donor-shift"), key=b_pe)
        prim_ed_name = min(("cell-mean", "donor-shift"), key=b_ed)
        prim_au_name = max(("cell-mean", "donor-shift"), key=lambda n: (b_au(n) if b_au(n) == b_au(n) else -1e9))

        # ---- CellOT per seed ----
        s_pe, s_ed, s_au, s_mmd = [], [], [], []
        ctrl_strat_str = cs.obs.iloc[ctrl_idx]["cell_type_coarse"].astype(str).to_numpy()
        for seed in args.seeds:
            t0 = time.time()
            pred_genes, best_mmd = R.run_cellot_on_split(cs, sp, seed, args.ae_iters, args.cellot_iters)
            dt = time.time() - t0
            aligned = R.stratum_align(pred_genes, ctrl_strata, test_strata)
            pe = float(pearson_delta(aligned, test_X, ctrl_mean, test_strata, rg)["macro"])
            ed = R.edist_clouds(pred_genes, ctrl_strata, test_X, test_strata, ed_basis)
            au = program_delta_mae(aligned, test_X, ctrl_X, test_strata, ctrl_strat_str, cs)["aucell_delta_score"]
            s_pe.append(pe); s_ed.append(ed); s_au.append(au); s_mmd.append(best_mmd)
            timing.append(dict(donor=str(d), seed=seed, sec=round(dt, 1), best_mmd=round(best_mmd, 5),
                               n_train=int(len(sp.train_idx)), n_test=int(len(sp.test_idx)),
                               n_ctrl=int(len(ctrl_idx))))
            print(f"  [{d} seed{seed}] {dt:.0f}s pearsonD={pe:.4f} eDist={ed:.4f} "
                  f"aucellScore={au:.4f} mmd={best_mmd:.4f}", flush=True)
        cellot_pe, cellot_ed = float(np.mean(s_pe)), float(np.mean(s_ed))
        au_vals = [x for x in s_au if x == x]
        cellot_au = float(np.mean(au_vals)) if au_vals else float("nan")

        for metric, cscore, pname, pscore, orient in [
            ("pearson_delta", cellot_pe, prim_pe_name, b_pe(prim_pe_name), +1),
            ("e_distance", cellot_ed, prim_ed_name, b_ed(prim_ed_name), -1),
            ("aucell_delta_score", cellot_au, prim_au_name, b_au(prim_au_name), +1),
        ]:
            if metric == "e_distance":
                delta = pscore - cscore   # lower-better -> positive favours cellot
            else:
                delta = cscore - pscore
            rows.append(dict(
                donor=str(d), metric=metric, cellot_score=(round(cscore, 4) if cscore == cscore else ""),
                primary_baseline=pname, baseline_score=(round(pscore, 4) if pscore == pscore else ""),
                delta_vs_primary=(round(delta, 4) if (cscore == cscore and pscore == pscore) else ""),
                seed_scores=json.dumps([round(x, 4) for x in
                                        (s_pe if metric == "pearson_delta" else
                                         s_ed if metric == "e_distance" else au_vals)]),
                seeds=",".join(map(str, args.seeds)),
                n_test=int(len(sp.test_idx)), n_ctrl=int(len(ctrl_idx)),
                n_strata=int(len(np.unique(test_strata))), n_response_genes=int(len(rg)),
                best_mmd=round(float(np.mean(s_mmd)), 5), leak_free=bool(audit["leak_free"]),
            ))
        pd.DataFrame(rows).to_csv(out_path, index=False)
        json.dump(timing, open(args.timing_out, "w"), indent=2)
        print(f"[{k+1}/{len(donors)}] donor {d} done", flush=True)

    df = pd.DataFrame(rows)
    print(f"\nWROTE {out_path} ({len(rows)} rows)")
    pe = df[df.metric == "pearson_delta"]
    pe = pe[pe.delta_vs_primary != ""]
    if len(pe):
        dvp = pe.delta_vs_primary.astype(float)
        print(f"\nSoskic Pearson-delta: mean delta_vs_primary={dvp.mean():+.4f}; "
              f"%positive={100*(dvp>0).mean():.1f}%; n_donors={len(pe)}")


if __name__ == "__main__":
    main()
