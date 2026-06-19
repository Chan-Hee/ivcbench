#!/usr/bin/env python
"""CellOT donor-transfer LEARNING CURVE on Soskic CD4 activation (C2).

Substantiates the "data is rate-limiting" claim: how does CellOT's leave-one-donor-out activation
transfer scale with the NUMBER of TRAINING donors? We FIX a held-out evaluation set of donors, then
train CellOT on increasing-size random subsets of the REMAINING training donors (the grid), at >=1
seed each, and score each eval donor's 16h response from its OWN 0h cells through the trained g.

Reuses the EXACT CellOT Soskic infrastructure:
  * scripts/c2_soskic_donor.py:load_soskic_donor (same 106 donors, caps, shared-gene re-standardization)
  * scripts/c2_soskic_donor.py response_gene_idx / e_distance_basis / program_delta_mae / SOSKIC_PROGRAMS
  * cellot_runner.py run_cellot_on_split (scgen AE + f/g ICNN, faithful Bunne 2023 CellOT)

Method (leak-safe):
  - EVAL donors: a fixed deterministic sample of `--n-eval` donors (seed-independent), NEVER trained on.
  - TRAIN pool: the other donors. For each grid size k we draw a random k-subset (seeded), restrict the
    split's train_idx to ONLY those donors' cells, train CellOT, and evaluate on EVERY eval donor.
  - Each eval donor is scored exactly as in cellot_soskic.py: encode its 0h cells, push through g,
    decode, then pearson_delta (training-only response genes from the TRAIN-subset) / e_distance
    (PCA basis from the TRAIN-subset) / AUCell program delta. Primary baseline per eval donor = better
    of cell-mean / donor-shift (same as cellot_soskic.py), fit on the SAME train subset.

One CellOT model is trained per (grid_size, seed) and reused across all eval donors -> the curve is the
mean over eval donors at each grid size. Output CSV is one row per (grid_size, seed, eval_donor, metric).

DEVICE: respects CUDA_VISIBLE_DEVICES (set by the launcher to a SINGLE physical GPU, 0 or 1).
RESUMABILITY: --skip-existing skips (grid_size, seed) cells already fully present in --out.
"""
from __future__ import annotations
import sys, time, argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ivcbench.splits.builder import Split
from ivcbench.metrics.response import pearson_delta
from ivcbench.baselines.simple import CellMean, DonorShift

import cellot_runner as R
from c2_soskic_donor import (load_soskic_donor, lodo_spec, e_distance_basis,
                             response_gene_idx, program_delta_mae)


def build_subset_split(cs, eval_donor, train_donors):
    """Leave-one-donor-out split for `eval_donor`, but with train_idx RESTRICTED to `train_donors`.

    test = eval_donor 16h cells; inference input = eval_donor 0h cells; train = (0h+16h) cells of the
    given train_donors ONLY. Built directly (not via build_split) so the training pool is the subset.
    The held eval donor is never in train_donors, so the split stays leak-free by construction.
    """
    obs = cs.obs
    donor = obs["donor_id"].to_numpy()
    is_ctrl = obs["is_control"].to_numpy().astype(bool)
    in_eval = donor == eval_donor
    in_train_pool = np.isin(donor, np.asarray(train_donors, dtype=object))
    assert eval_donor not in set(train_donors), "eval donor leaked into train pool"

    spec = lodo_spec(eval_donor)
    train_idx = np.where(in_train_pool)[0]
    test_idx = np.where(in_eval & ~is_ctrl)[0]
    inference_input_idx = np.where(in_eval & is_ctrl)[0]
    test_strata = np.array([spec.stratum_key(obs.iloc[i]) for i in test_idx], dtype=object)
    return Split(spec, train_idx, test_idx, inference_input_idx, test_strata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs="*", default=[8, 16, 32, 64, 96],
                    help="numbers of TRAINING donors to sweep")
    ap.add_argument("--n-eval", type=int, default=10, help="fixed held-out eval donors")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--ae-iters", type=int, default=12000)
    ap.add_argument("--cellot-iters", type=int, default=8000)
    ap.add_argument("--out", default=str(ROOT / "results/newdata/cellot_donor_learning_curve.csv"))
    ap.add_argument("--timing-out",
                    default=str(ROOT / "results/newdata/cellot_donor_learning_curve_timing.json"))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    print(f"[device] {R.DEVICE}", flush=True)
    cs = load_soskic_donor(args.cap)
    all_donors = sorted(cs.obs.donor_id.unique())
    n_total = len(all_donors)

    # FIXED eval donors: deterministic, seed-independent (a fixed evenly-spaced sample so the eval
    # set is held constant across the ENTIRE curve and across seeds).
    eval_rng = np.random.default_rng(20240607)
    eval_donors = sorted(eval_rng.choice(all_donors, size=args.n_eval, replace=False).tolist())
    train_pool = [d for d in all_donors if d not in set(eval_donors)]
    max_k = len(train_pool)
    grid = sorted({min(k, max_k) for k in args.grid})
    print(f"[soskic] {cs.X.shape[0]} cells x {cs.X.shape[1]} genes; donors_total={n_total}; "
          f"n_eval={len(eval_donors)}; train_pool={max_k}; grid={grid}; seeds={args.seeds}", flush=True)
    print(f"[eval_donors] {eval_donors}", flush=True)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, timing = [], []
    done_cells = set()  # (grid_size, seed)
    if args.skip_existing and out_path.exists():
        old = pd.read_csv(out_path)
        rows = old.to_dict("records")
        for (gs, sd), sub in old.groupby(["n_train_donors", "seed"]):
            # a cell is complete when all eval donors x 3 metrics are present
            if len(sub) >= len(eval_donors) * 3:
                done_cells.add((int(gs), int(sd)))
        print(f"resume: {len(old)} rows; skipping {len(done_cells)} complete (grid,seed) cells",
              flush=True)
    if Path(args.timing_out).exists():
        try:
            timing = json.load(open(args.timing_out))
        except Exception:
            timing = []

    # Pre-fit the two simple baselines per eval donor on each train subset (depends only on subset).
    for seed in args.seeds:
        sub_rng = np.random.default_rng(1000 + seed)
        for k in grid:
            if (k, seed) in done_cells:
                print(f"  skip grid={k} seed={seed} (already complete)", flush=True)
                continue
            # random k-subset of the train pool (seed+k controlled so each cell is reproducible)
            pick_rng = np.random.default_rng(7919 * seed + 31 * k)
            train_donors = sorted(pick_rng.choice(train_pool, size=k, replace=False).tolist())
            train_set = set(train_donors)

            # Train CellOT ONCE on this donor subset; the AE+ICNN transport is donor-agnostic (source =
            # all subset 0h cells, target = all subset 16h cells), so the SAME model is reused to score
            # every fixed eval donor by pushing its OWN 0h cells. This isolates the effect of training-
            # data size and keeps each grid point to a single training (cheap + reproducible). The
            # train_idx are the subset's cells; NO eval-donor cell is ever in the pool (leak-free).
            train_idx_global = np.where(np.isin(cs.obs["donor_id"].to_numpy(),
                                                np.asarray(train_donors, dtype=object)))[0]
            assert not (set(cs.obs.iloc[train_idx_global]["donor_id"]) & set(eval_donors)), \
                "eval donor leaked into training subset"
            t0 = time.time()
            ae, g, best_mmd = R.train_cellot_model(cs, train_idx_global, seed,
                                                   args.ae_iters, args.cellot_iters)
            print(f"  [trained grid={k} seed={seed}] {time.time()-t0:.0f}s "
                  f"n_train_cells={len(train_idx_global)} n_train_donors={k} mmd={best_mmd:.4f}",
                  flush=True)

            for j, ed in enumerate(eval_donors):
                sp = build_subset_split(cs, ed, train_donors)
                # leak-free guarantee: eval donor cells absent from train_idx
                assert not np.isin(sp.train_idx, np.concatenate([sp.test_idx, sp.inference_input_idx])).any()
                test_X = cs.X[sp.test_idx]; test_strata = sp.test_strata
                ctrl_idx = sp.inference_input_idx; ctrl_X = cs.X[ctrl_idx]
                ctrl_mean = ctrl_X.mean(0)
                ctrl_strata = np.array([f"cell_type_coarse={cs.obs.iloc[i]['cell_type_coarse']}"
                                        for i in ctrl_idx], dtype=object)
                ctrl_strat_str = cs.obs.iloc[ctrl_idx]["cell_type_coarse"].astype(str).to_numpy()
                rg = response_gene_idx(cs, sp.train_idx)
                ed_basis = e_distance_basis(cs, sp.train_idx)

                # score this eval donor with the SHARED trained model (push its own 0h cells)
                tA = time.time()
                pred_genes = R.predict_cellot(ae, g, ctrl_X)
                dt = time.time() - tA
                aligned = R.stratum_align(pred_genes, ctrl_strata, test_strata)
                co_pe = float(pearson_delta(aligned, test_X, ctrl_mean, test_strata, rg)["macro"])
                co_ed = R.edist_clouds(pred_genes, ctrl_strata, test_X, test_strata, ed_basis)
                co_au = program_delta_mae(aligned, test_X, ctrl_X, test_strata, ctrl_strat_str,
                                          cs)["aucell_delta_score"]

                # baselines on the SAME subset
                bp = {}
                for B in (CellMean, DonorShift):
                    b = B(); b.fit(cs, sp, side_info=cs.side_info)
                    bp[b.name] = b.predict(cs, sp, side_info=cs.side_info)
                b_pe = {n: float(pearson_delta(bp[n].pred_cells, test_X, bp[n].control_mean,
                                               test_strata, rg)["macro"]) for n in bp}
                from ivcbench.metrics.distribution import e_distance
                b_ed = {n: float(e_distance(bp[n].pred_cells, test_X, test_strata,
                                            fit_on=ed_basis)["macro"]) for n in bp}
                b_au = {n: program_delta_mae(bp[n].pred_cells, test_X, ctrl_X, test_strata,
                                             ctrl_strat_str, cs)["aucell_delta_score"] for n in bp}
                prim_pe = max(b_pe, key=b_pe.get)
                prim_ed = min(b_ed, key=b_ed.get)
                prim_au = max(b_au, key=lambda n: (b_au[n] if b_au[n] == b_au[n] else -1e9))

                for metric, cscore, pname, pscore in [
                    ("pearson_delta", co_pe, prim_pe, b_pe[prim_pe]),
                    ("e_distance", co_ed, prim_ed, b_ed[prim_ed]),
                    ("aucell_delta_score", co_au, prim_au, b_au[prim_au]),
                ]:
                    if metric == "e_distance":
                        delta = pscore - cscore  # lower-better -> positive favours cellot
                    else:
                        delta = cscore - pscore
                    ok = (cscore == cscore) and (pscore == pscore)
                    rows.append(dict(
                        n_train_donors=int(k), seed=int(seed), eval_donor=str(ed), metric=metric,
                        cellot_score=(round(cscore, 4) if cscore == cscore else ""),
                        primary_baseline=pname,
                        baseline_score=(round(pscore, 4) if pscore == pscore else ""),
                        delta_vs_primary=(round(delta, 4) if ok else ""),
                        best_mmd=round(float(best_mmd), 5),
                        n_train_cells=int(len(sp.train_idx)), n_test=int(len(sp.test_idx)),
                        n_ctrl=int(len(ctrl_idx)), n_response_genes=int(len(rg)),
                    ))
                timing.append(dict(n_train_donors=int(k), seed=int(seed), eval_donor=str(ed),
                                   sec=round(dt, 1), best_mmd=round(float(best_mmd), 5),
                                   n_train_cells=int(len(sp.train_idx))))
                print(f"  [grid={k} seed={seed} eval={ed} {j+1}/{len(eval_donors)}] "
                      f"{dt:.0f}s pearsonD={co_pe:.4f} eDist={co_ed:.4f} aucell={co_au:.4f} "
                      f"mmd={best_mmd:.4f} (vs {prim_pe} pe={b_pe[prim_pe]:.4f})", flush=True)
                # checkpoint after every (eval donor) so the job is fully resumable
                pd.DataFrame(rows).to_csv(out_path, index=False)
                json.dump(timing, open(args.timing_out, "w"), indent=2)
            print(f"[cell grid={k} seed={seed}] done in {time.time()-t0:.0f}s "
                  f"(n_train_cells={len(train_idx_global)})", flush=True)

    df = pd.DataFrame(rows)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWROTE {out_path} ({len(df)} rows)", flush=True)
    pe = df[(df.metric == "pearson_delta") & (df.delta_vs_primary != "")].copy()
    if len(pe):
        pe["delta_vs_primary"] = pe["delta_vs_primary"].astype(float)
        pe["cellot_score"] = pe["cellot_score"].astype(float)
        g = pe.groupby("n_train_donors").agg(
            cellot=("cellot_score", "mean"), delta=("delta_vs_primary", "mean"),
            n=("delta_vs_primary", "size"))
        print("\nLearning curve (pearson_delta, mean over eval donors x seeds):")
        print(g.to_string())


if __name__ == "__main__":
    main()
