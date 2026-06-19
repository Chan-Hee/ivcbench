#!/usr/bin/env python
"""EXPERIMENT (② Donor x Latent, 2nd latent model): CPA on Soskic CD4 activation leave-one-donor-out.

This is the SECOND latent model on the donor axis (scGen is the first, in c2_soskic_donor.py;
CellOT in cellot_soskic.py). It mirrors scripts/cellot_soskic.py exactly: it reuses
c2_soskic_donor.py:load_soskic_donor (same 106 donors, same caps, same shared-gene re-standardization)
and lodo_spec (same split). For each held donor, CPA trains on all NON-held donors (0h source / 16h
target), then predicts the held donor's 16h response from its OWN 0h cells via classic latent
δ-arithmetic in CPA's latent space (δ = mean(latent[stim]) − mean(latent[ctrl]); decode held-donor
ctrl + δ). Metrics = repo pearson_delta / e_distance (e_distance_basis builder) / SOSKIC AUCell
programs. Universal two-member floor = {cell-mean, linear-PCA} (a model must beat BOTH members);
donor-shift / training-mean are descriptive context comparators, not universal-floor members. Strata =
cell_type_coarse (CD4 Naive/Memory).

Unlike the in-process CellOT runner (scperturbench_eval env), CPA needs the dedicated `ivc-cpa` conda
env, so the per-fold model call shells out through ivcbench.baselines.heavy.CPAC1 (which invokes
model_runners/cpa_c1_runner.py — the dataset-agnostic seen-condition / held-group latent δ runner,
condition/cell_type obs, gene-side built in-runner, requires_gene_side=False). The CPAC1 adapter pins
the job's GPU via CUDA_VISIBLE_DEVICES (set here through --gpu), so --chunk I 2 + two --gpu launches
shard donors across exactly 2 GPUs.

LEAK-SAFETY (HARD): the held donor's 16h-stim cells never enter training. Response genes and the
E-distance PCA basis are selected per fold from TRAINING cells only and passed only to the metric call;
the CPA runner receives split.train_idx + the held donor's control cells and the held condition LABEL
only (never split.test_idx expression). audit_split asserts leak_free every fold.

ANCHOR / SANITY GATE: CPA pearson_delta must be finite and land within the scGen/CellOT donor band
(per-donor mean pearson_delta in [0.00, 0.45]; scGen mean≈0.144, CellOT mean≈0.369, cell-mean
baseline mean≈0.260). A CPA mean inside that band establishes Latent within-family replicability on the
donor axis (consistency with scGen). The full result is only adoptable if the gate passes.

COMPUTE: full 106 = target. --test K runs the first K donors (sorted) as EXPLORATORY only.
SMOKE: --test 1 --seeds 0 --cap 40 --cpa-epochs 2  (one donor, one seed, minimal CPA epochs, tiny cap)
just proves the path runs end-to-end and writes a sane per-unit row; it is NOT the full job.
"""
from __future__ import annotations
import sys, time, argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split
from ivcbench.metrics.response import pearson_delta
from ivcbench.metrics.distribution import e_distance
from ivcbench.baselines.simple import CellMean, DonorShift, CtrlPred, LinearPCA
from ivcbench.baselines.heavy import CPAC1

from c2_soskic_donor import (load_soskic_donor, lodo_spec, e_distance_basis,
                             response_gene_idx, program_delta_mae, SOSKIC_PROGRAMS)

# Latent-family donor band (scGen + CellOT). scGen mean≈0.144 (range [-0.10, 0.42]);
# CellOT mean≈0.369 (range [0.05, 0.69]); cell-mean baseline mean≈0.260. CPA, the 2nd latent model on
# this axis, must land its per-donor mean pearson_delta inside this band to count as within-family
# replicable. Lower bound a touch below scgen's mean-minus-slack; upper bound at cellot's mean.
ANCHOR_LO, ANCHOR_HI = 0.00, 0.45


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0, help="first K donors (EXPLORATORY subset)")
    ap.add_argument("--chunk", type=int, nargs=2, default=None, metavar=("I", "N"),
                    help="donor shard I of N (e.g. --chunk 0 2 / --chunk 1 2 across 2 GPUs)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--cpa-epochs", type=int, default=60, help="CPA max_epochs (IVCBENCH_CPA_EPOCHS)")
    ap.add_argument("--cpa-maxcells", type=int, default=None,
                    help="cap training cells fed to CPA (IVCBENCH_CPA_MAXCELLS)")
    ap.add_argument("--gpu", type=str, default=None,
                    help="CUDA device for this shard's CPA runner (CUDA_VISIBLE_DEVICES); pair with --chunk")
    ap.add_argument("--out", default=str(ROOT / "outputs/additional_models/cpa_soskic_raw.csv"))
    ap.add_argument("--timing-out", default=str(ROOT / "outputs/additional_models/cpa_soskic_timing.json"))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    if args.cpa_maxcells is not None:
        os.environ["IVCBENCH_CPA_MAXCELLS"] = str(args.cpa_maxcells)
    # CPA epochs are read by the runner from the env; set per seed below so seeds are reproducible.

    cs = load_soskic_donor(args.cap)
    donors = sorted(cs.obs.donor_id.unique())
    if args.test:
        donors = donors[: args.test]
    if args.chunk:
        i, n = args.chunk; donors = donors[i::n]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, timing, done = [], [], set()
    if args.skip_existing and out_path.exists():
        old = pd.read_csv(out_path); rows = old.to_dict("records")
        done = set(old["donor"].astype(str).unique())
        donors = [d for d in donors if d not in done]
        print(f"resume: {len(old)} rows; skipping {len(done)} donors", flush=True)
    if Path(args.timing_out).exists():
        try:
            timing = json.load(open(args.timing_out))
        except Exception:
            timing = []
    print(f"[soskic] {cs.X.shape[0]} cells x {cs.X.shape[1]} genes; running {len(donors)} donors; "
          f"seeds={args.seeds}; cpa_epochs={args.cpa_epochs}; gpu={args.gpu}; "
          f"chunk={args.chunk}", flush=True)

    for k, d in enumerate(donors):
        sp = build_split(cs, lodo_spec(d))
        audit = audit_split(cs, sp)
        assert audit["leak_free"], f"LEAK {d}"
        test_X = cs.X[sp.test_idx]; test_strata = sp.test_strata
        ctrl_idx = sp.inference_input_idx; ctrl_X = cs.X[ctrl_idx]
        ctrl_mean = ctrl_X.mean(0)
        # response genes + E-distance PCA basis: TRAINING-fold only (leak-safe), per-fold refit.
        rg = response_gene_idx(cs, sp.train_idx)
        ed_basis = e_distance_basis(cs, sp.train_idx)
        ctrl_strat_str = cs.obs.iloc[ctrl_idx]["cell_type_coarse"].astype(str).to_numpy()

        # ---- baselines (same as cellot_soskic.py) ----
        bp = {}
        for B in (CtrlPred, CellMean, DonorShift, LinearPCA):
            b = B(); b.fit(cs, sp, side_info=cs.side_info); bp[b.name] = b.predict(cs, sp, side_info=cs.side_info)
        def b_pe(n): return float(pearson_delta(bp[n].pred_cells, test_X, bp[n].control_mean, test_strata, rg)["macro"])
        def b_ed(n): return float(e_distance(bp[n].pred_cells, test_X, test_strata, fit_on=ed_basis)["macro"])
        def b_au(n):
            return program_delta_mae(bp[n].pred_cells, test_X, ctrl_X, test_strata, ctrl_strat_str, cs)["aucell_delta_score"]
        prim_pe_name = max(("cell-mean", "donor-shift"), key=b_pe)
        prim_ed_name = min(("cell-mean", "donor-shift"), key=b_ed)
        prim_au_name = max(("cell-mean", "donor-shift"), key=lambda n: (b_au(n) if b_au(n) == b_au(n) else -1e9))

        # ---- CPA per seed (latent δ-arithmetic via the ivc-cpa CPAC1 runner) ----
        s_pe, s_ed, s_au = [], [], []
        for seed in args.seeds:
            os.environ["IVCBENCH_CPA_EPOCHS"] = str(args.cpa_epochs)
            os.environ["IVCBENCH_CPA_SEED"] = str(seed)
            t0 = time.time()
            model = CPAC1()
            if args.gpu is not None:
                model.cuda_device = args.gpu   # pins CUDA_VISIBLE_DEVICES for this shard's runner
            model.fit(cs, sp, side_info=cs.side_info)
            pr = model.predict(cs, sp, side_info=cs.side_info)
            dt = time.time() - t0
            # CPAC1 returns one predicted 16h profile (the stim label), tiled onto every test cell by
            # the SubprocessAdapter; pred_cells is already row-aligned to sp.test_idx. control_mean is
            # the held-donor 0h mean. pearson_delta/program metrics consume that directly.
            pe = float(pearson_delta(pr.pred_cells, test_X, pr.control_mean, test_strata, rg)["macro"])
            ed = float(e_distance(pr.pred_cells, test_X, test_strata, fit_on=ed_basis)["macro"])
            au = program_delta_mae(pr.pred_cells, test_X, ctrl_X, test_strata, ctrl_strat_str, cs)["aucell_delta_score"]
            s_pe.append(pe); s_ed.append(ed); s_au.append(au)
            timing.append(dict(donor=str(d), seed=seed, sec=round(dt, 1),
                               n_train=int(len(sp.train_idx)), n_test=int(len(sp.test_idx)),
                               n_ctrl=int(len(ctrl_idx))))
            print(f"  [{d} seed{seed}] {dt:.0f}s pearsonD={pe:.4f} eDist={ed:.4f} "
                  f"aucellScore={au:.4f}", flush=True)
        cpa_pe, cpa_ed = float(np.mean(s_pe)), float(np.mean(s_ed))
        au_vals = [x for x in s_au if x == x]
        cpa_au = float(np.mean(au_vals)) if au_vals else float("nan")

        for metric, cscore, pname, pscore, orient in [
            ("pearson_delta", cpa_pe, prim_pe_name, b_pe(prim_pe_name), +1),
            ("e_distance", cpa_ed, prim_ed_name, b_ed(prim_ed_name), -1),
            ("aucell_delta_score", cpa_au, prim_au_name, b_au(prim_au_name), +1),
        ]:
            if metric == "e_distance":
                delta = pscore - cscore   # lower-better -> positive favours cpa
            else:
                delta = cscore - pscore
            rows.append(dict(
                donor=str(d), metric=metric, cpa_score=(round(cscore, 4) if cscore == cscore else ""),
                primary_baseline=pname, baseline_score=(round(pscore, 4) if pscore == pscore else ""),
                delta_vs_primary=(round(delta, 4) if (cscore == cscore and pscore == pscore) else ""),
                seed_scores=json.dumps([round(x, 4) for x in
                                        (s_pe if metric == "pearson_delta" else
                                         s_ed if metric == "e_distance" else au_vals)]),
                seeds=",".join(map(str, args.seeds)),
                n_test=int(len(sp.test_idx)), n_ctrl=int(len(ctrl_idx)),
                n_strata=int(len(np.unique(test_strata))), n_response_genes=int(len(rg)),
                cpa_epochs=int(args.cpa_epochs), leak_free=bool(audit["leak_free"]),
            ))
        pd.DataFrame(rows).to_csv(out_path, index=False)
        json.dump(timing, open(args.timing_out, "w"), indent=2)
        print(f"[{k+1}/{len(donors)}] donor {d} done", flush=True)

    df = pd.DataFrame(rows)
    print(f"\nWROTE {out_path} ({len(rows)} rows)")
    pe = df[df.metric == "pearson_delta"]
    pe = pe[pe.cpa_score != ""]
    if len(pe):
        cpa = pe.cpa_score.astype(float)
        dvp = pe[pe.delta_vs_primary != ""].delta_vs_primary.astype(float)
        mean_pe = float(cpa.mean())
        in_band = ANCHOR_LO <= mean_pe <= ANCHOR_HI
        print(f"\nSoskic CPA Pearson-delta: mean cpa_score={mean_pe:+.4f} "
              f"(per-donor range [{cpa.min():.4f}, {cpa.max():.4f}]); "
              f"mean delta_vs_primary={dvp.mean():+.4f}; n_donors={len(pe)}")
        print(f"ANCHOR GATE [{ANCHOR_LO:.2f}, {ANCHOR_HI:.2f}] (scGen/CellOT donor band): "
              f"{'PASS' if (in_band and np.isfinite(mean_pe)) else 'FAIL'} "
              f"(finite={bool(np.isfinite(mean_pe))}, in_band={in_band})")


if __name__ == "__main__":
    main()
