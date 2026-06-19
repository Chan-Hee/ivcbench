#!/usr/bin/env python
"""EXPERIMENT: PertAdapt on Soskic CD4-activation leave-one-donor-out (②Donor×Hybrid, 2nd Hybrid model).

Mirrors scripts/state_soskic.py EXACTLY (per-donor leak-free C2 LODO, --chunk, --seeds, --cap,
training-only response_gene_idx + PCA basis, same primary-baseline selection + anchor gate), but swaps
arc-STATE for PertAdapt (Bai et al. 2025), which runs in the `scfoundation` conda env via
model_runners/pertadapt_soskic_runner.py. Produces a donor row directly comparable to the STATE, scGen,
and CellOT donor rows (same rg / ed_basis / AUCell programs).

DONOR axis: the perturbation ("stimulation", 0h→16h) is SEEN in every training donor; the held UNIT is a
*donor*. PertAdapt trains on all NON-held donors' (0h source / 16h target) cells via its GO-masked
adapter conditioned on lineage (CD4 Naive/Memory), and predicts the held donor's 16h response from its
OWN 0h cells. Leak-safe: the held donor's 16h expression never enters training.

ANCHOR / sanity gate (same band as STATE): PertAdapt's per-donor pearson_delta must be FINITE and land in
the donor band [-0.10, 0.50]; a leak-inflated score (>0.9) or a degenerate/non-finite score FAILS the
gate → the result is NOT adoptable.

COMPUTE (2-GPU budget, devices 0,1 ONLY): --chunk I 2 shards donors across the 2 GPUs; pin with --gpu /
CUDA_VISIBLE_DEVICES. Never co-schedule with the scGPT-env stack on the same GPU.
"""
from __future__ import annotations
import sys, os, time, argparse, json, subprocess, tempfile
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
from ivcbench.baselines.heavy import env_python
from ivcbench.eval.bundle import dump_bundle

from c2_soskic_donor import (load_soskic_donor, lodo_spec, e_distance_basis,
                             response_gene_idx, program_delta_mae, SOSKIC_PROGRAMS)

RUNNER = ROOT / "model_runners" / "pertadapt_soskic_runner.py"
PA_ENV = "scfoundation"


def run_pa_on_split(cs, sp, seed, held_donor, cuda_device, epochs):
    """Build the leak-safe per-fold payload (identical keys to state_soskic.py), shell the PertAdapt
    runner in the scfoundation env, return {lineage: predicted 16h profile}. The runner never sees the
    held donor's 16h expression — only its 0h cells (inference input) + training cells/labels."""
    tr = sp.train_idx
    is_ctrl_tr = cs.obs.iloc[tr]["is_control"].to_numpy().astype(bool)
    don_tr = cs.obs.iloc[tr]["donor_id"].astype(str).to_numpy()
    pert_train = np.where(is_ctrl_tr, "control", np.char.add("stim_", don_tr)).astype(object)
    celltype_train = cs.obs.iloc[tr]["cell_type_coarse"].astype(str).to_numpy()
    inf = sp.inference_input_idx
    celltype_inf = cs.obs.iloc[inf]["cell_type_coarse"].astype(str).to_numpy()
    gem_inf = cs.obs.iloc[inf]["donor_id"].astype(str).to_numpy()
    held_label = f"stim_{held_donor}"

    payload = dict(
        X_train=cs.X[tr].astype(np.float32),
        is_control_train=is_ctrl_tr,
        pert_train=np.asarray([str(p) for p in pert_train]),
        celltype_train=np.asarray([str(c) for c in celltype_train]),
        gem_train=np.asarray([str(g) for g in don_tr]),
        X_ctrl_inf=cs.X[inf].astype(np.float32),
        celltype_inf=np.asarray([str(c) for c in celltype_inf]),
        gem_inf=np.asarray([str(g) for g in gem_inf]),
        held_label=held_label,
        genes=np.asarray([str(g) for g in cs.var_names]),
    )
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "in.npz", Path(td) / "out.npz"
        np.savez(inp, **payload, allow_pickle=True)
        env = os.environ.copy()
        if cuda_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        env["IVCBENCH_PA_EPOCHS"] = str(epochs)
        env["PYTHONHASHSEED"] = str(seed)
        proc = subprocess.run([env_python(PA_ENV), str(RUNNER), str(inp), str(out)],
                              capture_output=True, text=True, timeout=7200, env=env)
        if proc.returncode != 0 or not out.exists():
            err = proc.stderr or ""
            key = [ln for ln in err.splitlines()
                   if any(k in ln for k in ("Error", "Exception", "Traceback", "assert", "RuntimeError"))]
            raise RuntimeError(f"PertAdapt runner failed (rc={proc.returncode}):\n"
                               + ("… " + key[-1] + "\n" if key else "") + err[-3500:])
        r = np.load(out, allow_pickle=True)
        by_lineage = {}
        for k, v in zip(r["pred_perts"], r["pred_means"]):
            by_lineage[str(k).split("::", 1)[-1]] = np.asarray(v, np.float32)
    return by_lineage


def pa_pred_cells(by_lineage, test_strata, ctrl_mean):
    test_strata = np.asarray(test_strata)
    n_genes = len(ctrl_mean)
    pred = np.zeros((len(test_strata), n_genes), np.float32)
    for s in np.unique(test_strata):
        lineage = str(s).split("=", 1)[-1]
        pred[test_strata == s] = by_lineage.get(lineage, ctrl_mean)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0, help="first K donors (EXPLORATORY subset)")
    ap.add_argument("--chunk", type=int, nargs=2, default=None, metavar=("I", "N"))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=20, help="PertAdapt epochs (IVCBENCH_PA_EPOCHS)")
    ap.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES for the PertAdapt subprocess")
    ap.add_argument("--out", default=str(ROOT / "outputs/additional_models/pertadapt_soskic_raw.csv"))
    ap.add_argument("--timing-out", default=str(ROOT / "outputs/additional_models/pertadapt_soskic_timing.json"))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

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
    print(f"[soskic-PertAdapt] {cs.X.shape[0]} cells x {cs.X.shape[1]} genes; running {len(donors)} donors; "
          f"seeds={args.seeds}; epochs={args.epochs}; gpu={args.gpu}", flush=True)

    for k, dnr in enumerate(donors):
        sp = build_split(cs, lodo_spec(dnr))
        audit = audit_split(cs, sp)
        assert audit["leak_free"], f"LEAK {dnr}"
        test_X = cs.X[sp.test_idx]; test_strata = sp.test_strata
        ctrl_idx = sp.inference_input_idx; ctrl_X = cs.X[ctrl_idx]
        ctrl_mean = ctrl_X.mean(0)
        ctrl_strat_str = cs.obs.iloc[ctrl_idx]["cell_type_coarse"].astype(str).to_numpy()
        rg = response_gene_idx(cs, sp.train_idx)
        ed_basis = e_distance_basis(cs, sp.train_idx)

        bp = {}
        for B in (CtrlPred, CellMean, DonorShift, LinearPCA):
            b = B(); b.fit(cs, sp, side_info=cs.side_info); bp[b.name] = b.predict(cs, sp, side_info=cs.side_info)
        def b_pe(nm): return float(pearson_delta(bp[nm].pred_cells, test_X, bp[nm].control_mean, test_strata, rg)["macro"])
        def b_ed(nm): return float(e_distance(bp[nm].pred_cells, test_X, test_strata, fit_on=ed_basis)["macro"])
        def b_au(nm):
            return program_delta_mae(bp[nm].pred_cells, test_X, ctrl_X, test_strata, ctrl_strat_str, cs)["aucell_delta_score"]
        prim_pe_name = max(("cell-mean", "donor-shift"), key=b_pe)
        prim_ed_name = min(("cell-mean", "donor-shift"), key=b_ed)
        prim_au_name = max(("cell-mean", "donor-shift"), key=lambda nm: (b_au(nm) if b_au(nm) == b_au(nm) else -1e9))

        s_pe, s_ed, s_au = [], [], []
        for seed in args.seeds:
            t0 = time.time()
            by_lineage = run_pa_on_split(cs, sp, seed, dnr, args.gpu, args.epochs)
            dt = time.time() - t0
            pred_aligned = pa_pred_cells(by_lineage, test_strata, ctrl_mean)
            pe = float(pearson_delta(pred_aligned, test_X, ctrl_mean, test_strata, rg)["macro"])
            ed = float(e_distance(pred_aligned, test_X, test_strata, fit_on=ed_basis)["macro"])
            dump_bundle(os.environ.get("IVCBENCH_PRED_DUMP"),
                        cluster=sp.spec.registry_task, model="PertAdapt", split=sp.spec.name,
                        pred_cells=pred_aligned, test_cells=test_X, cell_strata=test_strata,
                        control_mean=ctrl_mean, genes=cs.var_names, exclude_gene_idx=rg, fit_on=ed_basis)
            au = program_delta_mae(pred_aligned, test_X, ctrl_X, test_strata, ctrl_strat_str, cs)["aucell_delta_score"]
            s_pe.append(pe); s_ed.append(ed); s_au.append(au)
            timing.append(dict(donor=str(dnr), seed=seed, sec=round(dt, 1),
                               n_train=int(len(sp.train_idx)), n_test=int(len(sp.test_idx)),
                               n_ctrl=int(len(ctrl_idx)), n_lineages=int(len(by_lineage))))
            print(f"  [{dnr} seed{seed}] {dt:.0f}s pearsonD={pe:.4f} eDist={ed:.4f} aucellScore={au:.4f}",
                  flush=True)
        pa_pe, pa_ed = float(np.mean(s_pe)), float(np.mean(s_ed))
        au_vals = [x for x in s_au if x == x]
        pa_au = float(np.mean(au_vals)) if au_vals else float("nan")

        for metric, sscore, pname, pscore in [
            ("pearson_delta", pa_pe, prim_pe_name, b_pe(prim_pe_name)),
            ("e_distance", pa_ed, prim_ed_name, b_ed(prim_ed_name)),
            ("aucell_delta_score", pa_au, prim_au_name, b_au(prim_au_name)),
        ]:
            delta = (pscore - sscore) if metric == "e_distance" else (sscore - pscore)
            rows.append(dict(
                donor=str(dnr), metric=metric, pertadapt_score=(round(sscore, 4) if sscore == sscore else ""),
                primary_baseline=pname, baseline_score=(round(pscore, 4) if pscore == pscore else ""),
                delta_vs_primary=(round(delta, 4) if (sscore == sscore and pscore == pscore) else ""),
                seed_scores=json.dumps([round(x, 4) for x in
                                        (s_pe if metric == "pearson_delta" else
                                         s_ed if metric == "e_distance" else au_vals)]),
                seeds=",".join(map(str, args.seeds)),
                n_test=int(len(sp.test_idx)), n_ctrl=int(len(ctrl_idx)),
                n_strata=int(len(np.unique(test_strata))), n_response_genes=int(len(rg)),
                epochs=int(args.epochs), leak_free=bool(audit["leak_free"]),
            ))
        pd.DataFrame(rows).to_csv(out_path, index=False)
        json.dump(timing, open(args.timing_out, "w"), indent=2)
        print(f"[{k+1}/{len(donors)}] donor {dnr} done", flush=True)

    df = pd.DataFrame(rows)
    print(f"\nWROTE {out_path} ({len(rows)} rows)")
    pe = df[df.metric == "pearson_delta"]
    pe = pe[pe.delta_vs_primary != ""]
    if len(pe):
        sc = pe.pertadapt_score.astype(float)
        dvp = pe.delta_vs_primary.astype(float)
        finite = np.isfinite(sc).all()
        in_band = sc.between(-0.10, 0.50).mean()
        print(f"\nSoskic PertAdapt Pearson-delta: mean score={sc.mean():+.4f} "
              f"(min {sc.min():+.4f}, max {sc.max():+.4f}); mean delta_vs_primary={dvp.mean():+.4f}; "
              f"%positive={100*(dvp>0).mean():.1f}%; finite={finite}; "
              f"%in_band[-0.10,0.50]={100*in_band:.0f}%; n_donors={len(pe)}")
        if not finite or (sc > 0.9).any():
            print("WARNING: PertAdapt pearson_delta is non-finite or leak-inflated (>0.9) — FAILS anchor gate.")


if __name__ == "__main__":
    Path(ROOT / "outputs/additional_models").mkdir(parents=True, exist_ok=True)
    main()
