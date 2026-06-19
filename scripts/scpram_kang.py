#!/usr/bin/env python
"""EXPERIMENT: scPRAM on Kang IFN-beta PBMC leave-one-lineage-out (C1 LOCT, 2nd OT model).

Mirrors scripts/cellot_kang.py EXACTLY (same repo split c1.coarse_loct, same lineages>=50-treated rule,
same primary/best-of-4 baselines, same pearson_delta / e_distance / type-I-IFN AUCell-delta scoring and
orientation), but swaps in-process CellOT for scPRAM (Jiang et al. 2024), the 2nd conditioned
Optimal-Transport-family model. scPRAM runs in the `ivc-scpram` conda env via
model_runners/scpram_runner.py: it trains a VAE + OT cell-matching + per-cell attention on the train
fold and predicts the held lineage's IFN-beta-stimulated MEAN profile from that lineage's OWN control
cells. Produces a lineage row directly comparable to the CellOT / scGen / CPA Kang rows.

LEAK-SAFETY: the held lineage's stimulated cells never enter the runner; only its control cells (the
inference input) are passed for prediction. The VAE + OT matching refit per fold (one subprocess per
held lineage). Same leak audit as cellot_kang.py.

COMPUTE (2-GPU budget, devices 0,1 ONLY): pin with --gpu / CUDA_VISIBLE_DEVICES; the 8 Kang lineages
run sequentially (each is seconds-to-minutes of GPU training). Never co-schedule on a busy GPU.
"""
from __future__ import annotations
import sys, os, time, argparse, json, subprocess, tempfile
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
from ivcbench.baselines.heavy import env_python
from ivcbench.eval.bundle import dump_bundle

RUNNER = ROOT / "model_runners" / "scpram_runner.py"
SCPRAM_ENV = "ivc-scpram"

IFN_LIST = ["ISG15", "IFI6", "MX1", "MX2", "OAS1", "OAS2", "IFIT1", "IFIT3", "ISG20",
            "STAT1", "IRF7", "IFI44", "IFI44L", "RSAD2", "USP18"]


def lineages_for(cs):
    ct = cs.obs["cell_type_coarse"].astype(str)
    is_ctrl = cs.obs["is_control"].astype(bool)
    treated = ct[~is_ctrl]
    return [l for l in sorted(ct.unique()) if int((treated == l).sum()) >= 50]


def baseline_preds(cs, sp):
    out = {}
    for B in (CtrlPred, CellMean, DonorShift, LinearPCA):
        b = B(); b.fit(cs, sp, side_info=cs.side_info); out[b.name] = b.predict(cs, sp, side_info=cs.side_info)
    return out


def run_scpram_on_split(cs, sp, seed, cuda_device, epochs, ratio):
    """Build the leak-safe per-fold payload (same keys the SubprocessAdapter builds), shell the scPRAM
    runner in the ivc-scpram env, return the predicted held-lineage stimulated MEAN profile (n_genes,).
    The runner never sees the held lineage's stimulated expression — only its control cells."""
    tr = sp.train_idx
    is_ctrl_tr = cs.obs.iloc[tr]["is_control"].to_numpy().astype(bool)
    pert_tr = cs.obs.iloc[tr]["perturbation"].astype(str).to_numpy()
    test_perts = cs.obs.iloc[sp.test_idx]["perturbation"].astype(str).to_numpy()
    payload = dict(
        X_train=cs.X[tr].astype(np.float32),
        is_control_train=is_ctrl_tr,
        pert_train=np.asarray([str(p) for p in pert_tr]),
        X_ctrl_inf=cs.X[sp.inference_input_idx].astype(np.float32),
        test_perts=np.asarray([str(p) for p in test_perts]),
        genes=np.asarray([str(g) for g in cs.var_names]),
    )
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "in.npz", Path(td) / "out.npz"
        np.savez(inp, **payload, allow_pickle=True)
        env = os.environ.copy()
        if cuda_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        env["IVCBENCH_SEED"] = str(seed)
        env["IVCBENCH_SCPRAM_EPOCHS"] = str(epochs)
        env["IVCBENCH_SCPRAM_RATIO"] = str(ratio)
        env["PYTHONHASHSEED"] = str(seed)
        proc = subprocess.run([env_python(SCPRAM_ENV), str(RUNNER), str(inp), str(out)],
                              capture_output=True, text=True, timeout=7200, env=env)
        if proc.returncode != 0 or not out.exists():
            err = proc.stderr or ""
            key = [ln for ln in err.splitlines()
                   if any(k in ln for k in ("Error", "Exception", "Traceback", "assert", "RuntimeError"))]
            raise RuntimeError(f"scPRAM runner failed (rc={proc.returncode}):\n"
                               + ("… " + key[-1] + "\n" if key else "") + err[-3500:])
        r = np.load(out, allow_pickle=True)
        return np.asarray(r["pred_means"][0], np.float32)       # held-lineage stimulated mean profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineages", nargs="*", default=None, help="subset of lineages (default all)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--ratio", type=float, default=0.005)
    ap.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES for the scPRAM subprocess")
    ap.add_argument("--out", default=str(ROOT / "outputs/additional_models/scpram_kang_raw.csv"))
    ap.add_argument("--timing-out", default=str(ROOT / "outputs/additional_models/scpram_kang_timing.json"))
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cs = kang_mod.load()
    gs_ifn = np.asarray(cs.gene_index(IFN_LIST), dtype=int)
    lins = args.lineages or lineages_for(cs)
    print(f"[kang-scPRAM] {cs.n_cells} cells x {cs.n_genes} genes; lineages={lins}; gpu={args.gpu}",
          flush=True)

    rows, timing = [], []
    for lin in lins:
        sp = build_split(cs, c1.coarse_loct(lin))
        audit = audit_split(cs, sp)
        assert audit["leak_free"], f"LEAK {lin}"
        test_X = cs.X[sp.test_idx]
        test_strata = sp.test_strata
        ctrl_idx = sp.inference_input_idx
        ctrl_X = cs.X[ctrl_idx]
        ctrl_mean = ctrl_X.mean(0)
        ed_basis = cs.X[sp.train_idx]
        if len(ed_basis) > 5000:
            ed_basis = ed_basis[np.random.default_rng(0).choice(len(ed_basis), 5000, replace=False)]
        ctrl_auc = float(aucell(ctrl_X, gs_ifn).mean())

        bpreds = baseline_preds(cs, sp)
        def b_pearson(name):
            p = bpreds[name]
            return float(pearson_delta(p.pred_cells, test_X, p.control_mean, test_strata)["macro"])
        def b_edist(name):
            p = bpreds[name]
            return float(e_distance(p.pred_cells, test_X, test_strata, fit_on=ed_basis)["macro"])
        def b_aucell(name):
            p = bpreds[name]
            return float(aucell(p.pred_cells, gs_ifn).mean()) - ctrl_auc
        prim_pearson_name = max(("cell-mean", "donor-shift"), key=b_pearson)
        prim_edist_name = min(("cell-mean", "donor-shift"), key=b_edist)
        bof_pearson = max(("ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"), key=b_pearson)
        obs_ifn = float(aucell(test_X, gs_ifn).mean()) - ctrl_auc

        # ---- scPRAM per seed (subprocess) ----
        seed_pearson, seed_edist, seed_aucell = [], [], []
        for seed in args.seeds:
            t0 = time.time()
            prof = run_scpram_on_split(cs, sp, seed, args.gpu, args.epochs, args.ratio)
            dt = time.time() - t0
            # scPRAM emits the held lineage's stimulated MEAN profile → tile to all test cells (the
            # per-stratum mean used by pearson_delta is invariant to tiling; E-dist compares the cloud
            # against this mean prediction, the same convention as a mean-shift baseline).
            pred_cells = np.tile(prof[None, :], (len(test_strata), 1)).astype(np.float32)
            pe = float(pearson_delta(pred_cells, test_X, ctrl_mean, test_strata)["macro"])
            ed = float(e_distance(pred_cells, test_X, test_strata, fit_on=ed_basis)["macro"])
            # GPU-free reproduction bundle: the EXACT arrays scored above for this scPRAM (model, split).
            dump_bundle(os.environ.get("IVCBENCH_PRED_DUMP"),
                        cluster=sp.spec.registry_task, model="scPRAM", split=sp.spec.name,
                        pred_cells=pred_cells, test_cells=test_X, cell_strata=test_strata,
                        control_mean=ctrl_mean, genes=cs.var_names, exclude_gene_idx=None,
                        fit_on=ed_basis, n_pca=50)
            au = float(aucell(prof[None, :], gs_ifn).mean()) - ctrl_auc
            seed_pearson.append(pe); seed_edist.append(ed); seed_aucell.append(au)
            timing.append(dict(lineage=lin, seed=seed, sec=round(dt, 1),
                               n_train=int(len(sp.train_idx)), n_test=int(len(sp.test_idx)),
                               n_ctrl=int(len(ctrl_idx))))
            print(f"  [{lin} seed{seed}] {dt:.0f}s pearsonD={pe:.4f} eDist={ed:.4f} aucellD={au:+.4f}",
                  flush=True)

        scpram_pe = float(np.mean(seed_pearson)); scpram_ed = float(np.mean(seed_edist))
        scpram_au = float(np.mean(seed_aucell))
        for metric, scpram_score, prim_name, prim_score, bof_name, bof_score, orient in [
            ("pearson_delta", scpram_pe, prim_pearson_name, b_pearson(prim_pearson_name),
             bof_pearson, b_pearson(bof_pearson), +1),
            ("e_distance", scpram_ed, prim_edist_name, b_edist(prim_edist_name),
             prim_edist_name, b_edist(prim_edist_name), -1),
            ("aucell_ifn_delta", scpram_au, "cell-mean", b_aucell("cell-mean"),
             "cell-mean", b_aucell("cell-mean"), None),
        ]:
            if metric == "aucell_ifn_delta":
                delta_prim = abs(prim_score - obs_ifn) - abs(scpram_score - obs_ifn)
                delta_bof = delta_prim
            elif orient > 0:
                delta_prim = scpram_score - prim_score
                delta_bof = scpram_score - bof_score
            else:
                delta_prim = prim_score - scpram_score
                delta_bof = bof_score - scpram_score
            rows.append(dict(
                lineage=lin, metric=metric, scpram_score=round(scpram_score, 4),
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
                n_strata=int(len(np.unique(test_strata))), epochs=int(args.epochs),
                leak_free=bool(audit["leak_free"]),
            ))
        pd.DataFrame(rows).to_csv(args.out, index=False)
        json.dump(timing, open(args.timing_out, "w"), indent=2)

    print(f"\nWROTE {args.out} ({len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    pe = df[df.metric == "pearson_delta"]
    if len(pe):
        print("\n=== Kang Pearson-delta scPRAM vs primary baseline ===")
        print(pe[["lineage", "scpram_score", "primary_baseline", "baseline_score",
                  "delta_vs_primary", "delta_vs_bestof4"]].to_string(index=False))
        print(f"\nmean delta_vs_primary = {pe.delta_vs_primary.mean():+.4f}; "
              f"%positive = {100*(pe.delta_vs_primary>0).mean():.0f}%")
        print(f"mean delta_vs_bestof4 = {pe.delta_vs_bestof4.mean():+.4f}; "
              f"%positive = {100*(pe.delta_vs_bestof4>0).mean():.0f}%")


if __name__ == "__main__":
    main()
