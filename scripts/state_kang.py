#!/usr/bin/env python
"""EXPERIMENT: STATE on Kang IFN-β PBMC leave-one-lineage-out (C1 LOCT, ①Cytokine×Hybrid).

Builds the Hybrid-family (STATE) FILL on the Figure-1 cytokine cell where Hybrid is "partial" and we
lack a STATE row. Mirrors scripts/scpram_kang.py EXACTLY (same repo split c1.coarse_loct, same
lineages≥50-treated rule, same universal-floor selection, same pearson_delta / e_distance / type-I-IFN
AUCell-Δ scoring + orientation), but swaps the OT model for arc-STATE's State-Transition model, which
runs in its own `ivc-state` conda env via model_runners/state_c1_runner.py.

CYTOKINE / CELL axis: the perturbation ("IFN-β") is SEEN in every training lineage; the held UNIT is a
*cell-type lineage*. STATE trains on all NON-held lineages' (control → IFN-β) cells conditioned on
lineage (cell_type) + donor (gem_group), and predicts the held lineage's stimulated response from its
OWN control cells. EMPTY-STRONG: the single (seen) stim label shares one constant perturbation feature,
so STATE gets no held-lineage side rep — it transfers the shared control→stim transition through context
only (honest hybrid lower-bound; from-scratch ST, embed_key=null). Directly comparable to the scGen /
CPA / CellOT / scPRAM Kang LOCT rows.

LEAK-SAFE (HARD RULE): the held lineage's IFN-β cells never enter the runner; only its control cells (the
inference input) are passed for prediction. The VAE/ST refit per fold (one subprocess per held lineage).
Same repo build_split + audit_split leak gate as scpram_kang.py / cellot_kang.py. The response panel and
E-distance PCA basis are training-fold-only (cs.X[sp.train_idx]) — never the held expression.

ANCHOR / sanity gate: STATE's per-lineage pearson_delta must be FINITE and land in the cell-axis hybrid
band the STATE-Soskic donor run established (mean ~0.17, range [0.01, 0.44], 100% in [-0.10, 0.50]). We
require every run lineage in [-0.10, 0.60] and finite; a leak-inflated score (>0.9) or a degenerate
non-finite / ~constant-0 score FAILS the gate and the result is NOT adoptable.

COMPUTE (2-GPU budget): the 8 Kang lineages run sequentially (each is minutes of GPU training). Pin
with --gpu / CUDA_VISIBLE_DEVICES. FULL run pins CUDA_VISIBLE_DEVICES=2,3 ONLY (GPUs 0,1 off-limits).
SMOKE: --smoke forces CPU (IVCBENCH_STATE_FORCE_CPU=1), tiny steps + tiny caps, one lineage, to prove
the path end-to-end in seconds without touching a GPU.
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

RUNNER = ROOT / "model_runners" / "state_c1_runner.py"
STATE_ENV = "ivc-state"

IFN_LIST = ["ISG15", "IFI6", "MX1", "MX2", "OAS1", "OAS2", "IFIT1", "IFIT3", "ISG20",
            "STAT1", "IRF7", "IFI44", "IFI44L", "RSAD2", "USP18"]

# anchor band (from the STATE-Soskic donor cell-axis run: 100% in [-0.10, 0.50], mean ~0.17)
ANCHOR_LO, ANCHOR_HI = -0.10, 0.60


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


def run_state_on_split(cs, sp, seed, held_lineage, cuda_device, steps, smoke=False):
    """Build the leak-safe per-fold payload (held lineage = unit, donors = strata), shell the STATE-C1
    runner in the ivc-state env, and return {donor: predicted IFN-β profile (n_genes,)} for the held
    lineage. The runner never sees the held lineage's IFN-β expression — only its control cells."""
    tr = sp.train_idx
    is_ctrl_tr = cs.obs.iloc[tr]["is_control"].to_numpy().astype(bool)
    celltype_train = cs.obs.iloc[tr]["cell_type_coarse"].astype(str).to_numpy()
    gem_train = cs.obs.iloc[tr]["donor_id"].astype(str).to_numpy()
    inf = sp.inference_input_idx
    celltype_inf = cs.obs.iloc[inf]["cell_type_coarse"].astype(str).to_numpy()
    gem_inf = cs.obs.iloc[inf]["donor_id"].astype(str).to_numpy()

    payload = dict(
        X_train=cs.X[tr].astype(np.float32),
        is_control_train=is_ctrl_tr,
        celltype_train=np.asarray([str(c) for c in celltype_train]),
        gem_train=np.asarray([str(g) for g in gem_train]),
        X_ctrl_inf=cs.X[inf].astype(np.float32),
        celltype_inf=np.asarray([str(c) for c in celltype_inf]),
        gem_inf=np.asarray([str(g) for g in gem_inf]),
        held_lineage=str(held_lineage),
        genes=np.asarray([str(g) for g in cs.var_names]),
    )
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "in.npz", Path(td) / "out.npz"
        np.savez(inp, **payload, allow_pickle=True)
        env = os.environ.copy()
        if cuda_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        if smoke:
            env["IVCBENCH_STATE_FORCE_CPU"] = "1"
            env["IVCBENCH_STATE_MAXCELLS"] = "1500"
        env["IVCBENCH_STATE_STEPS"] = str(steps)
        env["PYTHONHASHSEED"] = str(seed)
        proc = subprocess.run([env_python(STATE_ENV), str(RUNNER), str(inp), str(out)],
                              capture_output=True, text=True, timeout=7200, env=env)
        if proc.returncode != 0 or not out.exists():
            err = proc.stderr or ""
            key = [ln for ln in err.splitlines()
                   if any(k in ln for k in ("Error", "Exception", "Traceback", "assert", "RuntimeError"))]
            raise RuntimeError(f"STATE-C1 runner failed (rc={proc.returncode}):\n"
                               + ("… " + key[-1] + "\n" if key else "") + err[-3500:])
        r = np.load(out, allow_pickle=True)
        by_donor = {}
        for k, v in zip(r["pred_perts"], r["pred_means"]):
            donor = str(k).split("::", 1)[-1]
            by_donor[donor] = np.asarray(v, np.float32)
    return by_donor


def state_pred_cells(by_donor, test_strata, ctrl_mean):
    """Tile each held-donor predicted profile onto the test rows of that stratum (pearson_delta's
    per-stratum mean is invariant to tiling); fall back to control mean for an unseen donor stratum."""
    test_strata = np.asarray(test_strata)
    n_genes = len(ctrl_mean)
    pred = np.zeros((len(test_strata), n_genes), np.float32)
    for s in np.unique(test_strata):
        donor = str(s).split("=", 1)[-1]
        prof = by_donor.get(donor, ctrl_mean)
        pred[test_strata == s] = prof
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineages", nargs="*", default=None, help="subset of lineages (default all)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=400, help="STATE max_steps (IVCBENCH_STATE_STEPS)")
    ap.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES for the STATE subprocess")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU-only 1-lineage path-proof (forces FORCE_CPU, tiny steps/caps)")
    ap.add_argument("--anchor", action="store_true", default=True, help="enforce anchor gate (default on)")
    ap.add_argument("--no-anchor", dest="anchor", action="store_false")
    ap.add_argument("--out", default=str(ROOT / "outputs/additional_models/state_kang_raw.csv"))
    ap.add_argument("--timing-out", default=str(ROOT / "outputs/additional_models/state_kang_timing.json"))
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cs = kang_mod.load()
    gs_ifn = np.asarray(cs.gene_index(IFN_LIST), dtype=int)
    lins = args.lineages or lineages_for(cs)
    if args.smoke:
        lins = lins[:1]; args.seeds = [0]; args.steps = min(args.steps, 6)
    print(f"[kang-STATE] {cs.n_cells} cells x {cs.n_genes} genes; lineages={lins}; "
          f"seeds={args.seeds}; steps={args.steps}; gpu={args.gpu}; smoke={args.smoke}", flush=True)

    rows, timing = [], []
    any_fail = False
    for lin in lins:
        sp = build_split(cs, c1.coarse_loct(lin))
        audit = audit_split(cs, sp)
        assert audit["leak_free"], f"LEAK {lin}"
        test_X = cs.X[sp.test_idx]
        test_strata = sp.test_strata
        ctrl_idx = sp.inference_input_idx
        ctrl_X = cs.X[ctrl_idx]
        ctrl_mean = ctrl_X.mean(0)
        ed_basis = cs.X[sp.train_idx]               # training-only PCA basis (leak-safe)
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

        # ---- STATE per seed (subprocess) ----
        seed_pearson, seed_edist, seed_aucell = [], [], []
        for seed in args.seeds:
            t0 = time.time()
            by_donor = run_state_on_split(cs, sp, seed, lin, args.gpu, args.steps, smoke=args.smoke)
            dt = time.time() - t0
            pred_cells = state_pred_cells(by_donor, test_strata, ctrl_mean)
            pe = float(pearson_delta(pred_cells, test_X, ctrl_mean, test_strata)["macro"])
            ed = float(e_distance(pred_cells, test_X, test_strata, fit_on=ed_basis)["macro"])
            # IFN AUCell-Δ on the (tiled) predicted cloud vs control (same convention as the floor)
            au = float(aucell(pred_cells, gs_ifn).mean()) - ctrl_auc
            seed_pearson.append(pe); seed_edist.append(ed); seed_aucell.append(au)
            timing.append(dict(lineage=lin, seed=seed, sec=round(dt, 1),
                               n_train=int(len(sp.train_idx)), n_test=int(len(sp.test_idx)),
                               n_ctrl=int(len(ctrl_idx)), n_donors=int(len(by_donor))))
            print(f"  [{lin} seed{seed}] {dt:.0f}s pearsonD={pe:.4f} eDist={ed:.4f} aucellD={au:+.4f}",
                  flush=True)

        state_pe = float(np.mean(seed_pearson)); state_ed = float(np.mean(seed_edist))
        state_au = float(np.mean(seed_aucell))

        # anchor gate on Pearson-Δ (cell-axis hybrid band)
        finite = np.isfinite(state_pe)
        in_band = bool(finite and ANCHOR_LO <= state_pe <= ANCHOR_HI)
        leak_inflated = bool(finite and state_pe > 0.9)
        anchor_pass = bool(in_band and not leak_inflated)
        if args.anchor and not args.smoke and not anchor_pass:
            any_fail = True

        for metric, state_score, prim_name, prim_score, bof_name, bof_score, orient in [
            ("pearson_delta", state_pe, prim_pearson_name, b_pearson(prim_pearson_name),
             bof_pearson, b_pearson(bof_pearson), +1),
            ("e_distance", state_ed, prim_edist_name, b_edist(prim_edist_name),
             prim_edist_name, b_edist(prim_edist_name), -1),
            ("aucell_ifn_delta", state_au, "cell-mean", b_aucell("cell-mean"),
             "cell-mean", b_aucell("cell-mean"), None),
        ]:
            if metric == "aucell_ifn_delta":
                delta_prim = abs(prim_score - obs_ifn) - abs(state_score - obs_ifn)
                delta_bof = delta_prim
            elif orient > 0:
                delta_prim = state_score - prim_score
                delta_bof = state_score - bof_score
            else:
                delta_prim = prim_score - state_score
                delta_bof = bof_score - state_score
            rows.append(dict(
                lineage=lin, metric=metric, state_score=round(state_score, 4),
                primary_baseline=prim_name, baseline_score=round(prim_score, 4),
                delta_vs_primary=round(delta_prim, 4),
                bestof4_baseline=bof_name, bestof4_score=round(bof_score, 4),
                delta_vs_bestof4=round(delta_bof, 4),
                obs_ifn_delta=(round(obs_ifn, 4) if metric == "aucell_ifn_delta" else ""),
                anchor_pass=(anchor_pass if metric == "pearson_delta" else ""),
                seed_scores=json.dumps([round(x, 4) for x in
                                        (seed_pearson if metric == "pearson_delta" else
                                         seed_edist if metric == "e_distance" else seed_aucell)]),
                seeds=",".join(map(str, args.seeds)),
                n_test=int(len(sp.test_idx)), n_ctrl=int(len(ctrl_idx)),
                n_strata=int(len(np.unique(test_strata))), steps=int(args.steps),
                leak_free=bool(audit["leak_free"]), smoke=bool(args.smoke),
            ))
        if not args.smoke:
            pd.DataFrame(rows).to_csv(args.out, index=False)
            json.dump(timing, open(args.timing_out, "w"), indent=2)

    if args.smoke:
        smoke_path = Path(args.out).with_name("state_kang_smoke.csv")
        pd.DataFrame(rows).to_csv(smoke_path, index=False)
        pe = [r for r in rows if r["metric"] == "pearson_delta"]
        ok = bool(pe and np.isfinite(pe[0]["state_score"]) and rows[0]["leak_free"])
        print(f"\n[SMOKE] wrote {smoke_path}; path_ok={ok} "
              f"(pearsonD={pe[0]['state_score'] if pe else 'NA'})", flush=True)
        sys.exit(0 if ok else 1)

    print(f"\nWROTE {args.out} ({len(rows)} rows)", flush=True)
    df = pd.DataFrame(rows)
    pe = df[df.metric == "pearson_delta"]
    if len(pe):
        sc = pe.state_score.astype(float)
        print("\n=== Kang Pearson-Δ STATE vs primary baseline ===")
        print(pe[["lineage", "state_score", "primary_baseline", "baseline_score",
                  "delta_vs_primary", "delta_vs_bestof4", "anchor_pass"]].to_string(index=False))
        print(f"\nmean state_score = {sc.mean():+.4f} (min {sc.min():+.4f}, max {sc.max():+.4f}); "
              f"mean delta_vs_primary = {pe.delta_vs_primary.mean():+.4f}; "
              f"%positive = {100*(pe.delta_vs_primary>0).mean():.0f}%; "
              f"finite={bool(np.isfinite(sc).all())}; "
              f"%anchor_pass={100*pe.anchor_pass.mean():.0f}%")
    if args.anchor and any_fail:
        print("[ANCHOR] FAIL: ≥1 lineage outside the cell-axis hybrid band [-0.10, 0.60] or leak-inflated.",
              flush=True)
        sys.exit(2)
    print("[ANCHOR] PASS (all run lineages finite and in band).", flush=True)


if __name__ == "__main__":
    main()
