#!/usr/bin/env python
"""CellOT -> Frangieh (C4 complex-context, Axis-2 modality generalization) — ④xOT (Fig-1 STRONG).

CellOT (Bunne et al. 2023) is the CONDITIONED Optimal-Transport headline model and is `applicable`
(headline-eligible) on C4_Axis2 in the registry. It mirrors scripts/cinemaot_frangieh.py EXACTLY (same
C4 modality_lo_ko split, same Frangieh RNA loader, same anchor/sanity gate, same per-fold leak-safe
scoring), but swaps the CINEMA-OT global-OT floor for the REAL CellOT map: an scgen autoencoder (50-dim
latent) trained on all NON-held train cells, then CellOT f/g ICNN potentials learned IN that latent
space with source = non-targeting control, target = the pooled training KO cells (the KO-vs-control
contrast — Frangieh is KO vs control, not paired stimulation, so the "treatment" the OT map learns is
the global control->KO response shift). At inference the held-KO group's control cells (= the shared
non-targeting control, split.inference_input_idx) are encoded, pushed through g.transport, and decoded
to gene space (the official ae-embedding / data_space prediction path). The same control cloud is pushed
for every held KO, so the prediction is perturbation-agnostic in the KO axis — exactly the CINEMA-OT
floor's status, but with a genuine learned OT map rather than a pooled treatment-effect mean. This is
the OT-family entry for the ④ (complex-context) x OT = STRONG Figure-1 cell.

Reuses scripts/cellot_runner.py (the SAME AE+OT helpers that produced the CellOT-Kang / CellOT-Soskic
rows) via model_runners/cellot_frangieh_runner.py. The Frangieh scPerturb h5ad needs a NEW anndata
(numpy 2.x) but the cellot package lives in the old `cellot` env (numpy 1.19), so — like every heavy
baseline — this driver LOADS + SCORES in the GPU-free `.venv` and shells the MODEL step into the `cellot`
env (cross-version-safe unicode+f32 payload). stratum_align / edist_clouds are imported in-process from
cellot_runner (pure numpy/sklearn, env-agnostic). GPU budget: devices 2,3 ONLY (0,1 off-limits).

RNA ONLY (matches the scGen / CINEMA-OT Frangieh RNA anchor). Protein (CITE) is NOT conditioned here and
is intentionally NOT run — disclosed RNA-readout OT result for the modality axis.

LEAK-SAFE (hard rule): build_split -> audit_split (hard LeakError gate) per fold. The CellOT AE + f/g
train on split.train_idx only; the held KO's cells never enter training/model-selection. Prediction uses
only the non-held control cells (split.inference_input_idx). The Pearson-Δ excludes the held KO genes
(downstream-only). The e_distance PCA basis and the aucell control reference are fit on the TRAIN fold
only. Per-fold refit: each (fraction) fold rebuilds the split and retrains AE + OT from scratch.

ANCHOR / sanity gate (--anchor, default on): per fold, CellOT is adoptable only if it
  (a) BEATS the best NON-trivial learned-conditioning-free linear floor (linear-PCA pearson_delta) on
      this exact fold, AND
  (b) lands within an order of magnitude of the scGen Frangieh-RNA pearson_delta anchor
      (0.5376 @ 25, 0.5533 @ 50): we require 0.10 <= pearson_delta <= 0.95.
A fold that fails the gate is marked anchor_pass=False; the driver exits non-zero in --anchor mode if any
non-smoke fold fails, so a bad result can never be silently adopted. (cell-mean/donor-shift ~0.67 on this
split is a perturbation-agnostic training-mean-shift oracle the conditioned scGen reference 0.538 does
not itself beat — reported for context, NOT the gate.)

--chunk I N shards the fold-units (lo_ko_25 / lo_ko_50) one per GPU worker slot (units[i::n]), modelled
on cinemaot_frangieh.py. SMOKE: --smoke runs one tiny fold (lo_ko_25) with a small cell cap + few AE/OT
iters on CPU purely to prove the path runs end-to-end and writes a sane per-unit row; NOT the full job.

Envs: `cellot` (cellot repo + torch). FULL-run GPU budget = devices 2,3 ONLY.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ivcbench.data.loaders.frangieh import load
from ivcbench.clusters import c4
from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split
from ivcbench.metrics.response import pearson_delta
from ivcbench.metrics.distribution import e_distance
from ivcbench.metrics.program import aucell
from ivcbench.baselines.simple import CellMean, DonorShift, CtrlPred, LinearPCA
from ivcbench.baselines.heavy import env_python

RUNNER = ROOT / "model_runners" / "cellot_frangieh_runner.py"
CELLOT_ENV = "cellot"


# stratum_align / edist_clouds are pure numpy/sklearn (env-agnostic). The .venv has no torch, so we do
# NOT import cellot_runner here (it imports torch); these are byte-identical copies of the two helpers in
# scripts/cellot_runner.py used by the in-process CellOT-Kang/Soskic rows. The AE+OT MODEL step runs in
# the `cellot` env via model_runners/cellot_frangieh_runner.py (which imports the rest of cellot_runner).
def stratum_align(pred_cells, pred_strata, test_strata):
    pred_strata, test_strata = np.asarray(pred_strata), np.asarray(test_strata)
    aligned = np.zeros((len(test_strata), pred_cells.shape[1]), np.float32)
    for s in np.unique(test_strata):
        mt, mp = test_strata == s, pred_strata == s
        block = pred_cells[mp] if mp.sum() else pred_cells
        reps = int(np.ceil(mt.sum() / len(block)))
        aligned[mt] = np.tile(block, (reps, 1))[: mt.sum()]
    return aligned


def edist_clouds(pred_cells, pred_strata, test_cells, test_strata, fit_on, n_pca=50):
    """E-distance per test stratum: observed KO cloud vs the CellOT-pushed control cloud. On Frangieh the
    prediction is perturbation-agnostic — all pushed control cells carry the 'control' stratum, never a
    KO label — so when no pred cell matches a KO stratum we use the FULL pushed control cloud as that
    stratum's prediction (the same perturbation-agnostic fallback stratum_align uses for Pearson-Δ; the
    floors compare the test KO cloud against their own per-KO mean-shifted cloud, so this is apples-to-
    apples at the distributional level). PCA basis = train fold (leak-safe)."""
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import cdist
    pred_strata, test_strata = np.asarray(pred_strata), np.asarray(test_strata)
    k = int(min(n_pca, fit_on.shape[0] - 1, fit_on.shape[1]))
    pca = PCA(n_components=max(2, k), random_state=0).fit(fit_on)
    per = []
    for s in np.unique(test_strata):
        mt, mp = test_strata == s, pred_strata == s
        block = pred_cells[mp] if mp.sum() >= 2 else pred_cells     # perturbation-agnostic fallback
        if mt.sum() < 2 or block.shape[0] < 2:
            continue
        P, T = pca.transform(block), pca.transform(test_cells[mt])
        d = 2 * cdist(P, T).mean() - cdist(P, P).mean() - cdist(T, T).mean()
        per.append(float(d))
    return float(np.mean(per)) if per else float("nan")


def run_cellot_frangieh(cs, sp, seed, ae_iters, cellot_iters, cap, cuda_device, timeout_s=7200):
    """Build the leak-safe per-fold payload (train expression + control mask + the held-KO group's
    control cells), shell the CellOT runner in the `cellot` env, return (pred_genes, best_mmd). The
    runner never sees the held-out test expression — only the non-held control cells to push."""
    tr = sp.train_idx
    is_ctrl_tr = cs.obs.iloc[tr]["is_control"].to_numpy().astype(bool)
    payload = dict(
        X_train=cs.X[tr].astype(np.float32),
        is_control_train=is_ctrl_tr,
        X_ctrl_inf=cs.X[sp.inference_input_idx].astype(np.float32),
        ae_iters=np.int64(ae_iters), cellot_iters=np.int64(cellot_iters),
        seed=np.int64(seed), cap=np.int64(cap),
    )
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "in.npz", Path(td) / "out.npz"
        np.savez(inp, **payload, allow_pickle=True)
        env = os.environ.copy()
        if cuda_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        proc = subprocess.run([env_python(CELLOT_ENV), str(RUNNER), str(inp), str(out)],
                              capture_output=True, text=True, timeout=timeout_s, env=env)
        if proc.returncode != 0 or not out.exists():
            err = proc.stderr or ""
            key = [ln for ln in err.splitlines()
                   if any(k in ln for k in ("Error", "Exception", "Traceback", "assert", "RuntimeError"))]
            raise RuntimeError(f"CellOT-Frangieh runner failed (rc={proc.returncode}):\n"
                               + ("… " + key[-1] + "\n" if key else "") + err[-3500:])
        r = np.load(out, allow_pickle=True)
        return np.asarray(r["pred_genes"], np.float32), float(r["best_mmd"])

# scGen Frangieh-RNA anchor (results/C4/conditioned_rows.json) — the conditioned reference the OT result
# must land near (order of magnitude).
SCGEN_ANCHOR = {"25": 0.5376423142039247, "50": 0.5533400144983804}
# linear-PCA floor per split (results/C4/results.csv) — the genuine learned-conditioning-free non-trivial
# linear floor CellOT must beat. Recomputed per fold below; these are the cached reference values.
LINEAR_PCA_FLOOR = {"25": 0.2914931868172942, "50": 0.27515802689782914}
ANCHOR_LO, ANCHOR_HI = 0.10, 0.95

# Frangieh IFNγ immune-evasion / type-II-IFN program for the AUCell-Δ axis (Frangieh's headline biology:
# IFNγ-driven antigen-presentation + immune-checkpoint program in the melanoma KO screen). Scored
# identically for CellOT and the floors; intersected with the HVG panel at runtime.
FRANGIEH_IFN_PROGRAM = ["STAT1", "IRF1", "GBP1", "GBP2", "GBP5", "CXCL9", "CXCL10", "CXCL11",
                        "TAP1", "TAP2", "PSMB8", "PSMB9", "HLA-A", "HLA-B", "HLA-C", "B2M",
                        "IDO1", "CD274", "NLRC5", "IFI30", "UBE2L6", "WARS"]


def _floors(cs, sp, excl_idx, ed_basis, gs_ifn, ctrl_X):
    """Recompute the simple-baseline pearson_delta / e_distance / aucell-Δ floors for THIS fold
    (leak-safe, in-process) so the anchor comparison is against this run's split."""
    test_X = cs.X[sp.test_idx]
    ts = sp.test_strata
    ctrl_auc = float(aucell(ctrl_X, gs_ifn).mean()) if len(gs_ifn) else float("nan")
    out = {}
    for B in (CtrlPred, CellMean, DonorShift, LinearPCA):
        b = B(); b.fit(cs, sp, side_info=cs.side_info)
        p = b.predict(cs, sp, side_info=cs.side_info)
        pe = float(pearson_delta(p.pred_cells, test_X, p.control_mean, ts, excl_idx)["macro"])
        ed = float(e_distance(p.pred_cells, test_X, ts, fit_on=ed_basis)["macro"])
        au = (float(aucell(p.pred_cells, gs_ifn).mean()) - ctrl_auc) if len(gs_ifn) else float("nan")
        out[b.name] = dict(pearson=pe, edist=ed, aucell=au)
    return out, ctrl_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny 1-fold path-proof (small cap/iters, CPU); does NOT run the full job")
    ap.add_argument("--chunk", type=int, nargs=2, default=None, metavar=("I", "N"),
                    help="shard fold-units units[i::n] across N GPU worker slots")
    ap.add_argument("--fractions", nargs="*", default=["25", "50"],
                    help="held-KO fractions to run as folds (default 25 50)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0],
                    help="seeds (technical repeats; mean-collapsed within a fold)")
    ap.add_argument("--ae-iters", type=int, default=12000)
    ap.add_argument("--cellot-iters", type=int, default=8000)
    ap.add_argument("--subsample-per-group", type=int, default=60,
                    help="cells/KO loaded by the Frangieh loader")
    ap.add_argument("--cap", type=int, default=4000, help="per-arm OT cell cap (control / pooled-KO)")
    ap.add_argument("--gpu", type=str, default=None,
                    help="CUDA_VISIBLE_DEVICES for the cellot subprocess (FULL runs: 2 or 3 ONLY)")
    ap.add_argument("--anchor", action="store_true", default=True,
                    help="enforce the anchor/sanity gate (default on)")
    ap.add_argument("--no-anchor", dest="anchor", action="store_false")
    ap.add_argument("--out", default=str(ROOT / "outputs/additional_models/cellot_frangieh_raw.csv"))
    ap.add_argument("--timing-out",
                    default=str(ROOT / "outputs/additional_models/cellot_frangieh_timing.json"))
    args = ap.parse_args()

    if args.smoke:
        args.fractions = ["25"]
        args.seeds = [0]
        args.ae_iters = 60
        args.cellot_iters = 40
        args.subsample_per_group = 12
        args.cap = 300

    if args.smoke and args.gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU smoke
    print(f"[cellot env] {env_python(CELLOT_ENV)}  gpu={args.gpu}", flush=True)
    print(f"[load] Frangieh RNA (subsample_per_group={args.subsample_per_group})", flush=True)
    cs = load(modality="rna", subsample_per_group=args.subsample_per_group)
    genes_perturbed = cs.uns["genes_perturbed"]
    ds_name = cs.uns.get("dataset", "frangieh_rna")
    gs_ifn = np.asarray(cs.gene_index([g for g in FRANGIEH_IFN_PROGRAM if g in set(cs.var_names)]),
                        dtype=int)
    print(f"[load] {cs.n_cells} cells x {cs.n_genes} genes; {len(genes_perturbed)} KOs; "
          f"ctrl_frac={float(cs.obs['is_control'].mean()):.3f}; IFNg-program genes={len(gs_ifn)}",
          flush=True)

    units = list(args.fractions)
    if args.chunk:
        i, n = args.chunk
        units = units[i::n]
        print(f"[chunk {i}/{n}] units -> {units}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, timing = [], []
    if not args.smoke and out_path.exists():
        try:
            rows = pd.read_csv(out_path).to_dict("records")
        except Exception:
            rows = []
    if not args.smoke and Path(args.timing_out).exists():
        try:
            timing = json.load(open(args.timing_out))
        except Exception:
            timing = []

    any_fail = False
    for frac_label in units:
        frac = 0.25 if frac_label == "25" else 0.50
        held = c4.held_ko_fraction(genes_perturbed, frac, seed=0)
        spec = c4.modality_lo_ko(held, frac_label)
        excl = list(spec.held_values)                      # downstream-only: held KO genes excluded
        excl_idx = np.asarray(cs.gene_index(excl), dtype=int) if excl else None

        sp = build_split(cs, spec)
        audit = audit_split(cs, sp)                        # hard LeakError gate
        assert audit["leak_free"], f"LEAK {frac_label}"
        test_X = cs.X[sp.test_idx]
        test_strata = sp.test_strata
        ctrl_idx = sp.inference_input_idx                  # shared non-targeting control
        ctrl_X = cs.X[ctrl_idx]
        ctrl_mean = ctrl_X.mean(0)
        ctrl_strata = np.array([spec.stratum_key(cs.obs.iloc[i]) for i in ctrl_idx], dtype=object)
        ed_basis = cs.X[sp.train_idx]
        if len(ed_basis) > 5000:
            ed_basis = ed_basis[np.random.default_rng(0).choice(len(ed_basis), 5000, replace=False)]

        floors, ctrl_auc = _floors(cs, sp, excl_idx, ed_basis, gs_ifn, ctrl_X)
        best_simple = floors["linear-PCA"]["pearson"]      # gate floor (genuine non-trivial linear floor)
        obs_ifn = ((float(aucell(test_X, gs_ifn).mean()) - ctrl_auc) if len(gs_ifn) else float("nan"))

        # ---- CellOT per seed (cellot-env subprocess; technical repeats, mean-collapsed) ----
        gpu = args.gpu if not args.smoke else (args.gpu or "")
        seed_pe, seed_ed, seed_au, seed_mmd = [], [], [], []
        for seed in args.seeds:
            t0 = time.time()
            try:
                pred_genes, best_mmd = run_cellot_frangieh(
                    cs, sp, seed, args.ae_iters, args.cellot_iters, args.cap, gpu)
            except Exception as e:  # noqa: BLE001
                print(f"  [{frac_label} seed{seed}] FAILED: {type(e).__name__}: {e}", flush=True)
                pred_genes, best_mmd = None, float("nan")
            dt = time.time() - t0
            if pred_genes is None:
                seed_pe.append(float("nan")); seed_ed.append(float("nan"))
                seed_au.append(float("nan")); seed_mmd.append(float("nan"))
                continue
            aligned = stratum_align(pred_genes, ctrl_strata, test_strata)
            pe = float(pearson_delta(aligned, test_X, ctrl_mean, test_strata, excl_idx)["macro"])
            ed = edist_clouds(pred_genes, ctrl_strata, test_X, test_strata, ed_basis)
            au = ((float(aucell(pred_genes, gs_ifn).mean()) - ctrl_auc) if len(gs_ifn) else float("nan"))
            if seed == args.seeds[0]:  # deposit the seed-0 prediction bundle (C4 CellOT fill)
                from ivcbench.eval.bundle import dump_bundle
                dump_bundle(os.environ.get("IVCBENCH_PRED_DUMP"), cluster="C4", model="CellOT", split=spec.name,
                            pred_cells=aligned, test_cells=test_X, cell_strata=test_strata,
                            control_mean=ctrl_mean, genes=cs.var_names, exclude_gene_idx=excl_idx,
                            fit_on=ed_basis, n_pca=50)
            seed_pe.append(pe); seed_ed.append(ed); seed_au.append(au); seed_mmd.append(best_mmd)
            timing.append(dict(frac=frac_label, seed=seed, sec=round(dt, 1),
                               best_mmd=round(float(best_mmd), 5) if best_mmd == best_mmd else None,
                               n_train=int(len(sp.train_idx)), n_test=int(len(sp.test_idx)),
                               n_ctrl=int(len(ctrl_idx))))
            print(f"  [{frac_label} seed{seed}] {dt:.0f}s pearsonD={pe:.4f} eDist={ed:.4f} "
                  f"aucellD={au:+.4f} mmd={best_mmd:.4f}", flush=True)

        def _mean(v):
            v = [x for x in v if x == x]
            return float(np.mean(v)) if v else float("nan")
        ce_pe, ce_ed, ce_au, ce_mmd = _mean(seed_pe), _mean(seed_ed), _mean(seed_au), _mean(seed_mmd)
        ran = ce_pe == ce_pe

        # anchor / sanity gate
        scgen = SCGEN_ANCHOR.get(frac_label, float("nan"))
        beats_floor = bool(ran and ce_pe > best_simple)
        in_window = bool(ran and ANCHOR_LO <= ce_pe <= ANCHOR_HI)
        anchor_pass = bool(beats_floor and in_window)
        if not args.smoke and args.anchor and ran and not anchor_pass:
            any_fail = True
        if not args.smoke and args.anchor and not ran:
            any_fail = True

        row = dict(
            model="CellOT", cluster="C4", dataset=ds_name, modality="RNA",
            split=spec.name, frac_held=frac_label, family="ot",
            action="run_headline", ran=ran, leak_free=bool(audit["leak_free"]),
            n_train=int(len(sp.train_idx)), n_test=int(len(sp.test_idx)),
            n_test_strata=int(len(np.unique(test_strata))), n_ctrl=int(len(ctrl_idx)),
            pearson_delta=ce_pe, e_distance=ce_ed, aucell_ifn_delta=ce_au, obs_ifn_delta=obs_ifn,
            best_mmd=ce_mmd,
            linear_pca_pearson=round(floors["linear-PCA"]["pearson"], 4),
            cell_mean_pearson=round(floors["cell-mean"]["pearson"], 4),   # oracle-ish, context only
            cell_mean_edist=round(floors["cell-mean"]["edist"], 4),
            cell_mean_aucell=round(floors["cell-mean"]["aucell"], 4) if floors["cell-mean"]["aucell"] == floors["cell-mean"]["aucell"] else None,
            delta_pearson_vs_linpca=round(ce_pe - best_simple, 4) if ran else None,
            delta_edist_vs_cellmean=round(floors["cell-mean"]["edist"] - ce_ed, 4) if ran else None,
            beats_linpca_floor=beats_floor,
            scgen_anchor=round(scgen, 4) if scgen == scgen else None, in_anchor_window=in_window,
            anchor_pass=anchor_pass,
            ae_iters=args.ae_iters, cellot_iters=args.cellot_iters, cap=args.cap,
            subsample_per_group=args.subsample_per_group, seeds=",".join(map(str, args.seeds)),
            seed_pearson=json.dumps([round(x, 4) if x == x else None for x in seed_pe]),
            smoke=bool(args.smoke),
        )
        # de-dup any prior row for this split before appending (resume safety)
        rows = [r for r in rows if not (str(r.get("split")) == spec.name and not r.get("smoke", False))]
        rows.append(row)
        print(json.dumps({k: row[k] for k in (
            "split", "frac_held", "ran", "leak_free", "n_train", "n_test", "pearson_delta",
            "e_distance", "aucell_ifn_delta", "linear_pca_pearson", "beats_linpca_floor",
            "scgen_anchor", "in_anchor_window", "anchor_pass")}, default=str), flush=True)

        if not args.smoke:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            json.dump(timing, open(args.timing_out, "w"), indent=2)

    if args.smoke:
        smoke_path = out_path.with_name("cellot_frangieh_smoke.csv")
        pd.DataFrame(rows).to_csv(smoke_path, index=False)
        print(f"\n[SMOKE] wrote {smoke_path}", flush=True)
        r0 = rows[-1] if rows else {}
        ok = bool(r0.get("ran") and r0.get("leak_free")
                  and r0.get("pearson_delta") is not None
                  and r0.get("pearson_delta") == r0.get("pearson_delta"))
        print(f"[SMOKE] path_ok={ok} pearson_delta={r0.get('pearson_delta')}", flush=True)
        sys.exit(0 if ok else 1)

    print(f"\nWROTE {out_path} ({len(rows)} rows)", flush=True)
    if args.anchor and any_fail:
        print("[ANCHOR] FAIL: at least one fold did not pass the sanity gate "
              "(must beat the linear-PCA floor AND land in the scGen order-of-magnitude window).",
              flush=True)
        sys.exit(2)
    print("[ANCHOR] PASS (all run folds beat the linear-PCA floor and land near the scGen anchor).",
          flush=True)


if __name__ == "__main__":
    main()
