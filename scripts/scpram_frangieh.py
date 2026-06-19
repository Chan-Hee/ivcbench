#!/usr/bin/env python
"""scPRAM -> Frangieh (C4 complex-context, Axis-2) — STRUCTURALLY NOT-DEFINED on KO; run as an OT FLOOR.

HONEST APPLICABILITY (the task's "if a model is structurally inapplicable to KO say so honestly"):
scPRAM (Jiang et al. 2024) is registered `not_defined` on C4_Axis2 (src/ivcbench/baselines/registry.py).
Its mechanism is a PAIRED-stimulation cell/donor transfer: a HELD cell-type (Kang C1_LOCT) or HELD donor
(Soskic C2_LODO) is predicted from that unit's OWN control cells, with the ctrl->stim delta learned from
OTHER reference cell-types (cell_type != held). Frangieh is a CRISPR-KO screen, NOT paired stimulation:
  * there is ONE melanoma cell type and ONE shared non-targeting control population (ctrl_frac~0.004),
  * the held axis is a KO GENE, which has NO own-control population distinct from the global control,
  * there are no multiple reference cell-types to learn a transferable, unit-specific ctrl->stim delta.
So scPRAM has no conditioned signal to exploit on the unseen-KO modality axis. We therefore DO NOT claim
it as a headline OT predictor here. We run it ONLY as a perturbation-agnostic OT FLOOR — exactly the
status CINEMA-OT carries on the C3 unseen-gene task — to SHOW where scPRAM's VAE+OT machinery lands when
the conditioning it needs is absent: pooled training KO cells become the single "stim" group, the shared
non-targeting control becomes the held unit's control, one shared REF token carries the reference cells,
and the predicted profile is the global VAE+OT shift applied to the control. This is the KO-vs-control
contrast the task asked for, run honestly as a floor (action="run_floor", headline_eligible=False).

Mirrors scripts/cinemaot_frangieh.py (same C4 modality_lo_ko split, same Frangieh RNA loader, same
leak-safe scoring against the floors) and reuses model_runners/scpram_runner.py UNCHANGED (the same
runner that produced the scPRAM-Kang / scPRAM-Soskic rows). The runner trains the VAE only on the train
fold; the held-KO test expression never enters it (leak-safe by construction; build_split+audit_split
enforce it). The prediction is the same single global profile for every held KO (perturbation-agnostic),
so it is reported as a floor reference and EXCLUDED from headline ranking.

We do NOT apply an anchor "must beat floor / land in scGen window" PASS/FAIL gate (a floor is not
required to beat the floors). The sanity check is only that the path runs leak-free and emits a finite,
non-degenerate profile. RNA ONLY (protein not run). Env: ivc-scpram. FULL-run GPU budget: devices 2,3
ONLY (0,1 off-limits).
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

from ivcbench.data.loaders.frangieh import load
from ivcbench.clusters import c4
from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split
from ivcbench.metrics.response import pearson_delta
from ivcbench.metrics.distribution import e_distance
from ivcbench.metrics.program import aucell
from ivcbench.baselines.simple import CellMean, DonorShift, CtrlPred, LinearPCA
from ivcbench.baselines.heavy import env_python

RUNNER = ROOT / "model_runners" / "scpram_runner.py"
SCPRAM_ENV = "ivc-scpram"

SCGEN_ANCHOR = {"25": 0.5376423142039247, "50": 0.5533400144983804}
FRANGIEH_IFN_PROGRAM = ["STAT1", "IRF1", "GBP1", "GBP2", "GBP5", "CXCL9", "CXCL10", "CXCL11",
                        "TAP1", "TAP2", "PSMB8", "PSMB9", "HLA-A", "HLA-B", "HLA-C", "B2M",
                        "IDO1", "CD274", "NLRC5", "IFI30", "UBE2L6", "WARS"]


def run_scpram_floor(cs, sp, seed, epochs, ratio, cuda_device, cap, timeout_s=7200):
    """Build the leak-safe per-fold payload (train expression + control mask + held control cells), shell
    the UNCHANGED scpram_runner.py in the ivc-scpram env, return the predicted global profile (n_genes,).
    The runner treats the pooled training KO cells as the single 'stim' group, the shared non-targeting
    control as the held unit's control. The held-out KO expression never enters the runner."""
    tr = sp.train_idx
    is_ctrl_tr = cs.obs.iloc[tr]["is_control"].to_numpy().astype(bool)
    # cap the training cells fed to scPRAM (the VAE is the cost driver; control is scarce so keep ALL
    # control + a capped sample of pooled KO cells, preserving both arms).
    rng = np.random.default_rng(seed)
    ctrl_pos = np.where(is_ctrl_tr)[0]
    treat_pos = np.where(~is_ctrl_tr)[0]
    if len(treat_pos) > cap:
        treat_pos = rng.choice(treat_pos, cap, replace=False)
    keep = np.sort(np.concatenate([ctrl_pos, treat_pos]))
    X_tr = cs.X[tr][keep].astype(np.float32)
    is_ctrl_keep = is_ctrl_tr[keep]
    pert_keep = np.where(is_ctrl_keep, "control", "KO")        # pooled KO = single 'stim' label
    payload = dict(
        X_train=X_tr,
        is_control_train=is_ctrl_keep,
        pert_train=np.asarray([str(p) for p in pert_keep]),
        X_ctrl_inf=cs.X[sp.inference_input_idx].astype(np.float32),
        test_perts=np.asarray(["KO"]),                        # single SEEN 'stim' label for the runner
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
                              capture_output=True, text=True, timeout=timeout_s, env=env)
        if proc.returncode != 0 or not out.exists():
            err = proc.stderr or ""
            key = [ln for ln in err.splitlines()
                   if any(k in ln for k in ("Error", "Exception", "Traceback", "assert", "RuntimeError"))]
            raise RuntimeError(f"scPRAM-Frangieh runner failed (rc={proc.returncode}):\n"
                               + ("… " + key[-1] + "\n" if key else "") + err[-3500:])
        r = np.load(out, allow_pickle=True)
        return np.asarray(r["pred_means"][0], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny 1-fold path-proof (few epochs); does NOT run the full job")
    ap.add_argument("--chunk", type=int, nargs=2, default=None, metavar=("I", "N"))
    ap.add_argument("--fractions", nargs="*", default=["25", "50"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--ratio", type=float, default=0.005)
    ap.add_argument("--cap", type=int, default=4000, help="cap pooled-KO training cells fed to the VAE")
    ap.add_argument("--subsample-per-group", type=int, default=60)
    ap.add_argument("--gpu", type=str, default=None,
                    help="CUDA_VISIBLE_DEVICES for the scPRAM subprocess (FULL runs: 2 or 3 ONLY)")
    ap.add_argument("--out", default=str(ROOT / "outputs/additional_models/scpram_frangieh_raw.csv"))
    ap.add_argument("--timing-out",
                    default=str(ROOT / "outputs/additional_models/scpram_frangieh_timing.json"))
    args = ap.parse_args()

    if args.smoke:
        args.fractions = ["25"]
        args.epochs = 3
        args.cap = 600
        args.subsample_per_group = 12
        if args.gpu is None:
            args.gpu = ""                                # CPU smoke

    print(f"[scpram env] {env_python(SCPRAM_ENV)}  gpu={args.gpu}  epochs={args.epochs}", flush=True)
    print("[applicability] scPRAM is NOT-DEFINED on C4_Axis2 (KO, not paired stimulation) — running as "
          "a perturbation-agnostic OT FLOOR (headline_eligible=False).", flush=True)
    print(f"[load] Frangieh RNA (subsample_per_group={args.subsample_per_group})", flush=True)
    cs = load(modality="rna", subsample_per_group=args.subsample_per_group)
    genes_perturbed = cs.uns["genes_perturbed"]
    ds_name = cs.uns.get("dataset", "frangieh_rna")
    gs_ifn = np.asarray(cs.gene_index([g for g in FRANGIEH_IFN_PROGRAM if g in set(cs.var_names)]),
                        dtype=int)
    print(f"[load] {cs.n_cells} cells x {cs.n_genes} genes; {len(genes_perturbed)} KOs; "
          f"IFNg-program genes={len(gs_ifn)}", flush=True)

    units = list(args.fractions)
    if args.chunk:
        i, n = args.chunk; units = units[i::n]
        print(f"[chunk {i}/{n}] units -> {units}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, timing = [], []
    if not args.smoke and out_path.exists():
        try:
            rows = pd.read_csv(out_path).to_dict("records")
        except Exception:
            rows = []

    for frac_label in units:
        frac = 0.25 if frac_label == "25" else 0.50
        held = c4.held_ko_fraction(genes_perturbed, frac, seed=0)
        spec = c4.modality_lo_ko(held, frac_label)
        excl = list(spec.held_values)
        excl_idx = np.asarray(cs.gene_index(excl), dtype=int) if excl else None

        sp = build_split(cs, spec)
        audit = audit_split(cs, sp)
        assert audit["leak_free"], f"LEAK {frac_label}"
        test_X = cs.X[sp.test_idx]; ts = sp.test_strata
        ctrl_X = cs.X[sp.inference_input_idx]; ctrl_mean = ctrl_X.mean(0)
        ed_basis = cs.X[sp.train_idx]
        if len(ed_basis) > 5000:
            ed_basis = ed_basis[np.random.default_rng(0).choice(len(ed_basis), 5000, replace=False)]
        ctrl_auc = float(aucell(ctrl_X, gs_ifn).mean()) if len(gs_ifn) else float("nan")

        # floors (context)
        floors = {}
        for B in (CtrlPred, CellMean, DonorShift, LinearPCA):
            b = B(); b.fit(cs, sp, side_info=cs.side_info)
            p = b.predict(cs, sp, side_info=cs.side_info)
            floors[b.name] = dict(
                pearson=float(pearson_delta(p.pred_cells, test_X, p.control_mean, ts, excl_idx)["macro"]),
                edist=float(e_distance(p.pred_cells, test_X, ts, fit_on=ed_basis)["macro"]))
        lin_pca = floors["linear-PCA"]["pearson"]
        obs_ifn = ((float(aucell(test_X, gs_ifn).mean()) - ctrl_auc) if len(gs_ifn) else float("nan"))

        t0 = time.time()
        try:
            prof = run_scpram_floor(cs, sp, 0, args.epochs, args.ratio, args.gpu, args.cap)
            err = None
        except Exception as e:  # noqa: BLE001
            print(f"  [{frac_label}] FAILED: {type(e).__name__}: {e}", flush=True)
            prof, err = None, f"{type(e).__name__}: {e}"
        dt = round(time.time() - t0, 1)

        if prof is not None:
            pred_cells = np.tile(prof[None, :], (len(ts), 1)).astype(np.float32)
            pe = float(pearson_delta(pred_cells, test_X, ctrl_mean, ts, excl_idx)["macro"])
            ed = float(e_distance(pred_cells, test_X, ts, fit_on=ed_basis)["macro"])
            au = ((float(aucell(prof[None, :], gs_ifn).mean()) - ctrl_auc) if len(gs_ifn) else float("nan"))
            ran = True
            non_degenerate = bool(np.isfinite(prof).all() and float(prof.std()) > 1e-6)
        else:
            pe = ed = au = float("nan"); ran = False; non_degenerate = False

        row = dict(
            model="scPRAM", cluster="C4", dataset=ds_name, modality="RNA", family="optimal-transport",
            split=spec.name, frac_held=frac_label,
            action="run_floor", headline_eligible=False,   # structurally not-defined on KO
            applicability_note="not_defined on C4_Axis2 (KO, not paired stimulation); run as OT floor",
            ran=ran, leak_free=bool(audit["leak_free"]),
            n_train=int(len(sp.train_idx)), n_test=int(len(sp.test_idx)),
            n_test_strata=int(len(np.unique(ts))), n_ctrl=int(len(sp.inference_input_idx)),
            pearson_delta=pe, e_distance=ed, aucell_ifn_delta=au, obs_ifn_delta=obs_ifn,
            linear_pca_pearson=round(lin_pca, 4),
            cell_mean_pearson=round(floors["cell-mean"]["pearson"], 4),
            cell_mean_edist=round(floors["cell-mean"]["edist"], 4),
            scgen_anchor=round(SCGEN_ANCHOR.get(frac_label, float("nan")), 4),
            floor_non_degenerate=non_degenerate, epochs=args.epochs, cap=args.cap,
            subsample_per_group=args.subsample_per_group, elapsed_s=dt, smoke=bool(args.smoke),
            error=err,
        )
        rows = [rr for rr in rows if not (str(rr.get("split")) == spec.name and not rr.get("smoke", False))]
        rows.append(row)
        timing.append(dict(frac=frac_label, sec=dt, n_train=int(len(sp.train_idx)),
                           n_test=int(len(sp.test_idx))))
        print(json.dumps({k: row[k] for k in (
            "split", "frac_held", "action", "headline_eligible", "ran", "leak_free", "pearson_delta",
            "e_distance", "aucell_ifn_delta", "linear_pca_pearson", "scgen_anchor",
            "floor_non_degenerate", "elapsed_s", "error")}, default=str), flush=True)

        if not args.smoke:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            json.dump(timing, open(args.timing_out, "w"), indent=2)

    if args.smoke:
        smoke_path = out_path.with_name("scpram_frangieh_smoke.csv")
        pd.DataFrame(rows).to_csv(smoke_path, index=False)
        print(f"\n[SMOKE] wrote {smoke_path}", flush=True)
        r0 = rows[-1] if rows else {}
        # floor smoke passes if the path runs leak-free and emits a finite, non-degenerate profile
        ok = bool(r0.get("ran") and r0.get("leak_free") and r0.get("floor_non_degenerate")
                  and r0.get("pearson_delta") == r0.get("pearson_delta"))
        print(f"[SMOKE] path_ok={ok} pearson_delta={r0.get('pearson_delta')} "
              f"non_degenerate={r0.get('floor_non_degenerate')} error={r0.get('error')}", flush=True)
        sys.exit(0 if ok else 1)

    print(f"\nWROTE {out_path} ({len(rows)} rows) — scPRAM run as an OT FLOOR (not headline-eligible).",
          flush=True)


if __name__ == "__main__":
    main()
