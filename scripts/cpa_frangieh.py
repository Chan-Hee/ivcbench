#!/usr/bin/env python
"""CPA -> Frangieh (C4 complex-context, Axis-2 modality generalization) — ④xLatent (Fig-1 PARTIAL).

CPA (Lotfollahi et al. 2023) is the LATENT-family fill on the complex-context cell. It is `applicable`
(headline-eligible) on C4_Axis2 in the registry, alongside scGen (the 1st latent model, already run).
This driver mirrors scripts/cinemaot_frangieh.py EXACTLY (same C4 modality_lo_ko split, same Frangieh RNA
loader, same run_job pipeline, same anchor/sanity gate, same per-fold leak-safe scoring) but swaps the
CINEMA-OT floor for the CPA `adapted` runner (model_runners/cpa_runner.py, ivc-cpa env): CPA trains on
the leak-safe train cells, takes the per-train-gene latent shift δ in CPA's latent space, regresses δ on
a LEAK-SAFE control-only-PCA gene embedding, predicts δ for each held KO gene, and decodes
(z_ctrl + δ_held) through CPA's generative head. This is the SAME `adapted` strategy scGen uses on this
split (and CPA uses on C3) — the held KO is predicted from its gene-side embedding, never from held-KO
expression. CPA is the 2nd conditioned latent model; the Fig-1 ④xLatent cell is PARTIAL (we expect CPA
to land near/under the scGen anchor, possibly below the strong mean-shift oracle).

RNA ONLY (matches the scGen / CINEMA-OT Frangieh RNA anchor). Protein (CITE) NOT run (disclosed).

LEAK-SAFE (hard rule): execution goes through the SAME run_job pipeline as every C4 entry:
build_split -> audit_split (hard LeakError gate) -> CPA.predict. The runner only ever receives the
leak-safe payload SubprocessAdapter._build_payload emits: train-fold expression, the train-fold control
mask, the NON-held control cells (split.inference_input_idx), and the held KO LABELS only — never held-out
expression. The control-only-PCA gene embedding is fit inside the runner on control cells only. The
e_distance PCA basis is fit on cs.X[split.train_idx] (train fold). Per-fold refit: each (fraction) fold
rebuilds the split and retrains CPA from scratch.

ANCHOR / sanity gate (--anchor, default on): per fold, CPA is adoptable only if it
  (a) lands within an order of magnitude of the scGen Frangieh-RNA pearson_delta anchor (0.5376 @ 25,
      0.5533 @ 50): we require 0.05 <= pearson_delta <= 0.95 (a touch wider on the low side than the OT
      gate — the ④xLatent cell is PARTIAL, so a sub-scGen-but-positive CPA is an HONEST partial result,
      not a failure). A NEGATIVE or NaN pearson_delta fails the gate.
  (b) The linear-PCA floor (~0.29) is reported for context (delta_vs_linpca); CPA need NOT beat the
      strong perturbation-agnostic mean-shift oracle (~0.67) — that oracle is not the gate (the
      conditioned scGen reference 0.538 does not beat it either).
A fold that fails the gate is marked anchor_pass=False; the driver exits non-zero in --anchor mode if any
non-smoke fold fails, so a bad result can never be silently adopted.

--chunk I N shards the fold-units (lo_ko_25 / lo_ko_50) one per GPU worker slot. SMOKE: --smoke runs one
tiny fold (lo_ko_25) with few CPA epochs + small cell cap purely to prove the path runs end-to-end; NOT
the full job.

Envs: ivc-cpa (cpa-tools 0.8.8). FULL-run GPU budget = devices 2,3 ONLY (0,1 off-limits).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivcbench.data.loaders.frangieh import load
from ivcbench.clusters import c4
from ivcbench.runner.run import run_job
from ivcbench.splits.builder import build_split
from ivcbench.metrics.response import pearson_delta
from ivcbench.metrics.distribution import e_distance
from ivcbench.metrics.program import aucell
from ivcbench.baselines.heavy import CPA
from ivcbench.baselines.simple import CellMean, DonorShift, CtrlPred, LinearPCA

SCGEN_ANCHOR = {"25": 0.5376423142039247, "50": 0.5533400144983804}
LINEAR_PCA_FLOOR = {"25": 0.2914931868172942, "50": 0.27515802689782914}
# PARTIAL cell: a positive sub-scGen CPA is an honest partial result, so the low bound is wider than the
# OT gate. Negative / NaN fails.
ANCHOR_LO, ANCHOR_HI = 0.05, 0.95

# Frangieh IFNγ immune-evasion program (same set as cellot_frangieh.py) for the AUCell-Δ axis.
FRANGIEH_IFN_PROGRAM = ["STAT1", "IRF1", "GBP1", "GBP2", "GBP5", "CXCL9", "CXCL10", "CXCL11",
                        "TAP1", "TAP2", "PSMB8", "PSMB9", "HLA-A", "HLA-B", "HLA-C", "B2M",
                        "IDO1", "CD274", "NLRC5", "IFI30", "UBE2L6", "WARS"]


def _floors(cs, spec, excl, gs_ifn):
    """Recompute simple-baseline pearson/edist/aucell-Δ floors for THIS fold (leak-safe, in-process)."""
    from sklearn.decomposition import PCA  # noqa: F401  (kept for parity; edist uses train basis)
    sp = build_split(cs, spec)
    test_X = cs.X[sp.test_idx]; ts = sp.test_strata
    ctrl_X = cs.X[sp.inference_input_idx]
    ed_basis = cs.X[sp.train_idx]
    if len(ed_basis) > 5000:
        ed_basis = ed_basis[np.random.default_rng(0).choice(len(ed_basis), 5000, replace=False)]
    exi = np.asarray(cs.gene_index(excl), dtype=int) if excl else None
    ctrl_auc = float(aucell(ctrl_X, gs_ifn).mean()) if len(gs_ifn) else float("nan")
    out = {}
    for B in (CtrlPred, CellMean, DonorShift, LinearPCA):
        b = B(); b.fit(cs, sp, side_info=cs.side_info)
        p = b.predict(cs, sp, side_info=cs.side_info)
        out[b.name] = dict(
            pearson=float(pearson_delta(p.pred_cells, test_X, p.control_mean, ts, exi)["macro"]),
            edist=float(e_distance(p.pred_cells, test_X, ts, fit_on=ed_basis)["macro"]),
            aucell=((float(aucell(p.pred_cells, gs_ifn).mean()) - ctrl_auc) if len(gs_ifn)
                    else float("nan")),
        )
    obs_ifn = ((float(aucell(test_X, gs_ifn).mean()) - ctrl_auc) if len(gs_ifn) else float("nan"))
    return out, obs_ifn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny 1-fold path-proof (few epochs, small cap); does NOT run the full job")
    ap.add_argument("--chunk", type=int, nargs=2, default=None, metavar=("I", "N"))
    ap.add_argument("--fractions", nargs="*", default=["25", "50"])
    ap.add_argument("--cpa-epochs", type=int, default=60, help="CPA max_epochs (IVCBENCH_CPA_EPOCHS)")
    ap.add_argument("--cpa-maxcells", type=int, default=None, help="cap train cells (IVCBENCH_CPA_MAXCELLS)")
    ap.add_argument("--subsample-per-group", type=int, default=60)
    ap.add_argument("--gpu", type=str, default=None,
                    help="CUDA_VISIBLE_DEVICES for the CPA subprocess (FULL runs: 2 or 3 ONLY)")
    ap.add_argument("--anchor", action="store_true", default=True)
    ap.add_argument("--no-anchor", dest="anchor", action="store_false")
    ap.add_argument("--out", default=str(ROOT / "outputs/additional_models/cpa_frangieh_raw.csv"))
    ap.add_argument("--timing-out",
                    default=str(ROOT / "outputs/additional_models/cpa_frangieh_timing.json"))
    ap.add_argument("--timeout-s", type=int, default=7200)
    args = ap.parse_args()

    if args.smoke:
        args.fractions = ["25"]
        args.cpa_epochs = 2
        args.cpa_maxcells = 600
        args.subsample_per_group = 12
        if args.gpu is None:
            args.gpu = ""                                # CPU smoke

    os.environ["IVCBENCH_CPA_EPOCHS"] = str(args.cpa_epochs)
    if args.cpa_maxcells is not None:
        os.environ["IVCBENCH_CPA_MAXCELLS"] = str(args.cpa_maxcells)

    print(f"[load] Frangieh RNA (subsample_per_group={args.subsample_per_group}); gpu={args.gpu}; "
          f"cpa_epochs={args.cpa_epochs}", flush=True)
    cs = load(modality="rna", subsample_per_group=args.subsample_per_group)
    genes_perturbed = cs.uns["genes_perturbed"]
    ds_name = cs.uns.get("dataset", "frangieh_rna")
    gs_ifn = np.asarray(cs.gene_index([g for g in FRANGIEH_IFN_PROGRAM if g in set(cs.var_names)]),
                        dtype=int)
    # gene-side embedding for CPA is built INSIDE cpa_runner.py from control-only PCA — but CPA on C4 is
    # `adapted`; run_job needs adapted_implemented=True (the gene map is leak-safe-by-construction).
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
        excl = list(spec.held_values)
        excl_idx = np.asarray(cs.gene_index(excl), dtype=int) if excl else None

        floors, obs_ifn = _floors(cs, spec, excl, gs_ifn)
        lin_pca = floors["linear-PCA"]["pearson"]

        adapter = CPA()
        if args.gpu is not None:
            adapter.cuda_device = args.gpu
        adapter.timeout_s = args.timeout_s

        t0 = time.time()
        try:
            r = run_job(cs, spec, adapter, seed=0, exclude_genes=excl,
                        immune_programs={"frangieh_ifn": [g for g in FRANGIEH_IFN_PROGRAM
                                                          if g in set(cs.var_names)]},
                        adapted_implemented=True)
        except Exception as e:  # noqa: BLE001
            r = {"baseline": adapter.name, "family": adapter.family, "split": spec.name,
                 "action": "failed", "ran": False, "error": f"{type(e).__name__}: {e}"}
        dt = round(time.time() - t0, 1)

        pe = r.get("pearson_delta"); ed = r.get("e_distance")
        # Axis-3 = run_job's across-strata AUCell-Δ correlation on the Frangieh IFNγ program (the headline
        # immune-program metric). The floors' aucell magnitude (cell_mean_aucell) is logged for context.
        au_corr = r.get("aucell_program_corr")
        ran = bool(r.get("ran"))
        leak_free = bool(r.get("leak_free", False))

        scgen = SCGEN_ANCHOR.get(frac_label, float("nan"))
        in_window = bool(ran and pe is not None and pe == pe and ANCHOR_LO <= pe <= ANCHOR_HI)
        beats_linpca = bool(ran and pe is not None and pe == pe and pe > lin_pca)
        anchor_pass = bool(in_window)                    # PARTIAL cell: window membership is the gate
        if not args.smoke and args.anchor and not anchor_pass:
            any_fail = True

        row = dict(
            model="CPA", cluster="C4", dataset=ds_name, modality="RNA", family="latent",
            split=spec.name, frac_held=frac_label, action=r.get("action"), ran=ran, leak_free=leak_free,
            n_train=r.get("n_train"), n_test=r.get("n_test"), n_test_strata=r.get("n_test_strata"),
            pearson_delta=pe, pearson_delta_ontarget=r.get("pearson_delta_ontarget"),
            e_distance=ed, aucell_program_corr=au_corr,
            linear_pca_pearson=round(lin_pca, 4),
            cell_mean_pearson=round(floors["cell-mean"]["pearson"], 4),
            cell_mean_edist=round(floors["cell-mean"]["edist"], 4),
            cell_mean_aucell=(round(floors["cell-mean"]["aucell"], 4)
                              if floors["cell-mean"]["aucell"] == floors["cell-mean"]["aucell"] else None),
            obs_ifn_delta=(round(obs_ifn, 4) if obs_ifn == obs_ifn else None),
            delta_pearson_vs_linpca=(round(pe - lin_pca, 4) if (ran and pe is not None and pe == pe)
                                     else None),
            delta_edist_vs_cellmean=(round(floors["cell-mean"]["edist"] - ed, 4)
                                     if (ran and ed is not None and ed == ed) else None),
            beats_linpca_floor=beats_linpca,
            scgen_anchor=round(scgen, 4) if scgen == scgen else None,
            in_anchor_window=in_window, anchor_pass=anchor_pass,
            cpa_epochs=args.cpa_epochs, subsample_per_group=args.subsample_per_group,
            elapsed_s=dt, smoke=bool(args.smoke), error=r.get("error"),
        )
        rows = [rr for rr in rows if not (str(rr.get("split")) == spec.name and not rr.get("smoke", False))]
        rows.append(row)
        timing.append(dict(frac=frac_label, sec=dt, n_train=r.get("n_train"), n_test=r.get("n_test")))
        print(json.dumps({k: row[k] for k in (
            "split", "frac_held", "ran", "leak_free", "n_train", "n_test", "pearson_delta",
            "e_distance", "aucell_program_corr", "linear_pca_pearson", "beats_linpca_floor",
            "scgen_anchor", "in_anchor_window", "anchor_pass", "elapsed_s", "error")},
            default=str), flush=True)

        if not args.smoke:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            json.dump(timing, open(args.timing_out, "w"), indent=2)

    if args.smoke:
        smoke_path = out_path.with_name("cpa_frangieh_smoke.csv")
        pd.DataFrame(rows).to_csv(smoke_path, index=False)
        print(f"\n[SMOKE] wrote {smoke_path}", flush=True)
        r0 = rows[-1] if rows else {}
        ok = bool(r0.get("ran") and r0.get("leak_free")
                  and r0.get("pearson_delta") is not None
                  and r0.get("pearson_delta") == r0.get("pearson_delta"))
        print(f"[SMOKE] path_ok={ok} pearson_delta={r0.get('pearson_delta')} "
              f"error={r0.get('error')}", flush=True)
        sys.exit(0 if ok else 1)

    print(f"\nWROTE {out_path} ({len(rows)} rows)", flush=True)
    if args.anchor and any_fail:
        print("[ANCHOR] FAIL: at least one fold did not pass the sanity gate "
              "(pearson_delta must be in the scGen order-of-magnitude window [0.05, 0.95]).", flush=True)
        sys.exit(2)
    print("[ANCHOR] PASS (all run folds land in the scGen order-of-magnitude window).", flush=True)


if __name__ == "__main__":
    main()
