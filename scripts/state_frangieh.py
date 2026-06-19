#!/usr/bin/env python
"""STATE -> Frangieh (C4 complex-context, Axis-2 modality generalization) — ④×Hybrid FILL.

Builds the Hybrid-family (STATE) FILL on the Figure-1 complex-context cell where Hybrid is "partial" and
we lack a STATE row. Mirrors scripts/cinemaot_frangieh.py EXACTLY (same loader, same leave-one-KO-gene-out
split c4.modality_lo_ko, same run_job pipeline + hard leak audit, same --chunk 2-GPU sharding, same
smoke + anchor gate, same RNA-only scope) but swaps CINEMA-OT for arc-STATE's State-Transition model.

GENE axis: the held UNIT is an unseen KO gene. STATE is `applicable` (headline-eligible) on C4_Axis2 in
the registry: it predicts a held KO's response from its learned perturbation-feature vector, here the
leak-safe CONTROL-only PCA gene embedding built INSIDE model_runners/state_runner.py (the exact C3
unseen-gene mechanism). cell_load's fewshot split puts train KOs in `train`, held KOs in `test`; the held
KO's query cells are non-targeting controls re-tagged with it (its real KO cells are NEVER a training
target → leak-safe). embed_key=null (from-scratch ST, no SE-600M) — a conservative lower bound.

RNA ONLY. Protein (CITE) is intentionally NOT run (matching the cinemaot_frangieh.py disclosure): the
state_runner builds its gene embedding from the RNA control PCA; the 20-marker CITE panel is out of scope
for this driver.

LEAK-SAFE (HARD RULE): execution goes through the SAME run_job pipeline as every other C4 entry:
build_split -> audit_split (hard LeakError gate) -> STATE.predict on the SubprocessAdapter leak-safe
payload (train-fold expression + the NON-held control cells; the held KO's test expression is NEVER
seen). The response panel = full feature space minus the KO'd gene (downstream_only exclude), and the
E-distance PCA basis is fit on cs.X[split.train_idx] (train fold only). Per-fold refit: each fraction
fold rebuilds the split and retrains ST from scratch.

ANCHOR / sanity gate (--anchor, default on): per fold STATE is adoptable iff
  (a) it BEATS the genuine learned-conditioning-free NON-trivial linear floor (linear-PCA pearson_delta
      ~0.291 @25, ~0.275 @50; recomputed on THIS fold), AND
  (b) it lands in a sane band around the conditioned scGen Frangieh-RNA anchor (0.538 @25, 0.553 @50):
      0.10 <= pearson_delta <= 0.95.
NB cell-mean/donor-shift on Frangieh are a perturbation-agnostic training-mean-shift "oracle" (~0.67)
that the conditioned scGen reference itself does not beat — reported for context, NOT the gate. A fold
that fails the gate is anchor_pass=False; in --anchor mode the driver exits non-zero so a bad/leak-
inflated (>0.95) result can never be silently adopted.

--chunk I N shards the fold-units units[i::n] across exactly 2 GPUs. SMOKE: --smoke runs one tiny fold
(lo_ko_25, RNA) CPU-only (IVCBENCH_STATE_FORCE_CPU=1), tiny steps + tiny cell cap, to prove the path
end-to-end in seconds without touching a GPU. FULL run pins CUDA_VISIBLE_DEVICES=2,3 ONLY (0,1 off-limits).

Envs: ivc-state (arc-state; `state tx`). GPU budget = devices 2,3 ONLY for the full run.
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
from ivcbench.baselines.heavy import STATE
from ivcbench.baselines.simple import CellMean, DonorShift, CtrlPred, LinearPCA

# scGen Frangieh-RNA anchor (results/C4/results_raw.csv) — the conditioned reference STATE should land
# near (order of magnitude).
SCGEN_ANCHOR = {"25": 0.5376423142039247, "50": 0.5533400144983804}
# linear-PCA floor per split from results/C4/results.csv (the learned-conditioning-free non-trivial floor).
LINEAR_PCA_FLOOR = {"25": 0.2914931868172942, "50": 0.27515802689782914}
ANCHOR_LO, ANCHOR_HI = 0.10, 0.95


def _simple_floor_pe(cs, spec, excl):
    """Recompute the simple-baseline pearson_delta floors on THIS exact fold (leak-safe, in-process)."""
    from ivcbench.metrics.response import pearson_delta
    from ivcbench.splits.builder import build_split
    sp = build_split(cs, spec)
    test_X = cs.X[sp.test_idx]
    exi = cs.gene_index(excl) if excl else None
    out = {}
    for B in (CtrlPred, CellMean, DonorShift, LinearPCA):
        b = B(); b.fit(cs, sp, side_info=cs.side_info)
        pr = b.predict(cs, sp, side_info=cs.side_info)
        out[b.name] = float(pearson_delta(pr.pred_cells, test_X, pr.control_mean,
                                           sp.test_strata, exi)["macro"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny 1-fold CPU-only path-proof (forces FORCE_CPU, tiny steps/cap); not the full job")
    ap.add_argument("--chunk", type=int, nargs=2, default=None, metavar=("I", "N"),
                    help="shard fold-units units[i::n] across N (=2) GPUs")
    ap.add_argument("--fractions", nargs="*", default=["25", "50"],
                    help="held-KO fractions to run as folds (default 25 50)")
    ap.add_argument("--steps", type=int, default=400, help="STATE max_steps (IVCBENCH_STATE_STEPS)")
    ap.add_argument("--maxcells", type=int, default=50000, help="STATE train-cell cap (IVCBENCH_STATE_MAXCELLS)")
    ap.add_argument("--subsample-per-group", type=int, default=60,
                    help="cells/KO loaded by the Frangieh loader")
    ap.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES for the STATE subprocess")
    ap.add_argument("--anchor", action="store_true", default=True, help="enforce anchor gate (default on)")
    ap.add_argument("--no-anchor", dest="anchor", action="store_false")
    ap.add_argument("--out", default=str(ROOT / "outputs/additional_models/state_frangieh_raw.csv"))
    ap.add_argument("--timeout-s", type=int, default=7200)
    args = ap.parse_args()

    if args.smoke:
        args.fractions = ["25"]
        args.steps = min(args.steps, 6)
        args.maxcells = 1500
        args.subsample_per_group = 12
    os.environ["IVCBENCH_STATE_STEPS"] = str(args.steps)
    os.environ["IVCBENCH_STATE_MAXCELLS"] = str(args.maxcells)
    if args.smoke:
        os.environ["IVCBENCH_STATE_FORCE_CPU"] = "1"

    print(f"[load] Frangieh RNA (subsample_per_group={args.subsample_per_group})", flush=True)
    cs = load(modality="rna", subsample_per_group=args.subsample_per_group)
    genes_perturbed = cs.uns["genes_perturbed"]
    ds_name = cs.uns.get("dataset", "frangieh_rna")
    print(f"[load] {cs.n_cells} cells x {cs.n_genes} genes; {len(genes_perturbed)} perturbed KOs; "
          f"ctrl_frac={float(cs.obs['is_control'].mean()):.3f}", flush=True)

    units = list(args.fractions)
    if args.chunk:
        i, n = args.chunk
        units = units[i::n]
        print(f"[chunk {i}/{n}] units -> {units}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if not args.smoke and out_path.exists():
        try:
            rows = pd.read_csv(out_path).to_dict("records")
        except Exception:
            rows = []

    any_fail = False
    for frac_label in units:
        frac = 0.25 if frac_label == "25" else 0.50
        held = c4.held_ko_fraction(genes_perturbed, frac, seed=0)
        spec = c4.modality_lo_ko(held, frac_label)
        excl = list(spec.held_values)   # downstream-only: held KO genes excluded from the Pearson score

        floors = _simple_floor_pe(cs, spec, excl)
        best_simple = floors.get("linear-PCA", max(floors.values()))

        adapter = STATE()
        if args.gpu is not None:
            adapter.cuda_device = args.gpu
        elif args.chunk:
            adapter.cuda_device = str(args.chunk[0] % 2)   # placeholder slot; full run pins via --gpu
        adapter.timeout_s = args.timeout_s

        t0 = time.time()
        try:
            r = run_job(cs, spec, adapter, seed=0, exclude_genes=excl, adapted_implemented=True)
        except Exception as e:  # noqa: BLE001
            r = {"baseline": adapter.name, "family": adapter.family, "split": spec.name,
                 "action": "failed", "ran": False, "error": f"{type(e).__name__}: {e}"}
        dt = round(time.time() - t0, 1)

        pe = r.get("pearson_delta")
        ed = r.get("e_distance")
        au = r.get("aucell_program_corr")
        ran = bool(r.get("ran"))
        leak_free = bool(r.get("leak_free", False))

        scgen = SCGEN_ANCHOR.get(frac_label, float("nan"))
        beats_floor = bool(ran and pe is not None and pe == pe and pe > best_simple)
        in_window = bool(ran and pe is not None and pe == pe and ANCHOR_LO <= pe <= ANCHOR_HI)
        anchor_pass = bool(beats_floor and in_window)
        if not args.smoke and args.anchor and ran and not anchor_pass:
            any_fail = True

        row = dict(
            model="STATE", family="hybrid", cluster="C4", dataset=ds_name, modality="RNA",
            split=spec.name, frac_held=frac_label,
            action=r.get("action"), ran=ran, leak_free=leak_free,
            n_train=r.get("n_train"), n_test=r.get("n_test"), n_test_strata=r.get("n_test_strata"),
            pearson_delta=pe, pearson_delta_ontarget=r.get("pearson_delta_ontarget"),
            e_distance=ed, aucell_program_corr=au,
            best_simple_floor=round(best_simple, 4),                       # = linear-PCA (gate floor)
            cell_mean_floor=round(floors.get("cell-mean", float("nan")), 4),  # oracle-ish, context only
            beats_simple_floor=beats_floor,
            scgen_anchor=round(scgen, 4) if scgen == scgen else None, in_anchor_window=in_window,
            anchor_pass=anchor_pass, steps=args.steps, maxcells=args.maxcells,
            subsample_per_group=args.subsample_per_group, elapsed_s=dt, smoke=bool(args.smoke),
            error=r.get("error"),
        )
        rows.append(row)
        print(json.dumps({k: row[k] for k in (
            "split", "frac_held", "ran", "leak_free", "n_train", "n_test", "pearson_delta",
            "e_distance", "best_simple_floor", "beats_simple_floor", "scgen_anchor",
            "in_anchor_window", "anchor_pass", "elapsed_s", "error")}, default=str), flush=True)

        if not args.smoke:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    if args.smoke:
        smoke_path = out_path.with_name("state_frangieh_smoke.csv")
        pd.DataFrame(rows).to_csv(smoke_path, index=False)
        ok = bool(rows and rows[0].get("ran") and rows[0].get("leak_free")
                  and rows[0].get("pearson_delta") is not None
                  and rows[0].get("pearson_delta") == rows[0].get("pearson_delta"))
        print(f"\n[SMOKE] wrote {smoke_path}; path_ok={ok} "
              f"(pearsonD={rows[0].get('pearson_delta') if rows else 'NA'})", flush=True)
        sys.exit(0 if ok else 1)

    print(f"\nWROTE {out_path} ({len(rows)} rows)", flush=True)
    if args.anchor and any_fail:
        print("[ANCHOR] FAIL: ≥1 fold did not beat the linear-PCA floor or land in the scGen window.",
              flush=True)
        sys.exit(2)
    print("[ANCHOR] PASS (all run folds beat the simple floor and land near the scGen anchor).",
          flush=True)


if __name__ == "__main__":
    main()
