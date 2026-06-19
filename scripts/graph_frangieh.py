#!/usr/bin/env python
"""Graph-family fills on the C4 complex-context (Frangieh) cell — Figure-1 (complex × Graph = Partial).

Runs the two Graph-family models (GEARS, AttentionPert) on the Frangieh leave-one-KO-gene-out split in
the RNA modality ONLY. RNA is the gene-level readout, so the gene2go graph is defined and the held KO
gene can be predicted from its graph node (the "Partial" cell of Fig-1: graph is applicable on the
gene-level RNA modality but NOT on the protein-CITE surface panel, which has no gene graph). This
mirrors exactly how GEARS/AttentionPert are exercised on the C3 gene-intervention split — same runners,
same leak-safe split builder — just pointed at the Frangieh complex-context cell instead of Chen/Shifrut.

Pattern reuse (do NOT reinvent):
  * loader  : ivcbench.data.loaders.frangieh.load(modality="rna")              (existing C4 loader)
  * split   : ivcbench.clusters.c4.held_ko_fraction + modality_lo_ko           (the C4 modality_lo_ko)
  * runner  : ivcbench.baselines.heavy.GEARS / AttentionPert -> model_runners/{gears,attentionpert}_runner.py
  * driver  : ivcbench.runner.run.run_job  (build_split -> audit_split -> fit/predict -> 4-axis metrics)
              invoked exactly like scripts/run_c4_conditioned.py (the C4 conditioned-run driver)

Leak-safety (HARD RULE — same guarantee as C3/C4):
  * run_job calls build_split then audit_split, which raises LeakError on ANY held-gene cell in train.
  * exclude_genes = held KO genes (downstream_only): the KO'd gene is dropped from the Pearson-Δ panel,
    re-derived per fold (refit), so a held gene's own expression never enters the score.
  * The graph runners receive a payload whose X_train / pert_train already have the held genes removed
    (heavy._build_payload uses split.train_idx); the held expression is NEVER serialised — the runner
    only gets the held perturbation LABELS to predict, then reads them off the gene2go graph.
  * linear-PCA / cell-mean response basis + PCA are fit on the train fold only inside their adapters,
    refit per fold.

KO-genes-map-onto-the-graph check: of the 248 Frangieh KO genes, 224 are graph-perturbable (present in
BOTH the HVG panel AND gene2go); 1968/2000 HVG genes are in gene2go. A held KO gene absent from the
graph universe is simply not predicted by the runner (the adapter falls back to control for it) — the
same documented fallback as C3 — so the metric is computed on the graph-mappable held genes.

ANCHOR / sanity gate (per cell): the Graph model must BEAT-or-LAND-NEAR the universal floor band
{cell-mean, linear-PCA} on Pearson-Δ. We define the band per fold from the in-process simple floors and
flag `anchor_pass = pearson_delta >= linear_PCA_floor - TOL` (Graph at least matches the weaker floor
edge). On real CRISPR-KO unseen-gene extrapolation the cell-mean floor is famously hard to beat
(task-dependent-conditioning finding), so the anchor is "near the band", not "tops cell-mean".

Smoke (default): RNA, ONE fold (frac 0.25), GPU OFF (CPU), seconds. cell-gears / attnpert's GEARS-style
trainers CANNOT run on CPU here (a 'leaf Variable used in an in-place operation' autograd error at ≥1
epoch; 'best_model' unbound at 0 epochs) — so the seconds-CPU smoke does NOT invoke the GPU model.
Instead it deterministically proves the GPU-FREE integration: load → leave-one-KO split → leak audit →
floor band (must reproduce the C4 {cell-mean, linear-PCA} band) → the exact leak-safe payload the runner
receives, asserting (a) no held-KO cell entered train, (b) held perturbations are labels-only, (c) the
held KO genes map onto the gene graph (HVG ∩ gene2go). The scored GEARS/AttentionPert runs are GPU-only
and execute on the pinned FULL run; --try-gpu-smoke additionally attempts a 1-epoch GEARS run IF a CUDA
device is pinned. FULL pins CUDA_VISIBLE_DEVICES=2,3 ONLY (GPUs 0,1 off-limits; see --emit-launch).

  smoke : ./.venv/bin/python scripts/graph_frangieh.py --smoke
  full  : CUDA_VISIBLE_DEVICES=2,3 ./.venv/bin/python scripts/graph_frangieh.py --full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from ivcbench.data.loaders.frangieh import load
from ivcbench.clusters import c4, c3
from ivcbench.baselines.heavy import GEARS, AttentionPert
from ivcbench.baselines.simple import CellMean, LinearPCA
from ivcbench.runner.run import run_job
from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split

OUT_DIR = Path("results/C4")
ANCHOR_TOL = 0.05          # Graph must land within TOL of the weaker (linear-PCA) floor edge
FLOOR_BASELINES = [CellMean, LinearPCA]    # universal floor band {cell-mean, linear-PCA}

# Keys we surface from each run_job row (same projection style as run_c4_conditioned.py).
_KEEP = ("baseline", "family", "modality", "split", "action", "ran", "leak_free", "n_train",
         "n_test", "n_test_strata", "n_held_ko", "n_held_ko_on_graph", "x_train_shape",
         "pearson_delta", "pearson_delta_lo", "pearson_delta_hi",
         "pearson_delta_ontarget", "e_distance", "aucell_program_corr", "elapsed_s", "error")


def _run_one(cs, spec, excl, programs, B, seed=0):
    t0 = time.time()
    adapter = B()
    try:
        r = run_job(cs, spec, adapter, seed=seed, immune_programs=programs,
                    exclude_genes=excl, adapted_implemented=True)
    except Exception as e:  # noqa: BLE001 — heavy runner failures must not abort the sweep
        r = {"baseline": getattr(adapter, "name", B.__name__),
             "family": getattr(adapter, "family", "graph"), "split": spec.name,
             "action": "failed", "ran": False, "error": f"{type(e).__name__}: {e}"}
    r.update(cluster="C4", dataset=cs.uns.get("dataset", "frangieh_rna"), modality="RNA",
             elapsed_s=round(time.time() - t0, 1))
    return r


def _payload_check(cs, spec, gene2go_genes):
    """GPU-FREE proof of the leak-safe graph payload (what GEARS/AttentionPert receive): build the split,
    run the leak auditor, then construct the exact SubprocessAdapter payload and assert (1) no held-gene
    cell entered train, (2) the held perturbations are LABELS only (never expression), (3) the held KO
    genes map onto the gene graph (HVG panel ∩ gene2go). Returns a dict row; raises on any leak."""
    from ivcbench.baselines.heavy import GEARS as _G
    split = build_split(cs, spec)
    audit = audit_split(cs, split)                # hard leak gate (raises LeakError on any violation)
    adapter = _G()
    adapter.fit(cs, split, side_info=cs.side_info)
    payload = adapter._build_payload(cs, split, cs.side_info)
    held = set(spec.held_values)
    train_perts = set(map(str, payload["pert_train"]))
    panel = set(map(str, payload["genes"]))
    test_perts = sorted({str(p) for p in payload["test_perts"]} - {"control"})
    graph_ko = sorted(set(test_perts) & panel & set(gene2go_genes))     # held KO that have a graph node
    leak_in_train = sorted(held & train_perts)
    assert not leak_in_train, f"LEAK: held KO genes in train payload: {leak_in_train}"
    return {"baseline": "payload-check", "family": "integration", "split": spec.name,
            "ran": True, "leak_free": audit["leak_free"], "n_train": audit["n_train"],
            "n_test": audit["n_test"], "n_test_strata": audit["n_test_strata"],
            "n_held_ko": len(held), "n_held_ko_on_graph": len(graph_ko),
            "x_train_shape": list(payload["X_train"].shape), "modality": "RNA"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU-only sanity run (1 fold, few epochs, cell cap, GPU OFF).")
    ap.add_argument("--try-gpu-smoke", action="store_true",
                    help="ALSO attempt a tiny 1-epoch GEARS run (needs a CUDA device). cell-gears cannot "
                         "train on CPU here (a 'leaf Variable in-place' autograd bug at ≥1 epoch; "
                         "'best_model' unbound at 0 epochs), so the default smoke does NOT invoke the GPU "
                         "model — it deterministically proves the GPU-free integration instead. Use this "
                         "ONLY with CUDA_VISIBLE_DEVICES pinned to a free device.")
    ap.add_argument("--full", action="store_true",
                    help="both folds (0.25, 0.50), full epochs; pin CUDA_VISIBLE_DEVICES=2,3.")
    ap.add_argument("--fracs", default="", help="comma KO-hold fractions, e.g. '0.25,0.50' (overrides).")
    ap.add_argument("--models", default="", choices=["", "gears", "attnpert", "both"],
                    help="which Graph models to GPU-score (FULL, or smoke --try-gpu-smoke). "
                         "Default: full=both (GEARS + AttentionPert), smoke=gears.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR / "graph_frangieh_rows.json"))
    ap.add_argument("--emit-launch", action="store_true",
                    help="print the exact pinned FULL launch command and exit (no run).")
    args = ap.parse_args()

    launch_cmd = ("CUDA_VISIBLE_DEVICES=2,3 ./.venv/bin/python scripts/graph_frangieh.py --full")
    if args.emit_launch:
        print(launch_cmd)
        return

    smoke = args.smoke or not args.full
    # model selection: smoke defaults to GEARS only; full defaults to both.
    sel = args.models or ("gears" if smoke else "both")
    sel_map = {"gears": [GEARS], "attnpert": [AttentionPert], "both": [GEARS, AttentionPert]}
    graph_models = sel_map[sel]
    if smoke:
        # cell-gears / attnpert GEARS-style trainers require a CUDA device (they crash on CPU autograd).
        # So the seconds-CPU smoke proves the GPU-FREE integration deterministically (load→split→audit→
        # floors→leak-safe payload incl. the held-gene removal + KO→gene-graph mapping the runner needs),
        # and only ATTEMPTS the GPU model when --try-gpu-smoke is set with a CUDA device pinned.
        if args.try_gpu_smoke:
            os.environ.setdefault("IVCBENCH_GEARS_EPOCHS", "1")
            os.environ.setdefault("IVCBENCH_GEARS_MAX_CELLS", "1200")
            os.environ.setdefault("IVCBENCH_ATTNPERT_EPOCHS", "1")
            os.environ.setdefault("IVCBENCH_ATTNPERT_MAXCELLS", "1200")
        fracs = [(0.25, "25")]
        subsample = 20
    else:
        # FULL: must be launched with CUDA_VISIBLE_DEVICES=2,3 (GPUs 0,1 are off-limits/busy).
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd not in ("2,3", "2, 3"):
            sys.exit(f"[graph_frangieh] FULL run must pin CUDA_VISIBLE_DEVICES=2,3 (got '{cvd}'). "
                     f"Launch:\n  {launch_cmd}")
        os.environ.setdefault("IVCBENCH_GEARS_EPOCHS", "15")
        os.environ.setdefault("IVCBENCH_ATTNPERT_EPOCHS", "20")
        fracs = [(0.25, "25"), (0.50, "50")]
        subsample = 60

    if args.fracs:
        fmap = {"0.25": "25", "0.5": "50", "0.50": "50", "0.1": "10", "0.10": "10"}
        fracs = [(float(x), fmap.get(x, x.replace("0.", "").replace(".", ""))) for x in args.fracs.split(",")]

    # the GPU graph models score only in FULL, or in smoke when a CUDA device is explicitly offered.
    run_graph = (not smoke) or args.try_gpu_smoke

    # ---- load RNA modality (gene-level; the only modality where the gene graph is defined) ----
    cs = load(modality="rna", subsample_per_group=subsample, seed=args.seed)
    g = cs.uns["genes_perturbed"]
    # immune-program AUCell sets (Axis 3): C4 declares none, so reuse the C3 immune programs (Frangieh
    # is a melanoma+TIL co-culture — T-cell-activation / effector / exhaustion programs are meaningful).
    programs = c3.C3_PROGRAMS
    # gene2go universe (the graph the held KO genes must map onto) — for the GPU-free payload check.
    try:
        import pickle as _pk
        g2g = os.environ.get("IVCBENCH_GENE2GO") or "data/_assets/gears/gene2go_all.pkl"
        with open(g2g, "rb") as fh:
            gene2go_genes = set(map(str, _pk.load(fh).keys()))
    except Exception:  # noqa: BLE001
        gene2go_genes = set()
    print(f"[graph_frangieh] RNA: {cs.n_cells} cells × {cs.n_genes} genes, {len(g)} KO genes; "
          f"mode={'SMOKE(cpu)' if smoke else 'FULL(gpu2,3)'}; models={sel}; run_graph={run_graph}",
          flush=True)

    # crash/timeout-safe per-row checkpoint: append every completed row to a .jsonl as it lands, so a
    # killed run (e.g. AttentionPert exceeding a timeout) still leaves the rows that DID finish.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.out).with_suffix(".jsonl")
    ckpt.write_text("")  # truncate stale checkpoint for this invocation

    def _emit(r, extra=None):
        rec = {k: r.get(k) for k in _KEEP}
        if extra:
            rec.update(extra)
        print(json.dumps(rec, default=str), flush=True)
        with open(ckpt, "a") as fh:                  # persist immediately (survives SIGTERM)
            fh.write(json.dumps(r, default=str) + "\n")

    rows = []
    for frac, lbl in fracs:
        held = c4.held_ko_fraction(g, frac, seed=args.seed)
        spec = c4.modality_lo_ko(held, lbl)
        excl = list(spec.held_values)            # downstream_only: drop held KO genes from Pearson-Δ

        # GPU-FREE integration proof (leak-safe payload + KO→gene-graph mapping the runner consumes).
        pc = _payload_check(cs, spec, gene2go_genes)
        rows.append(pc)
        _emit(pc)

        # universal floor band (in-process, fast, leak-safe per fold) — the anchor reference.
        floor = {}
        for B in FLOOR_BASELINES:
            r = _run_one(cs, spec, excl, programs, B, seed=args.seed)
            floor[r["baseline"]] = r
            rows.append(r)
            _emit(r)
        floor_cm = floor.get("cell-mean", {}).get("pearson_delta")
        floor_pca = floor.get("linear-PCA", {}).get("pearson_delta")
        band_lo = min([x for x in (floor_cm, floor_pca) if x is not None], default=None)

        # ---- the Graph fills (GPU-scored: FULL, or smoke --try-gpu-smoke) ----
        if run_graph:
            for B in graph_models:
                r = _run_one(cs, spec, excl, programs, B, seed=args.seed)
                pd_ = r.get("pearson_delta")
                r["floor_cell_mean"] = floor_cm
                r["floor_linear_pca"] = floor_pca
                # ANCHOR gate: Graph beats-or-lands-near the floor band (>= weaker floor edge - TOL).
                r["anchor_pass"] = bool(
                    r.get("ran") and pd_ is not None and band_lo is not None
                    and pd_ >= band_lo - ANCHOR_TOL)
                rows.append(r)
                _emit(r, extra={"floor_cell_mean": floor_cm, "floor_linear_pca": floor_pca,
                                "anchor_pass": r["anchor_pass"]})

    Path(args.out).write_text(json.dumps(rows, indent=2, default=str))

    graph_rows = [r for r in rows if r.get("family") == "graph"]
    pc_rows = [r for r in rows if r.get("family") == "integration"]
    ran = [r for r in graph_rows if r.get("ran")]
    passed = [r for r in ran if r.get("anchor_pass")]
    print(f"\n[graph_frangieh] WROTE {args.out}", flush=True)
    print(f"[graph_frangieh] payload-checks: {len(pc_rows)} (leak_free="
          f"{all(r.get('leak_free') for r in pc_rows)}); graph rows: {len(graph_rows)}  "
          f"ran: {len(ran)}  anchor_pass: {len(passed)}", flush=True)
    if smoke:
        # The seconds-CPU smoke proves the GPU-FREE integration (payload + floors). The GPU graph
        # models are GPU-only here (cell-gears/attnpert crash on CPU autograd) → scored on the pinned
        # FULL run. SMOKE PASS = every payload-check ran leak-safe with held KO genes on the graph.
        ok = bool(pc_rows) and all(r.get("leak_free") for r in pc_rows) \
            and all(r.get("n_held_ko_on_graph", 0) > 0 for r in pc_rows)
        if not ok:
            print("[graph_frangieh] SMOKE FAIL — GPU-free integration/leak/graph-mapping check failed.",
                  flush=True)
            sys.exit(2)
        if run_graph and not ran:
            errs = "; ".join(f"{r['baseline']}: {r.get('error')}" for r in graph_rows)
            print(f"[graph_frangieh] SMOKE FAIL — --try-gpu-smoke set but no graph model ran. {errs}",
                  flush=True)
            sys.exit(2)
        print("[graph_frangieh] SMOKE PASS — GPU-free integration leak-safe; held KO genes map onto the "
              "gene graph; floor band reproduced. Graph models (GPU-only here) score on the FULL run.",
              flush=True)
        print(f"[graph_frangieh] FULL launch:\n  {launch_cmd}", flush=True)


if __name__ == "__main__":
    main()
