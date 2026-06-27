#!/usr/bin/env python
"""One paper cycle for a cluster: run all (split × baseline × seed) → results table + reproducibility
manifest + figure + draft + Supplementary Methods. Fully generic over the ClusterSpec registry —
adding a cluster never touches this driver.

    python scripts/run_cluster.py --cluster C3 --real --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ivcbench.clusters.spec import REGISTRY
from ivcbench.report.docx_export import build_docx
from ivcbench.report.draft import write_draft
from ivcbench.report.figures import c1_figure, c3_figure, c4_figure, c5_figure, cluster_figure
from ivcbench.report.methods import write_methods
from ivcbench.runner.manifest import write_manifest
from ivcbench.runner.run import run_job

METRICS = ["pearson_delta", "pearson_delta_ontarget", "e_distance", "aucell_program_corr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="C1", choices=list(REGISTRY))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--real", action="store_true", help="use the real dataset loader if available")
    ap.add_argument("--gpus", default="", help="comma GPU ids to run heavy jobs in parallel, e.g. "
                    "'0,1' (one heavy job per GPU concurrently). Empty = serial. Leave a GPU free on "
                    "shared servers.")
    ap.add_argument("--reuse", default="", help="path to a prior results_raw.csv: reuse already-"
                    "computed (dataset, split, baseline, seed) rows verbatim and only run the rows "
                    "missing from it (e.g. newly added baselines). Regenerates all reports from the "
                    "merged table via the normal code path — no separate merge step.")
    ap.add_argument("--test", type=int, default=0, help="run only the first K splits/units per dataset "
                    "(de-risk subset; 0 = all). Does not change which adapters run, only how many units.")
    ap.add_argument("--only", default="", help="comma-separated baseline names to run (e.g. "
                    "'cell-mean,linear-PCA,ctrl-pred,donor-shift,scGen'); empty = all spec.baselines. "
                    "Used to deposit a cluster's curated scheme without its bespoke-runner duplicates.")
    args = ap.parse_args()
    only_set = {s.strip() for s in args.only.split(",") if s.strip()}
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    spec = REGISTRY[args.cluster]
    rows = []
    ctx = {}

    # --reuse: load prior raw rows keyed by (dataset, split, baseline, seed). A matching job returns
    # the cached row verbatim (preserving the validated numbers); only cache-misses (new baselines)
    # are actually computed. Accepts a results_raw.csv OR a results_checkpoint.jsonl (so a sweep
    # killed mid-run — e.g. by a remote-session SIGHUP — resumes from its checkpoint). The final
    # agg/figure/draft are rebuilt from the merged `rows`.
    reuse: dict = {}
    if args.reuse:
        if str(args.reuse).endswith(".jsonl"):
            recs = [json.loads(ln) for ln in Path(args.reuse).read_text().splitlines() if ln.strip()]
        else:
            recs = [{k: v for k, v in rec.items() if not (isinstance(v, float) and pd.isna(v))}
                    for rec in pd.read_csv(args.reuse).to_dict("records")]
        for rec in recs:
            if rec.get("seed") is None or (isinstance(rec.get("seed"), float) and pd.isna(rec.get("seed"))):
                continue
            # don't cache timeout failures — let them recompute (e.g. after a longer timeout / cell
            # cap). Structural failures (other errors) stay cached so we don't retry the unfixable.
            if rec.get("action") == "failed" and "Timeout" in str(rec.get("error", "")):
                continue
            key = (str(rec.get("dataset")), str(rec.get("split")),
                   str(rec.get("baseline")), int(rec.get("seed")))
            reuse[key] = rec                           # later rows overwrite earlier (checkpoint dedup)
        print(f"[reuse] loaded {len(reuse)} cached rows from {args.reuse}", flush=True)

    # incremental checkpoint: append every completed row to results_checkpoint.jsonl as it lands, so a
    # crash/SIGHUP loses at most the in-flight job. Resume with --reuse <that .jsonl>.
    ckpt_path = Path(args.outdir) / args.cluster / "results_checkpoint.jsonl"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_lock = threading.Lock()

    def _checkpoint(row: dict) -> dict:
        with ckpt_lock:
            with open(ckpt_path, "a") as fh:
                fh.write(json.dumps({k: (None if (isinstance(v, float) and pd.isna(v)) else v)
                                     for k, v in row.items()}, default=str) + "\n")
        return row

    def run_on(cs, ds_name, modality):
        programs = spec.program_sets(cs)              # dataset-aware multi-program sets (Supp S3)
        sps = spec.splits(cs)
        if args.test:                                 # de-risk: first K units only (default 0 = all)
            sps = sps[:args.test]
        ctx["splits"] = sps
        print(f"[{args.cluster}] {ds_name} ({modality or 'single'}): {cs.n_cells} cells x {cs.n_genes} genes",
              flush=True)
        bsel = [B for B in spec.baselines
                if not only_set or getattr(B, "name", getattr(B, "__name__", "")) in only_set]
        jobs = [(sp, sp.held_values if spec.downstream_only else None, seed, B)
                for sp in sps for seed in args.seeds for B in bsel]

        def run1(sp, excl, seed, B, gpu=None):
            ck = (str(ds_name), str(sp.name), str(getattr(B, "name", B.__name__)), int(seed))
            if ck in reuse:                           # validated row from a prior sweep — reuse verbatim
                return _checkpoint(dict(reuse[ck]))
            adapter = B()
            if gpu is not None:
                adapter.cuda_device = gpu             # SubprocessAdapter pins CUDA_VISIBLE_DEVICES
            r = run_job(cs, sp, adapter, seed=seed, immune_programs=programs, exclude_genes=excl,
                        response_gene_fn=spec.extra.get("response_gene_fn"),  # C2: training-only panel
                        adapted_implemented=True,      # scGen/CPA adapted runners are implemented
                        # per-dataset bundle key ONLY for multi-dataset clusters (C3), where one split name
                        # is reused across datasets and would collide; single-dataset clusters keep the
                        # deposited (suffix-free) filenames so a re-dump overwrites in place.
                        dataset=(ds_name if spec.datasets else None))
            r.update(cluster=args.cluster, dataset=ds_name, modality=modality)
            return _checkpoint(r)                     # persist immediately (crash-safe)

        # heavy = subprocess/GPU adapters (have a `runner`); simple = in-process & fast
        heavy = [j for j in jobs if getattr(j[3], "runner", "")]
        simple = [j for j in jobs if not getattr(j[3], "runner", "")]
        for sp, excl, seed, B in simple:              # simple inline (CPU, fast)
            rows.append(run1(sp, excl, seed, B))
        if not gpus or not heavy:
            for sp, excl, seed, B in heavy:           # serial heavy
                rows.append(run1(sp, excl, seed, B))
        else:                                         # parallel heavy: one job per GPU concurrently
            gq = queue.Queue()
            for g in gpus:
                gq.put(g)
            tl = threading.local()
            def _hrun(job):
                if not hasattr(tl, "gpu"):
                    tl.gpu = gq.get()                 # pin this worker thread to one GPU
                sp, excl, seed, B = job
                return run1(sp, excl, seed, B, gpu=tl.gpu)
            with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
                for r in ex.map(_hrun, heavy):
                    rows.append(r)

    if spec.datasets:                       # multi-dataset cluster (C3): per-dataset, modality-tagged
        for ds_name, ds in spec.datasets.items():
            run_on(ds["loader"](), ds_name, ds["modality"])
        data_source = "multi: " + ", ".join(spec.datasets)
    else:
        cs = spec.load(args.real)
        run_on(cs, cs.uns.get("dataset", "unknown"), "")
        data_source = cs.uns.get("dataset", "unknown")

    df = pd.DataFrame(rows)
    keys = ["cluster", "dataset", "modality", "split", "baseline", "family", "action",
            "headline_eligible", "registry_task"]
    metric_cols = [c for c in METRICS if c in df.columns] + \
                  [c for c in df.columns if c.startswith("aucell::")]  # per-program AUCell-Δ
    agg = df[df["ran"]].groupby(keys, as_index=False)[metric_cols].mean()
    agg["ran"] = True

    out = Path(args.outdir) / args.cluster
    shared = Path(args.outdir) / "_shared"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results_raw.csv", index=False)
    agg.to_csv(out / "results.csv", index=False)
    splits = ctx["splits"]
    mpath = write_manifest(args.cluster, out, data_source=data_source, seeds=args.seeds,
                           splits=[{"name": s.name, "registry_task": s.registry_task} for s in splits],
                           rows=rows)
    manifest = json.loads(mpath.read_text())

    make_fig = {"C1": c1_figure, "C3": c3_figure, "C4": c4_figure, "C5": c5_figure}.get(args.cluster, cluster_figure)
    fig = make_fig(df, args.cluster, out / "figure.png")
    draft = write_draft(args.cluster, agg, data_source, out / f"draft_{args.cluster}.md")
    cmeth, smeth = write_methods(args.cluster, df, splits, manifest,
                                 out / f"supp_methods_{args.cluster}.md",
                                 shared / "methods_framework.md")
    docx = build_docx(args.cluster, draft.read_text(), out / "figure.png", out / f"draft_{args.cluster}.docx")
    build_docx(args.cluster, cmeth.read_text(), None, out / f"supp_methods_{args.cluster}.docx")
    build_docx(args.cluster, smeth.read_text(), None, shared / "methods_framework.docx")

    print(f"\n=== {args.cluster} cycle complete ({len(args.seeds)} seeds) — data: {data_source} ===")
    show = [c for c in ["dataset", "split", "baseline"] if c in agg.columns] + METRICS
    print(agg[show].round(3).to_string(index=False))
    print(f"\nresults  : {out}/  ->  results.csv  manifest.json  {fig.name} (+.pdf)  {draft.name}  {docx.name}")
    print(f"methods  : {out}/{cmeth.name} (+.docx)   |   shared: {shared}/methods_framework.md (+.docx)")


if __name__ == "__main__":
    main()
