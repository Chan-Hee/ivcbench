#!/usr/bin/env python
"""Multi-seed scGen replication (reviewer reqs 1/7/15): the entire benchmark is seed-0, and the only
stochastic conditioning WIN (scGen, C1 CD14+ monocyte LOCT, +0.098) plus the C4 conditioned leg are
single-seed. We replicate scGen across training seeds {0,1,2} to attach seed-variance CIs and resolve
whether the +0.098 can sign-flip.

scGen runs CPU in the scperturbench_eval conda env (the heavy adapter shells out). The runners now
plumb IVCBENCH_SEED -> scvi.settings.seed (1-line change). We re-run through the SAME framework run_job
path (leak audit + pearson_delta) so each seed's row is leak-audited identically to the deposited
seed-0 row.

C1: ScGenC1 on C1_loct_Mono_CD14 (the +0.098 win) — and, for context, the two other T-cell wins.
C4: ScGen on the 4 frangieh leave-one-KO modality splits (RNA/protein x 25/50%).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEEDS = [int(s) for s in os.environ.get("IVCBENCH_SEEDS", "0,1,2").split(",")]
os.environ.setdefault("IVCBENCH_SCGEN_EPOCHS", "60")  # C1 deposit default

from ivcbench.runner.run import run_job  # noqa: E402


def run_c1():
    from ivcbench.data.loaders import kang
    from ivcbench.clusters import c1
    from ivcbench.baselines.heavy import ScGenC1
    cs = kang.load()
    progs = None
    rows = []
    lineages = ["Mono_CD14", "CD8T", "CD4T"]  # the 3 scGen LOCT wins; CD14 is the headline +0.098
    for lin in lineages:
        spec = c1.coarse_loct(held_lineage=lin)
        for seed in SEEDS:
            os.environ["IVCBENCH_SEED"] = str(seed)
            t0 = time.time()
            try:
                r = run_job(cs, spec, ScGenC1(), seed=seed)
            except Exception as e:  # noqa: BLE001
                r = {"baseline": "scGen", "split": spec.name, "action": "failed", "ran": False,
                     "error": f"{type(e).__name__}: {str(e)[:300]}"}
            r.update(cluster="C1", lineage=lin, train_seed=seed, elapsed_s=round(time.time() - t0, 1))
            rows.append(r)
            print(f"C1 {lin:10s} seed={seed} ran={r.get('ran')} leak_free={r.get('leak_free')} "
                  f"pearson_delta={r.get('pearson_delta')} ({r.get('elapsed_s')}s)", flush=True)
    return rows


def run_c4():
    from ivcbench.data.loaders.frangieh import load
    from ivcbench.clusters import c4
    from ivcbench.baselines.heavy import ScGen
    os.environ["IVCBENCH_SCGEN_EPOCHS"] = "40"  # C4 conditioned-run default
    rows = []
    for modality, mod_tag in [("rna", "RNA"), ("protein", "protein-CITE")]:
        cs = load(modality=modality)
        g = cs.uns["genes_perturbed"]
        for frac, lbl in [(0.25, "25"), (0.50, "50")]:
            held = c4.held_ko_fraction(g, frac, seed=0)
            spec = c4.modality_lo_ko(held, lbl)
            excl = list(spec.held_values)
            for seed in SEEDS:
                os.environ["IVCBENCH_SEED"] = str(seed)
                t0 = time.time()
                try:
                    r = run_job(cs, spec, ScGen(), seed=seed, exclude_genes=excl,
                                adapted_implemented=True)
                except Exception as e:  # noqa: BLE001
                    r = {"baseline": "scGen", "split": spec.name, "action": "failed", "ran": False,
                         "error": f"{type(e).__name__}: {str(e)[:300]}"}
                r.update(cluster="C4", modality=mod_tag, train_seed=seed,
                         elapsed_s=round(time.time() - t0, 1))
                rows.append(r)
                print(f"C4 {mod_tag:12s} {lbl}% seed={seed} ran={r.get('ran')} "
                      f"leak_free={r.get('leak_free')} pearson_delta={r.get('pearson_delta')} "
                      f"ontarget={r.get('pearson_delta_ontarget')} ({r.get('elapsed_s')}s)", flush=True)
    return rows


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    rows = []
    if which in ("all", "c1"):
        rows += run_c1()
    if which in ("all", "c4"):
        rows += run_c4()
    out = ROOT / "results/_paper/multiseed_scgen.json"
    # merge with any prior partial
    if out.exists():
        try:
            prev = json.loads(out.read_text())
        except Exception:
            prev = []
        rows = prev + rows
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"WROTE {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
