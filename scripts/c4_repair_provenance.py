#!/usr/bin/env python
"""C4 provenance repair (reviewer reqs 2/6/14): the deposited results/C4/conditioned_rows.json had the
second conditioned family `linear-shift-KOemb` as action='failed' (KeyError: not in the applicability
registry), even though the standalone verify log reproduced its leak-safe numbers. We now (a) the name
is registered in baselines/registry.py (C4_Axis1/Axis2 -> APPLICABLE), so (b) re-run ONLY
linear-shift-KOemb through the framework run_job (the SAME path scGen used) so a SUCCESSFUL first-class
row lands; (c) preserve the already-verified scGen seed-0 rows from the existing JSON unchanged; and
(d) write a self-contained results/C4/results_raw.csv that includes both conditioned families alongside
the four simple floors. Pure CPU, fully in-process (linear-shift-KOemb needs no heavy env).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from ivcbench.data.loaders.frangieh import load
from ivcbench.clusters import c4
from ivcbench.runner.run import run_job

# reuse the EXACT adapter the original conditioned-run script defined
sys.path.insert(0, "scripts")
from run_c4_conditioned import LinearShiftKOEmb  # noqa: E402

ROOT = Path("results/C4")
CR = ROOT / "conditioned_rows.json"


def main():
    existing = json.loads(CR.read_text())
    # keep every row that is NOT a (now-repaired) failed linear-shift-KOemb row
    preserved = [r for r in existing
                 if not (r.get("baseline") == "linear-shift-KOemb" and r.get("action") == "failed")]

    new_rows = []
    for modality, mod_tag in [("rna", "RNA"), ("protein", "protein-CITE")]:
        cs = load(modality=modality)
        g = cs.uns["genes_perturbed"]
        ds_name = cs.uns.get("dataset", f"frangieh_{modality}")
        for frac, lbl in [(0.25, "25"), (0.50, "50")]:
            held = c4.held_ko_fraction(g, frac, seed=0)
            spec = c4.modality_lo_ko(held, lbl)
            excl = list(spec.held_values)  # downstream-only: exclude held-KO genes
            t0 = time.time()
            r = run_job(cs, spec, LinearShiftKOEmb(), seed=0, exclude_genes=excl,
                        adapted_implemented=True)
            r.update(cluster="C4", dataset=ds_name, modality=mod_tag,
                     elapsed_s=round(time.time() - t0, 1))
            new_rows.append(r)
            print(json.dumps({k: r.get(k) for k in
                              ("baseline", "modality", "split", "action", "ran", "leak_free",
                               "headline_eligible", "n_train", "n_test", "pearson_delta",
                               "pearson_delta_ontarget", "pearson_delta_lo", "pearson_delta_hi",
                               "e_distance", "elapsed_s", "error")}), flush=True)

    merged = preserved + new_rows
    CR.write_text(json.dumps(merged, indent=2, default=str))
    print(f"WROTE {CR} ({len(merged)} rows; {len(new_rows)} repaired linear-shift-KOemb)")

    # --- self-contained results_raw.csv: the 4 simple floors + the two conditioned families ---
    raw = pd.read_csv(ROOT / "results_raw.csv")
    cond_df = pd.DataFrame([r for r in merged if r.get("ran")])
    # align columns to raw schema; keep extra cols out
    cols = list(raw.columns)
    for c in cols:
        if c not in cond_df.columns:
            cond_df[c] = pd.NA
    cond_df = cond_df[cols]
    # drop any pre-existing conditioned rows in raw (idempotent), then append
    raw_simple = raw[~raw["baseline"].isin(["scGen", "linear-shift-KOemb"])]
    out = pd.concat([raw_simple, cond_df], ignore_index=True)
    out.to_csv(ROOT / "results_raw.csv", index=False)
    print(f"WROTE {ROOT/'results_raw.csv'} now self-contained: {len(out)} rows "
          f"({len(raw_simple)} simple + {len(cond_df)} conditioned)")
    # echo the conditioned block for the log
    print(out[out["baseline"].isin(["scGen", "linear-shift-KOemb"])]
          [["baseline", "modality", "split", "leak_free", "pearson_delta",
            "pearson_delta_ontarget", "e_distance", "headline_eligible"]].to_string())


if __name__ == "__main__":
    main()
