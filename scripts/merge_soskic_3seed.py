#!/usr/bin/env python
"""Build the 3-seed CellOT Soskic raw from seed-0 raw + the seeds-1/2 chunk CSVs.

Seed-0 lives in cellot_soskic_raw.csv (seed_scores=[v0]); seeds 1,2 are in cellot_soskic_s12_chunk*.csv
(seed_scores=[v1,v2]). Per (donor, metric) we form the 3-seed mean cellot_score = mean(v0,v1,v2) and
recompute delta_vs_primary against the SAME deterministic baseline (cross-checked equal across runs).
This matches how Kang collapses seeds 0/1/2 within unit before the donor bootstrap. Seed-0 raw is NOT
overwritten; output is cellot_soskic_raw_3seed.csv plus a seed-dispersion table (mean +/- SD across the
three per-seed donor-aggregated deltas).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "additional_models"


def parse_scores(s):
    return [float(x) for x in json.loads(s)]


def main():
    seed0 = pd.read_csv(OUT / "cellot_soskic_raw.csv")
    assert seed0["seeds"].astype(str).eq("0").all(), "cellot_soskic_raw.csv is not seed-0-only!"
    chunks = sorted(OUT.glob("cellot_soskic_s12_chunk*.csv"))
    assert chunks, "no seeds-1/2 chunk files found"
    s12 = pd.concat([pd.read_csv(c) for c in chunks], ignore_index=True)
    s12 = s12.drop_duplicates(subset=["donor", "metric"], keep="first").reset_index(drop=True)

    # index seed-0 and seeds-1/2 by (donor, metric)
    s0 = {(r.donor, r.metric): r for r in seed0.itertuples(index=False)}
    s1 = {(r.donor, r.metric): r for r in s12.itertuples(index=False)}

    rows, dispersion, baseline_mismatch = [], [], []
    donors0 = set(d for d, _ in s0)
    donors12 = set(d for d, _ in s1)
    common = sorted(donors0 & donors12)
    print(f"donors: seed0={len(donors0)} seeds12={len(donors12)} common={len(common)}")
    if donors0 - donors12:
        print(f"  WARNING missing in seeds12: {sorted(donors0 - donors12)}")

    metrics = seed0["metric"].unique().tolist()
    for donor in common:
        for metric in metrics:
            r0, r12 = s0.get((donor, metric)), s1.get((donor, metric))
            if r0 is None or r12 is None:
                continue
            v0 = parse_scores(r0.seed_scores)[0]
            v12 = parse_scores(r12.seed_scores)            # [v1, v2]
            vals = [v0] + v12
            # deterministic baseline must agree across runs
            if abs(float(r0.baseline_score) - float(r12.baseline_score)) > 1e-4:
                baseline_mismatch.append((donor, metric, float(r0.baseline_score), float(r12.baseline_score)))
            base = float(r0.baseline_score)
            cellot3 = float(np.mean(vals))
            # orientation must match cellot_soskic.py: e_distance is lower-better, so a positive
            # gap is (baseline - model); pearson_delta and aucell_delta_score are higher-better.
            oriented = (base - cellot3) if metric == "e_distance" else (cellot3 - base)
            rows.append(dict(
                donor=donor, metric=metric, cellot_score=round(cellot3, 4),
                primary_baseline=r0.primary_baseline, baseline_score=round(base, 4),
                delta_vs_primary=round(oriented, 4),
                seed_scores=json.dumps([round(x, 4) for x in vals]), seeds="0,1,2",
                n_test=r0.n_test, n_ctrl=r0.n_ctrl, n_strata=r0.n_strata,
                n_response_genes=r0.n_response_genes,
                best_mmd=round(float(np.mean([float(r0.best_mmd), float(r12.best_mmd)])), 5),
                leak_free=bool(str(r0.leak_free).lower() == "true" and str(r12.leak_free).lower() == "true"),
            ))
            if metric == "pearson_delta":
                for si, sv in zip([0, 1, 2], vals):
                    dispersion.append(dict(donor=donor, seed=si, delta=sv - base))

    out = pd.DataFrame(rows)
    leak_all = out["leak_free"].all()
    out.to_csv(OUT / "cellot_soskic_raw_3seed.csv", index=False)
    print(f"\nwrote cellot_soskic_raw_3seed.csv: {len(out)} rows, {out.donor.nunique()} donors, leak_free_all={leak_all}")
    if baseline_mismatch:
        print(f"  !! baseline mismatch on {len(baseline_mismatch)} (donor,metric) -- first 5: {baseline_mismatch[:5]}")
    else:
        print("  baseline cross-check: all seed-0 vs seeds-1/2 baselines agree (deterministic OK)")

    # headline (3-seed pearson_delta), to be confirmed by cellot_finalize
    p = out[out.metric == "pearson_delta"]
    print(f"  3-seed pearson_delta: mean delta={p.delta_vs_primary.mean():+.4f}, "
          f"%positive={100*(p.delta_vs_primary>0).mean():.1f}%, n={len(p)}")

    # seed dispersion: per-seed donor-aggregated mean delta -> mean +/- SD across seeds
    disp = pd.DataFrame(dispersion)
    per_seed = disp.groupby("seed")["delta"].mean()
    seed_mean, seed_sd = float(per_seed.mean()), float(per_seed.std(ddof=1))
    summary = dict(per_seed_mean_delta={int(k): round(float(v), 4) for k, v in per_seed.items()},
                   across_seed_mean=round(seed_mean, 4), across_seed_sd=round(seed_sd, 4))
    (OUT / "cellot_soskic_seed_dispersion.json").write_text(json.dumps(summary, indent=2))
    print(f"  per-seed aggregate delta: {summary['per_seed_mean_delta']}")
    print(f"  across-seed mean +/- SD: {seed_mean:+.4f} +/- {seed_sd:.4f}")


if __name__ == "__main__":
    main()
