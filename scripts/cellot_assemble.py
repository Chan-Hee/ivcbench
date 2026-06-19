#!/usr/bin/env python
"""Assemble the final CellOT output files from the Kang/Soskic raw per-unit CSVs.

Writes:
  outputs/additional_models/cellot_kang_by_lineage.csv
  outputs/additional_models/cellot_soskic_by_donor.csv
  outputs/additional_models/cellot_summary.csv
Cluster-bootstraps over the biological unit (lineage / donor); %positive; Wilcoxon signed-rank.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/additional_models"
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.metrics.stats import bootstrap_ci

KANG_RAW = OUT / "cellot_kang_raw.csv"
SOSKIC_RAW = OUT / "cellot_soskic_raw.csv"


def summarize(df, unit_col, dataset, split, applicability, seeds, downsampling_rule,
              integration_status, ds_metric_dir):
    """ds_metric_dir: {metric: +1 higher-better, -1 lower-better-already-oriented}. delta_vs_primary in
    the raw is ALWAYS oriented so positive favours cellot."""
    out = []
    for metric in df.metric.unique():
        sub = df[df.metric == metric].copy()
        sub = sub[sub["delta_vs_primary"].astype(str) != ""]
        if not len(sub):
            continue
        delta = sub["delta_vs_primary"].astype(float).to_numpy()
        cscore = pd.to_numeric(sub["cellot_score"], errors="coerce").to_numpy()
        bscore = pd.to_numeric(sub["baseline_score"], errors="coerce").to_numpy()
        ci = bootstrap_ci(delta, n_boot=5000, seed=0)
        pct_pos = 100 * float(np.mean(delta > 0))
        try:
            w = stats.wilcoxon(delta) if len(delta) >= 6 and np.any(delta != 0) else None
            pval = float(w.pvalue) if w is not None else float("nan")
        except Exception:
            pval = float("nan")
        verdict = ("CellOT beats primary baseline" if ci["lo"] > 0 else
                   "CellOT below primary baseline" if ci["hi"] < 0 else
                   "CellOT not distinguishable from primary baseline")
        out.append(dict(
            model="CellOT", dataset=dataset, split=split, applicability=applicability,
            biological_unit=unit_col, n_units=int(len(delta)), metric=metric,
            baseline_score=round(float(np.nanmean(bscore)), 4),
            model_score=round(float(np.nanmean(cscore)), 4),
            delta_vs_primary_baseline=round(ci["mean"], 4),
            CI_low=round(ci["lo"], 4), CI_high=round(ci["hi"], 4),
            percent_positive_units=round(pct_pos, 1),
            p_value_if_applicable=(round(pval, 4) if pval == pval else ""),
            seeds=seeds, downsampling_rule=downsampling_rule, verdict=verdict,
            integration_status=integration_status,
        ))
    return out


def main():
    summary_rows = []

    # ---- Kang A1 ----
    if KANG_RAW.exists():
        k = pd.read_csv(KANG_RAW)
        k.to_csv(OUT / "cellot_kang_by_lineage.csv", index=False)
        summary_rows += summarize(
            k, "lineage", "Kang IFN-beta PBMC", "leave-one-lineage-out (LOLO, 8 coarse lineages)",
            "native / task-matched optimal-transport stimulation transfer",
            seeds="0,1,2", downsampling_rule="kang loader subsample_per_group=80 (stim x celltype x donor)",
            integration_status="main", ds_metric_dir={})
        print(f"Kang: {len(k)} rows -> cellot_kang_by_lineage.csv")

    # ---- Soskic A2 ----
    if SOSKIC_RAW.exists():
        s = pd.read_csv(SOSKIC_RAW)
        s.to_csv(OUT / "cellot_soskic_by_donor.csv", index=False)
        n_don = s["donor"].nunique()
        status = "supplementary" if n_don >= 100 else "exploratory"
        dr = "load_soskic_donor cap=300 per donor x condition x celltype" + (
            "" if n_don >= 100 else f"; EXPLORATORY first-{n_don}-donor subset")
        summary_rows += summarize(
            s, "donor", "Soskic CD4 activation", f"leave-one-donor-out (LODO, n={n_don})",
            "adapted optimal-transport activation transfer",
            seeds=",".join(sorted(set(",".join(s["seeds"].astype(str)).split(",")))),
            downsampling_rule=dr, integration_status=status, ds_metric_dir={})
        print(f"Soskic: {len(s)} rows, {n_don} donors -> cellot_soskic_by_donor.csv")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(OUT / "cellot_summary.csv", index=False)
        print(f"\nWROTE cellot_summary.csv ({len(summary_rows)} rows)")
        print(pd.DataFrame(summary_rows)[["dataset", "metric", "baseline_score", "model_score",
              "delta_vs_primary_baseline", "CI_low", "CI_high", "percent_positive_units",
              "verdict", "integration_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
