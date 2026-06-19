#!/usr/bin/env python
"""Assemble the final CellOT deliverables from the completed raw per-unit CSVs.

The CellOT runner wrote per-unit raw CSVs (cellot_kang_raw.csv = 8 lineages, cellot_soskic_raw.csv =
20-donor exploratory subset) but its driver turn ended before assembling the summary. This finalizes:
  cellot_kang_by_lineage.csv, cellot_soskic_by_donor.csv, cellot_summary.csv
with a cluster bootstrap over the biological unit (lineages / donors), percent positive, and Wilcoxon.
No values are recomputed from cells; this only aggregates the already-computed leak-safe per-unit deltas.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT = Path(__file__).resolve().parents[1] / "outputs" / "additional_models"

def cluster_ci(vals, n=10000, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    boot = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

rows = []
def summarize(raw_csv, unit_col, dataset, split, applicability, integration, downsampling, wilcox):
    df = pd.read_csv(raw_csv)
    assert df["leak_free"].astype(str).str.lower().eq("true").all(), f"LEAK in {raw_csv}!"
    out_by_unit = df.copy()
    for metric in df["metric"].unique():
        sub = df[df.metric == metric]
        d = sub["delta_vs_primary"].to_numpy(float)
        mean, lo, hi = cluster_ci(d)
        pos = float((d > 0).mean() * 100)
        p = None
        if wilcox and len(d) >= 6 and np.any(d != 0):
            try:
                p = float(wilcoxon(d).pvalue)
            except Exception:
                p = None
        # verdict per metric (pearson is the headline)
        verdict = ("exceeds matched baseline" if lo > 0 else
                   "below matched baseline" if hi < 0 else
                   "competitive (CI includes 0)")
        rows.append(dict(model="CellOT", dataset=dataset, split=split, applicability=applicability,
                         biological_unit=unit_col, n_units=int(sub[unit_col].nunique()), metric=metric,
                         baseline_score=round(float(sub["baseline_score"].mean()), 4),
                         model_score=round(float(sub["cellot_score"].mean()), 4),
                         delta_vs_primary_baseline=round(mean, 4), CI_low=round(lo, 4), CI_high=round(hi, 4),
                         percent_positive_units=round(pos, 1),
                         p_value_if_applicable=None if p is None else round(p, 4),
                         seeds=str(sub["seeds"].iloc[0]), downsampling_rule=downsampling,
                         verdict=verdict, integration_status=integration))
    return out_by_unit

kang = summarize(OUT / "cellot_kang_raw.csv", "lineage", "kang_GSE96583",
                 "C1_leave_one_lineage_out", "native (task-matched OT stimulation transfer)",
                 "main", "none (full 8 lineages, seeds 0/1/2 collapsed within lineage)", wilcox=True)
kang.to_csv(OUT / "cellot_kang_by_lineage.csv", index=False)

_sos_raw = pd.read_csv(OUT / "cellot_soskic_raw.csv")
_n = _sos_raw["donor"].nunique()
_seeds_str = str(_sos_raw["seeds"].iloc[0])
_seed_desc = "seed 0" if _seeds_str == "0" else f"seeds {_seeds_str} collapsed within donor"
_status = "main" if _n >= 106 else "exploratory"
_downsamp = (f"none (full {_n}-donor leave-one-donor-out, {_seed_desc})" if _n >= 106
             else f"pre-specified first-{_n}-donor subset (exploratory); {_seed_desc}")
sos = summarize(OUT / "cellot_soskic_raw.csv", "donor", "soskic2022",
                "C2_leave_one_donor_out", "adapted (OT activation transfer)",
                _status, _downsamp, wilcox=(_n >= 30))
sos.to_csv(OUT / "cellot_soskic_by_donor.csv", index=False)

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "cellot_summary.csv", index=False)

print("=== CellOT summary (headline = pearson_delta) ===")
print(summary[["dataset", "metric", "n_units", "baseline_score", "model_score",
               "delta_vs_primary_baseline", "CI_low", "CI_high", "percent_positive_units",
               "p_value_if_applicable", "verdict", "integration_status"]].to_string(index=False))
