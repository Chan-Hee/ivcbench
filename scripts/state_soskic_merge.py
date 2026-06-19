#!/usr/bin/env python
"""Merge the 2 STATE-Soskic GPU shards into the full per-donor raw CSV.

Each shard ran a disjoint donor stride (--chunk 0 2 on GPU0, --chunk 1 2 on GPU1), so the union is the
106 donors with no overlap; dedup by (donor, metric) defensively. Verifies leak_free and reports the
anchor-gate summary (STATE pearson_delta must be finite and in the scGen/CellOT donor band, not >0.9).
Writes outputs/additional_models/state_soskic_raw.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "additional_models"
shards = sorted(OUT.glob("state_soskic_shard[0-9].csv"))
if not shards:
    raise SystemExit(f"no shards found under {OUT} (expected state_soskic_shard*.csv)")
df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
df = df.drop_duplicates(subset=["donor", "metric"], keep="first").reset_index(drop=True)
n_donor = df["donor"].nunique()
leak = df["leak_free"].astype(str).str.lower().eq("true").all()
df.to_csv(OUT / "state_soskic_raw.csv", index=False)
print(f"merged {len(shards)} shards -> state_soskic_raw.csv : {n_donor} unique donors, "
      f"{len(df)} rows, leak_free_all={leak}")

p = df[df.metric == "pearson_delta"].copy()
p = p[p.state_score != ""]
sc = p.state_score.astype(float)
dvp = p[p.delta_vs_primary != ""].delta_vs_primary.astype(float)
finite = bool(np.isfinite(sc).all())
inflated = bool((sc > 0.9).any())
in_band = float(sc.between(-0.10, 0.50).mean())
print(f"  pearson_delta: mean state_score={sc.mean():+.4f} (min {sc.min():+.4f}, max {sc.max():+.4f}); "
      f"mean delta_vs_primary={dvp.mean():+.4f}; %positive={100*(dvp>0).mean():.1f}%; n={len(p)}")
print(f"  ANCHOR GATE: finite={finite}; leak_inflated(>0.9)={inflated}; "
      f"%in_band[-0.10,0.50]={100*in_band:.0f}%  -> "
      f"{'PASS' if (finite and not inflated and in_band >= 0.8) else 'REVIEW'}")
