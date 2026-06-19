#!/usr/bin/env python
"""Merge the 4 Soskic CellOT shards into the full per-donor raw CSV.

Each shard was pre-seeded with the same 20 donors, so dedup by (donor, metric). Verifies leak_free and
reports the donor count. Writes outputs/additional_models/cellot_soskic_raw.csv (full union).
"""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "additional_models"
shards = sorted(OUT.glob("cellot_soskic_shard[0-9].csv"))
df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
df = df.drop_duplicates(subset=["donor", "metric"], keep="first").reset_index(drop=True)
n_donor = df["donor"].nunique()
leak = df["leak_free"].astype(str).str.lower().eq("true").all()
df.to_csv(OUT / "cellot_soskic_raw.csv", index=False)
print(f"merged {len(shards)} shards -> cellot_soskic_raw.csv : {n_donor} unique donors, {len(df)} rows, leak_free_all={leak}")
# quick headline (pearson_delta)
p = df[df.metric == "pearson_delta"]
print(f"  pearson_delta: mean delta_vs_primary={p.delta_vs_primary.mean():+.4f}, "
      f"%positive={100*(p.delta_vs_primary>0).mean():.1f}%, n={len(p)}")
