"""Aggregate completed raw runs by split, method and runtime applicability.

The compatibility function name predates the final supplement. This table is
a run diagnostic; publication tables come from the curated census instead.
"""

from __future__ import annotations

import pandas as pd

METRIC_COLS = ["pearson_delta", "e_distance", "aucell_program_corr"]


def results_to_supp_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([r for r in rows if r.get("ran")])
    if df.empty:
        return df
    # average over seeds per (split, baseline)
    agg = (
        df.groupby(["split", "baseline", "family", "action", "headline_eligible"])[
            METRIC_COLS
        ]
        .mean()
        .reset_index()
    )
    return agg.sort_values(
        ["split", "headline_eligible", "pearson_delta"], ascending=[True, False, False]
    )
