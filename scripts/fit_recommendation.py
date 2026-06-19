#!/usr/bin/env python
"""Mechanical implementation of PRE-REGISTRATION rule (5): the DESCRIPTIVE / EXPLORATORY
fit-recommendation matrix.

This is NOT a hypothesis test. It is a transparent, mechanical reading of a results table that, for each
(task, model family), reports whether the family "works" under the pre-registered rule:

    A family WORKS on a task iff at least one of its models exceeds the UNIVERSAL SIMPLE FLOOR
    {cell-mean shift, linear-PCA shift} with a cluster-bootstrap CI_low > 0 on the headline
    response-direction metric for that task. Ties are broken toward simplicity (indistinguishable ->
    recommend the simple floor).

Input : a results table (CSV) with at least the columns
            cluster, baseline, family, split, ran, leak_free,
            <response-direction metric> and its bootstrap CI columns,
        and the biological unit needed for the cluster bootstrap (lineage / dataset / donor / marker),
        carried in the table or supplied via --unit-col. If a per-unit value column is present we do a
        true biological-unit cluster bootstrap of the family-minus-floor gap; if only the pre-computed
        per-row CI (<metric>_lo/_hi) is available we fall back to it and say so in `ci_source`.
Output: the descriptive fit matrix as CSV (and JSON), one row per (cluster, split, family):
            point gap vs the universal floor, cluster-bootstrap CI, works flag, recommendation
            (family name if it works, else "simple-floor"), and ci_source.

No hardcoded numbers, no per-cluster special-casing of the verdict: the rule is applied identically to
every (task, family). Governed by benchmark/PREREGISTRATION.md section (5). Deterministic (seeded).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- pre-registered constants (PREREGISTRATION.md sections 2, 5) --------------------------------------
UNIVERSAL_FLOOR = ["cell-mean", "linear-PCA"]          # section (2): THE universal simple floor
CONTEXT_BASELINES = ["FP-ridge", "donor-shift"]        # context only, never the headline floor
SANITY_BASELINES = ["ctrl-pred"]                       # degeneracy check, not a floor
NON_CONDITIONED_FAMILIES = {"simple", "ot"}            # 'ot' floors excluded when asking "does conditioning help"

# default headline response-direction metric + its CI columns, and the CRISPR downstream-only variant
DEFAULT_METRIC = "pearson_delta"
DEFAULT_METRIC_ONTARGET = "pearson_delta_ontarget"     # CRISPR: on-target perturbed gene excluded
# clusters for which the downstream-only (on-target-excluded) variant is the headline metric
DOWNSTREAM_ONLY_CLUSTERS = {"C3", "C4"}

B_DEFAULT = 10000
SEED_DEFAULT = 0

# candidate columns naming the biological resampling unit, in preference order, per the seed/CI policy
UNIT_COL_CANDIDATES = ["unit", "lineage", "cell_type_fine", "cell_type_coarse", "dataset", "donor_id",
                       "donor", "marker"]


def _eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that actually ran leak-free (the maximal usable submatrix)."""
    out = df.copy()
    if "ran" in out.columns:
        out = out[out["ran"] == True]            # noqa: E712
    if "leak_free" in out.columns:
        out = out[out["leak_free"] == True]      # noqa: E712
    return out


def _headline_metric(cluster: str, df: pd.DataFrame, metric: str | None) -> str:
    """Resolve which response-direction column is the headline metric for this cluster."""
    if metric is not None:
        return metric
    if cluster in DOWNSTREAM_ONLY_CLUSTERS and DEFAULT_METRIC_ONTARGET in df.columns:
        return DEFAULT_METRIC_ONTARGET
    return DEFAULT_METRIC


def _pick_unit_col(df: pd.DataFrame, override: str | None) -> str | None:
    if override:
        return override if override in df.columns else None
    for c in UNIT_COL_CANDIDATES:
        if c in df.columns and df[c].nunique(dropna=True) > 1:
            return c
    return None


def _floor_value(rows: pd.DataFrame, metric: str) -> float:
    """Best of the universal simple floor on this (cluster, split) for the given metric."""
    f = rows[rows["baseline"].isin(UNIVERSAL_FLOOR)]
    return float(f[metric].max()) if len(f) else np.nan


def _cluster_bootstrap_gap(rows: pd.DataFrame, fam_baselines: list[str], metric: str,
                           unit_col: str, rng: np.random.Generator, B: int):
    """True biological-unit cluster bootstrap of (best-family-model - best-floor) gap.

    For each bootstrap draw, resample the units with replacement; within the resampled units recompute
    the family's best per-unit metric and the floor's best per-unit metric, take the mean over units of
    (family - floor). Returns (point_gap, ci_lo, ci_hi, frac_pos).
    """
    units = rows[unit_col].dropna().unique().tolist()
    if len(units) < 2:
        return (np.nan, np.nan, np.nan, np.nan)

    def per_unit_gap(unit_list):
        gaps = []
        for u in unit_list:
            sub = rows[rows[unit_col] == u]
            fam = sub[sub["baseline"].isin(fam_baselines)][metric]
            flr = sub[sub["baseline"].isin(UNIVERSAL_FLOOR)][metric]
            if len(fam) and len(flr):
                gaps.append(float(fam.max()) - float(flr.max()))
        return gaps

    point_gaps = per_unit_gap(units)
    if not point_gaps:
        return (np.nan, np.nan, np.nan, np.nan)
    point = float(np.mean(point_gaps))
    n = len(units)
    draws = []
    for _ in range(B):
        pick = [units[i] for i in rng.integers(0, n, n)]
        g = per_unit_gap(pick)
        if g:
            draws.append(np.mean(g))
    if not draws:
        return (point, np.nan, np.nan, np.nan)
    draws = np.asarray(draws, float)
    return (point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)),
            float((draws > 0).mean()))


def _per_row_ci_gap(rows: pd.DataFrame, fam_baselines: list[str], metric: str):
    """Fallback when no biological unit is available: use the pre-computed per-row bootstrap CI of the
    best family model and subtract the floor point. CI_low of the gap is approximated by
    (best-family CI_low - floor point). Flagged as ci_source='per_row_ci' so it is never mistaken for the
    true cluster bootstrap."""
    lo_col, hi_col = f"{metric}_lo", f"{metric}_hi"
    fam = rows[rows["baseline"].isin(fam_baselines)]
    if not len(fam) or lo_col not in rows.columns:
        return (np.nan, np.nan, np.nan, np.nan)
    best_idx = fam[metric].idxmax()
    fam_pt = float(fam.loc[best_idx, metric])
    fam_lo = float(fam.loc[best_idx, lo_col])
    fam_hi = float(fam.loc[best_idx, hi_col])
    floor = _floor_value(rows, metric)
    gap = fam_pt - floor
    return (gap, fam_lo - floor, fam_hi - floor, np.nan)


def fit_matrix(df: pd.DataFrame, metric: str | None = None, unit_col_override: str | None = None,
               B: int = B_DEFAULT, seed: int = SEED_DEFAULT,
               include_context: bool = False) -> pd.DataFrame:
    """Apply the pre-registered descriptive fit-recommendation rule to every (cluster, split, family)."""
    rng = np.random.default_rng(seed)
    df = _eligible(df)
    if "cluster" not in df.columns:
        raise ValueError("results table must have a 'cluster' column")
    split_col = "split" if "split" in df.columns else None

    records = []
    for cluster, cdf in df.groupby("cluster"):
        m = _headline_metric(str(cluster), cdf, metric)
        if m not in cdf.columns:
            continue
        unit_col = _pick_unit_col(cdf, unit_col_override)
        splits = cdf[split_col].unique() if split_col else [None]
        for sp in splits:
            rows = cdf if sp is None else cdf[cdf[split_col] == sp]
            floor = _floor_value(rows, m)
            # families to evaluate: every model family except the universal-simple/ot floors,
            # unless include_context (then also score the context baselines as their own pseudo-family)
            fam_names = [f for f in rows["family"].dropna().unique()
                         if f not in NON_CONDITIONED_FAMILIES]
            for fam in sorted(fam_names):
                fam_baselines = rows[rows["family"] == fam]["baseline"].unique().tolist()
                # context baselines never count as a conditioned family's member in the headline decision
                fam_baselines = [b for b in fam_baselines
                                 if b not in CONTEXT_BASELINES + SANITY_BASELINES]
                if not fam_baselines:
                    continue
                if unit_col is not None:
                    gap, lo, hi, fpos = _cluster_bootstrap_gap(rows, fam_baselines, m, unit_col, rng, B)
                    ci_source = f"cluster_bootstrap[{unit_col}]"
                else:
                    gap, lo, hi, fpos = _per_row_ci_gap(rows, fam_baselines, m)
                    ci_source = "per_row_ci"
                works = bool(np.isfinite(gap) and gap > 0 and np.isfinite(lo) and lo > 0)
                records.append(dict(
                    cluster=cluster, split=sp, family=fam,
                    metric=m, floor=floor,
                    best_model_gap=gap, ci_low=lo, ci_high=hi, boot_pos_frac=fpos,
                    works=works,
                    recommendation=(fam if works else "simple-floor"),  # tie -> simplicity
                    ci_source=ci_source,
                    n_floor_baselines=int(rows["baseline"].isin(UNIVERSAL_FLOOR).sum()),
                    family_baselines=";".join(sorted(fam_baselines)),
                ))
    out = pd.DataFrame.from_records(records)
    if include_context and len(out):
        out.attrs["note"] = ("context baselines (FP-ridge/donor-shift) are reported elsewhere as context "
                             "only and are excluded from this headline fit decision by design")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="path to a results table (CSV) governed by PREREGISTRATION.md")
    ap.add_argument("-o", "--out", default=None, help="output CSV path (default: <results>.fit_matrix.csv)")
    ap.add_argument("--metric", default=None,
                    help="override the response-direction column (default: pearson_delta, "
                         "or pearson_delta_ontarget for downstream-only CRISPR clusters)")
    ap.add_argument("--unit-col", default=None,
                    help="column naming the biological resampling unit (lineage/dataset/donor/marker); "
                         "auto-detected if omitted")
    ap.add_argument("-B", "--bootstrap", type=int, default=B_DEFAULT, help="bootstrap draws")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT, help="RNG seed (deterministic CIs)")
    args = ap.parse_args(argv)

    path = Path(args.results)
    if not path.exists():
        ap.error(f"results table not found: {path}")
    df = pd.read_csv(path)

    fm = fit_matrix(df, metric=args.metric, unit_col_override=args.unit_col,
                    B=args.bootstrap, seed=args.seed)
    if not len(fm):
        print("no eligible (cluster, split, family) rows found in the results table", file=sys.stderr)

    out_csv = Path(args.out) if args.out else path.with_suffix(".fit_matrix.csv")
    fm.to_csv(out_csv, index=False)
    fm.to_json(out_csv.with_suffix(".json"), orient="records", indent=2)

    # human-readable summary to stdout
    print(f"# DESCRIPTIVE / EXPLORATORY fit matrix (PREREGISTRATION.md rule 5) -> {out_csv}")
    if len(fm):
        cols = ["cluster", "split", "family", "best_model_gap", "ci_low", "works",
                "recommendation", "ci_source"]
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(fm[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
