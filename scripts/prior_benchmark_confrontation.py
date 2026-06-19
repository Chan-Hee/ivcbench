#!/usr/bin/env python
"""Empirical prior-benchmark confrontation (reviewer req 8): show that a MIXED-split protocol
(Open Problems style: pool all held-out conditions into one leaderboard, no axis stratification) or a
POOLED E-distance protocol (scPerturb style: rank by overall distributional separation) would CROWN a
conditioned model that our leak-safe AXIS-STRATIFIED protocol DEMOTES.

Pure analysis on deposited results/{C1,C3,C5}/results_raw.csv. No new runs.

Three rankings on the SAME per-split numbers:
  (A) Ours, axis-stratified: within each generalization axis, is the best conditioned model > the
      best simple floor? (cell-context axis vs perturbation axis kept SEPARATE; simple = first-class)
  (B) Open-Problems-style mixed pool: each model's mean Pearson-Δ averaged over ALL splits pooled
      (cell-context + perturbation mixed, all weighted equally), then ranked. Conditioned models that
      shine on the abundant/easy cell-context splits can top this pooled board.
  (C) scPerturb-style pooled E-distance: each model's mean E-distance over all pooled conditions
      (lower = better separation), ranked.
The contrast: the model that the pooled boards crown should be one our stratified protocol shows
LOSING on the perturbation axis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SIMPLE = {"ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"}
# axis tag per split prefix
def axis_of(split: str) -> str:
    s = str(split)
    if "loct" in s:
        return "cell-context"
    if "lo_gene" in s or "compound_holdout" in s or "lo_ko" in s:
        return "perturbation"
    if "lodo" in s:
        return "donor"
    if "randsplit" in s:
        return "donor-randomsplit"
    return "other"


def load_all():
    frames = []
    for c in ["C1", "C3", "C5"]:
        df = pd.read_csv(ROOT / f"results/{c}/results_raw.csv")
        df = df[df["ran"] == True].copy() if "ran" in df.columns else df  # noqa: E712
        df["cluster"] = c
        df["axis"] = df["split"].map(axis_of)
        df["is_simple"] = df["baseline"].isin(SIMPLE)
        frames.append(df[["cluster", "axis", "split", "baseline", "family", "is_simple",
                           "pearson_delta", "e_distance"]])
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_all()
    # restrict to the two axes where both conditioned + simple compete and the protocols diverge
    use = df[df["axis"].isin(["cell-context", "perturbation"])].copy()
    print("=== rows per cluster x axis ===")
    print(use.groupby(["cluster", "axis"]).size())

    # ---------------- (A) OUR axis-stratified verdict ----------------
    print("\n=== (A) Leak-safe AXIS-STRATIFIED (ours): best conditioned vs best simple floor, per axis ===")
    for axis in ["cell-context", "perturbation"]:
        sub = use[use["axis"] == axis]
        # best simple floor and best conditioned, per split, then averaged over splits in the axis
        recs = []
        for sp, g in sub.groupby("split"):
            simple = g[g["is_simple"]]["pearson_delta"].max()
            cond = g[~g["is_simple"]]["pearson_delta"].max()
            recs.append((sp, simple, cond, cond - simple))
        r = pd.DataFrame(recs, columns=["split", "best_simple", "best_cond", "gap"])
        mean_gap = r["gap"].mean()
        n_cond_win = int((r["gap"] > 0).sum())
        print(f"  {axis:13s}: best-cond − best-simple mean gap = {mean_gap:+.3f}  "
              f"(conditioned beats floor in {n_cond_win}/{len(r)} splits)")
        if axis == "perturbation":
            print(r.round(3).to_string(index=False))

    # ---------------- (B) Open-Problems-style MIXED POOL ----------------
    print("\n=== (B) Open-Problems-style MIXED POOL (all cell-context + perturbation splits, equal weight) ===")
    # each model: mean pearson_delta over every split it ran on (axes mixed)
    board = (use.groupby(["baseline", "family", "is_simple"])["pearson_delta"]
             .agg(["mean", "count"]).reset_index().sort_values("mean", ascending=False))
    board.columns = ["baseline", "family", "is_simple", "mean_pearson_delta", "n_splits"]
    print(board.round(3).to_string(index=False))
    top = board.iloc[0]
    # is the top-ranked pooled model a CONDITIONED one that loses on the perturbation axis?
    print(f"\n  TOP of pooled board: {top['baseline']} ({top['family']}, "
          f"simple={top['is_simple']}) mean Pearson-Δ={top['mean_pearson_delta']:.3f}")

    # ---------------- (B2) cluster-local pooled boards (C5 cleanest: both axes, conditioned+simple) ----------------
    for c in ["C5", "C1"]:
        sub = use[use["cluster"] == c]
        if sub.empty:
            continue
        bc = (sub.groupby(["baseline", "is_simple"])["pearson_delta"].mean()
              .reset_index().sort_values("pearson_delta", ascending=False))
        print(f"\n  --- {c} pooled (mixed-axis) board ---")
        print(bc.round(3).to_string(index=False))

    # ---------------- (C) scPerturb-style POOLED E-distance ----------------
    print("\n=== (C) scPerturb-style POOLED E-distance leaderboard (lower = better, axes mixed) ===")
    eboard = (use.dropna(subset=["e_distance"]).groupby(["baseline", "family", "is_simple"])["e_distance"]
              .agg(["mean", "count"]).reset_index().sort_values("mean", ascending=True))
    eboard.columns = ["baseline", "family", "is_simple", "mean_e_distance", "n"]
    print(eboard.round(3).to_string(index=False))

    # ---------------- The demonstration: who pooled crowns vs who stratified demotes ----------------
    print("\n=== DEMONSTRATION ===")
    # For each conditioned model, its perturbation-axis gap vs the floor (stratified verdict)
    for c, cond_model in [("C5", "FP-ridge"), ("C5", "scGen"), ("C5", "CPA"),
                          ("C1", "scGen"), ("C3", "CPA"), ("C3", "GEARS")]:
        sub = use[(use["cluster"] == c)]
        pert = sub[sub["axis"] == "perturbation"]
        cc = sub[sub["axis"] == "cell-context"]
        for axis_name, ax in [("perturbation", pert), ("cell-context", cc)]:
            if ax.empty:
                continue
            # gap vs best simple per split, averaged
            gaps = []
            for sp, g in ax.groupby("split"):
                if cond_model not in set(g["baseline"]):
                    continue
                cm = g[g["baseline"] == cond_model]["pearson_delta"].max()
                fl = g[g["is_simple"]]["pearson_delta"].max()
                gaps.append(cm - fl)
            if gaps:
                print(f"  {c} {cond_model:10s} {axis_name:13s}: mean gap vs floor = {np.mean(gaps):+.3f} "
                      f"({sum(1 for x in gaps if x>0)}/{len(gaps)} win)")

    df.to_csv(ROOT / "results/_paper/prior_confrontation_rows.csv", index=False)
    board.to_csv(ROOT / "results/_paper/prior_confrontation_pooled_board.csv", index=False)
    print("\nWROTE results/_paper/prior_confrontation_{rows,pooled_board}.csv")


if __name__ == "__main__":
    main()
