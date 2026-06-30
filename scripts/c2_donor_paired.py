#!/usr/bin/env python3
"""Recompute the C2 donor paired statistics (Supplementary Tables S9, S10) from the prediction bundles.

Single source of truth = the deposited bundles (same as the headline census). The donor-held-out task
carries two split-construction schemes among the deposited bundles: the bespoke `C2_soskic_LODO_*` folds
(CellOT, scPRAM) and the framework `C2_lodo_*` folds (the simple floors). Both index the same 106 donors,
so we pair them by donor id and run the paired Wilcoxon exactly as the paper reports it:

    CellOT vs the cell-mean floor   -> results/_paper/cellot_vs_floor_donor_paired.csv   (S9)
    scPRAM vs CellOT                -> results/_paper/scpram_vs_cellot_donor_paired.csv   (S10)

This reproduces the main-text headline donor result (CellOT 0.3666, mean gap +0.107, 93/106 donors,
paired Wilcoxon p = 1.58e-13). The matched-per-donor context baseline columns (matched_baseline,
gap_vs_matched, ci_low/ci_high) are NOT derivable from the standard model bundles, so they are preserved
from the deposited CSV; everything else is re-derived from the bundles. `scripts/headline_multiplicity.py`
reads the `wilcoxon_p` here, so re-running this then that regenerates S11 consistently.
"""
from __future__ import annotations
import os

import pandas as pd
from scipy.stats import wilcoxon

from assemble_cross_cluster import OUT, score_all


def _per_donor(df, cluster, prefix, model):
    d = df[(df["cluster"] == cluster) & df["split"].str.startswith(prefix) & (df["model"] == model)].copy()
    d["donor"] = d["split"].str.replace(prefix + "_", "", regex=False)
    return d.set_index("donor")["pearson_delta"]


def main():
    df = score_all()
    cellot = _per_donor(df, "C2", "C2_soskic_LODO", "CellOT")
    scpram = _per_donor(df, "C2", "C2_soskic_LODO", "scPRAM")
    cmean = _per_donor(df, "C2_LODO", "C2_lodo", "cell-mean")

    # ---- S5: CellOT vs cell-mean floor, paired by donor ----
    i = cellot.index.intersection(cmean.index)
    gap = cellot.loc[i] - cmean.loc[i]
    p5 = os.path.join(OUT, "cellot_vs_floor_donor_paired.csv")
    s5 = pd.read_csv(p5)
    s5.loc[0, "cellot"] = float(cellot.loc[i].mean())
    s5.loc[0, "cell_mean"] = float(cmean.loc[i].mean())
    s5.loc[0, "gap"] = float(gap.mean())
    s5.loc[0, "cellot_wins"] = int((gap > 0).sum())
    s5.loc[0, "wilcoxon_p"] = float(wilcoxon(cellot.loc[i], cmean.loc[i]).pvalue)
    s5.loc[0, "n"] = int(len(i))
    if "src" in s5.columns:
        s5.loc[0, "src"] = "prediction bundles (predictions/C2; C2_soskic_LODO CellOT vs C2_lodo cell-mean, paired by donor)"
    s5.to_csv(p5, index=False)

    # ---- S7: scPRAM vs CellOT, paired by donor ----
    j = cellot.index.intersection(scpram.index)
    g2 = scpram.loc[j] - cellot.loc[j]
    p7 = os.path.join(OUT, "scpram_vs_cellot_donor_paired.csv")
    s7 = pd.read_csv(p7)
    s7.loc[0, "scpram_wins"] = int((g2 > 0).sum())
    s7.loc[0, "mean_gap"] = float(g2.mean())
    s7.loc[0, "scpram_mean"] = float(scpram.loc[j].mean())
    s7.loc[0, "cellot_mean"] = float(cellot.loc[j].mean())
    s7.loc[0, "wilcoxon_p"] = float(wilcoxon(scpram.loc[j], cellot.loc[j]).pvalue)
    s7.to_csv(p7, index=False)

    print(f"S5 CellOT-vs-floor: CellOT {cellot.loc[i].mean():.4f}  gap {gap.mean():+.4f}  "
          f"{int((gap>0).sum())}/{len(i)}  p {float(wilcoxon(cellot.loc[i], cmean.loc[i]).pvalue):.3g}")
    print(f"S7 scPRAM-vs-CellOT: scPRAM {scpram.loc[j].mean():.4f}  CellOT {cellot.loc[j].mean():.4f}  "
          f"gap {g2.mean():+.4f}  wins {int((g2>0).sum())}/{len(j)}  p {float(wilcoxon(scpram.loc[j], cellot.loc[j]).pvalue):.3g}")
    print("main text: CellOT 0.3666, gap +0.107, 93/106, p 1.58e-13")


if __name__ == "__main__":
    main()
