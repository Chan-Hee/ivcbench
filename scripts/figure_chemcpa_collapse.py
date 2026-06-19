#!/usr/bin/env python
"""Supplementary figure: chemCPA chemistry-channel collapse on OP3 unseen-compound.

Every value is read from the deposited arrays/stats computed by chemcpa_collapse_analysis.py
(outputs/additional_models/chemcpa_collapse_arrays.npz + chemcpa_collapse_stats.json), which were
themselves computed from the deposited predictions + the real OP3 loader/split. No fabricated numbers.

Panels:
  (a) Per-gene between-compound variance — PREDICTED vs OBSERVED (log-log scatter). Predicted points
      sit ~4-5 orders of magnitude below observed and below the equality line: the chemistry channel
      reproduces essentially none of the real between-compound variance.
  (b) Near-constant prediction — for the 12 highest-observed-variance genes, the per-compound
      PREDICTED Δ (flat, all 28 compounds overplotted) vs the OBSERVED Δ spread (real compounds differ).
  (c) Pairwise between-compound similarity — distribution of pairwise Pearson r across the 28 held
      compounds, PREDICTED (≈1.000, a single profile) vs OBSERVED (broad, real chemistry).

Rendered in the navy editorial style (src/ivcbench/report/style.py).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(REPO, "src"))
PAPER = os.path.join(REPO, "results/_paper")
# chemcpa_collapse_arrays.npz + chemcpa_collapse_stats.json are deposited alongside the figures
OUT = PAPER

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ivcbench.report.style import (  # noqa: E402
    set_pub_style, NAVY, NAVY_DARK, CLAY_DARK, INK, GREY_MID, SLATE_BAND,
    panel_title, despine, style_legend,
)

PRED_C = CLAY_DARK     # the collapsed chemistry prediction (warm, "degenerate")
OBS_C = NAVY           # the real observed signal (the author's strong navy)


def _pairwise_r(M):
    C = np.corrcoef(M)
    iu = np.triu_indices(M.shape[0], k=1)
    return C[iu]


def main():
    set_pub_style()
    plt.rcParams["axes.unicode_minus"] = True

    arr = np.load(os.path.join(OUT, "chemcpa_collapse_arrays.npz"), allow_pickle=True)
    stats = json.load(open(os.path.join(OUT, "chemcpa_collapse_stats.json")))
    pred_delta = arr["pred_delta"].astype(np.float64)   # (28, 2000)
    obs_delta = arr["obs_delta"].astype(np.float64)
    n_cpd = pred_delta.shape[0]

    sa = stats["seed_averaged"]
    var_pred = sa["between_compound_var_PRED_delta"]
    var_obs = sa["between_compound_var_OBS_delta"]
    ratio = sa["collapse_ratio_delta_pred_over_obs"]

    # per-gene between-compound variance
    vpred_g = pred_delta.var(axis=0, ddof=1)
    vobs_g = obs_delta.var(axis=0, ddof=1)

    fig = plt.figure(figsize=(11.0, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.18, 0.92],
                          left=0.065, right=0.985, bottom=0.165, top=0.80, wspace=0.42)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # ---------------- (a) per-gene between-compound variance: pred vs obs ----------------
    eps = 1e-12
    axA.scatter(vobs_g + eps, vpred_g + eps, s=7, c=PRED_C, alpha=0.45,
                edgecolors="none", rasterized=True)
    lo = min((vpred_g + eps).min(), (vobs_g + eps).min())
    hi = max((vpred_g + eps).max(), (vobs_g + eps).max())
    line = np.array([lo, hi])
    axA.plot(line, line, color=INK, lw=0.9, ls="--", zorder=1)
    axA.text(hi, hi, " y = x", color=INK, fontsize=6.6, va="bottom", ha="right")
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel("observed between-compound\nvariance (per gene)")
    axA.set_ylabel("predicted between-compound\nvariance (per gene)")
    despine(axA)
    panel_title(axA, "a", "Chemistry channel reproduces\nno between-compound variance",
                x_letter=-0.20, y=1.07)
    axA.text(0.04, 0.93,
             f"mean var: predicted {var_pred:.1e}\nobserved {var_obs:.3f}\nratio {ratio:.1e}",
             transform=axA.transAxes, fontsize=6.6, va="top", ha="left", color=GREY_MID)

    # ---------------- (b) near-constant prediction on top-variance genes ----------------
    k = 12
    top = np.argsort(vobs_g)[::-1][:k]
    xpos = np.arange(k)
    # observed Δ spread per gene (across the 28 compounds): box-like via individual points
    for j, gi in enumerate(top):
        ov = obs_delta[:, gi]
        pv = pred_delta[:, gi]
        # observed: spread of real compounds (jittered)
        jit = (np.random.default_rng(j).random(n_cpd) - 0.5) * 0.34
        axB.scatter(np.full(n_cpd, j) + jit - 0.0, ov, s=6, c=OBS_C, alpha=0.55,
                    edgecolors="none", zorder=2)
        # observed mean bar
        axB.plot([j - 0.28, j + 0.28], [ov.mean(), ov.mean()], color=NAVY_DARK, lw=1.4, zorder=3)
        # predicted: all 28 collapse onto ~one value -> a single clay tick
        axB.plot([j - 0.28, j + 0.28], [pv.mean(), pv.mean()], color=PRED_C, lw=1.6, zorder=4)
    axB.axhline(0, color=GREY_MID, lw=0.6, ls=":", zorder=0)
    axB.set_xticks(xpos)
    axB.set_xticklabels([str(arr["genes"][gi]) for gi in top], rotation=60, ha="right", fontsize=5.6)
    axB.set_xlabel("top-12 highest observed-variance genes")
    axB.set_ylabel("per-compound Δ (treated − control)")
    despine(axB)
    panel_title(axB, "b", "Predicted Δ is near-constant across\nthe 28 chemically-diverse compounds",
                x_letter=-0.135, y=1.07)
    # mini legend
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker="o", color="none", markerfacecolor=OBS_C, markersize=4,
                label="observed Δ (per compound)"),
         Line2D([0], [0], color=NAVY_DARK, lw=1.4, label="observed mean"),
         Line2D([0], [0], color=PRED_C, lw=1.6, label="chemCPA predicted (all 28 ≈ identical)")]
    leg = axB.legend(handles=h, loc="upper right", fontsize=6.0, handlelength=1.3,
                     borderaxespad=0.3, labelspacing=0.35, frameon=True)
    style_legend(leg)

    # ---------------- (c) pairwise between-compound Pearson r ----------------
    r_pred = _pairwise_r(pred_delta)
    r_obs = _pairwise_r(obs_delta)
    bins = np.linspace(-0.7, 1.0, 36)
    axC.hist(r_obs, bins=bins, color=OBS_C, alpha=0.75, edgecolor="white", linewidth=0.3,
             label=f"observed (mean {r_obs.mean():.2f})")
    axC.hist(r_pred, bins=bins, color=PRED_C, alpha=0.85, edgecolor="white", linewidth=0.3,
             label=f"predicted (mean {r_pred.mean():.4f})")
    axC.axvline(1.0, color=PRED_C, lw=1.0, ls="--")
    axC.set_xlabel("pairwise Pearson r between\ncompound Δ profiles")
    axC.set_ylabel("compound pairs (count)")
    axC.set_xlim(-0.72, 1.04)
    despine(axC)
    panel_title(axC, "c", "Predicted profiles are mutually\nidentical (r ≈ 1.000)",
                x_letter=-0.20, y=1.07)
    leg2 = axC.legend(loc="upper left", fontsize=6.2, handlelength=1.1, borderaxespad=0.3,
                      labelspacing=0.4, frameon=True)
    style_legend(leg2)

    fig.suptitle(
        "chemCPA chemistry-aware drug encoder collapses to a near-constant profile on unseen OP3 compounds",
        fontsize=9.2, fontweight="bold", color=NAVY_DARK, x=0.065, ha="left", y=0.965)

    out_png = os.path.join(PAPER, "figure_chemcpa_collapse.png")
    out_pdf = os.path.join(PAPER, "figure_chemcpa_collapse.pdf")
    fig.savefig(out_png, dpi=350)
    fig.savefig(out_pdf)
    print(f"[fig] wrote {out_png}\n[fig] wrote {out_pdf}")
    print(f"[fig] var_pred={var_pred:.3e} var_obs={var_obs:.4f} ratio={ratio:.3e} "
          f"r_pred_mean={r_pred.mean():.5f} r_obs_mean={r_obs.mean():.3f}")


if __name__ == "__main__":
    main()
