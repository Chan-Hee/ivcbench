#!/usr/bin/env python
"""C3 predictability probe — analysis + supplementary figure (navy editorial style).

Reads results/C3/predictability_probe_pergene.csv (per-held-gene predictability + explanatory
variables computed from the deposited C3 data by c3_predictability_probe.py) and:
  1. ranks candidate explanatory factors by |Spearman| with per-gene predictability (pooled, and
     dataset-fixed-effect partial correlation so cross-dataset offsets don't drive the ranking);
  2. renders a 2-panel figure: (a) predictability vs the STRONGEST factor (scatter, datasets coloured,
     OLS fit + rho); (b) the factor-ranking bars (|Spearman|, 95% CI bootstrapped over genes), with
     effect-size vs representation-distance families colour-coded.

Writes results/_paper/figS_c3_predictability_probe.{png,pdf} and prints the numeric ranking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.report.style import (set_pub_style, despine, panel_title, style_legend,  # noqa: E402
                                   NAVY, NAVY_DARK, SLATE_BAND, CLAY_DARK, SIMPLE_GREY,
                                   GREY_MID, INK, NULL_GREY, LEGEND_EC, NAVY_RAMP)

PAPER = ROOT / "results" / "_paper"
SRC = ROOT / "results/C3/predictability_probe_pergene.csv"

# factor -> (display label, family: 'effect' | 'repr' | 'overlap', expected sign vs predictability)
FACTORS = {
    "effect_l2":     ("effect size  ||Δ_obs||",              "effect",  +1),
    "snr_raw":       ("SNR  ||Δ|| / within-pert SD",         "effect",  +1),
    "n_test_cells":  ("# treated cells (support)",            "effect",  +1),
    "go_jaccard_nn": ("GO-Jaccard distance to nearest train gene", "repr", -1),
    "go_jaccard_k5": ("GO-Jaccard distance (top-5 mean)",     "repr",   -1),
    "coexpr_nn":     ("co-expression distance to nearest train gene", "repr", -1),
    "resp_overlap":  ("response-gene overlap with train",     "overlap", +1),
}
FAM_COL = {"effect": NAVY, "repr": CLAY_DARK, "overlap": SLATE_BAND}
FAM_LAB = {"effect": "effect-size / SNR", "repr": "representation distance", "overlap": "response overlap"}


def _u(s: str) -> str:
    return s.replace("-", "−")


def spearman_ci(x, y, n=10000, seed=0):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 6:
        return np.nan, np.nan, np.nan, np.nan, len(x)
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        r, _p = stats.spearmanr(x[idx], y[idx])
        bs.append(r)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(rho), float(lo), float(hi), float(p), len(x)


def partial_spearman_within_dataset(df, factor, target="predictability"):
    """Rank-correlation after removing dataset means (rank within dataset, then pool)."""
    parts_x, parts_y = [], []
    for ds, g in df.groupby("dataset"):
        x = g[factor].to_numpy(float)
        y = g[target].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 4:
            continue
        # within-dataset rank residualization
        rx = stats.rankdata(x[m]); ry = stats.rankdata(y[m])
        parts_x.append(rx - rx.mean()); parts_y.append(ry - ry.mean())
    if not parts_x:
        return np.nan, np.nan, 0
    X = np.concatenate(parts_x); Y = np.concatenate(parts_y)
    if X.std() < 1e-9 or Y.std() < 1e-9:
        return np.nan, np.nan, len(X)
    r, p = stats.pearsonr(X, Y)
    return float(r), float(p), len(X)


def main():
    set_pub_style()
    df = pd.read_csv(SRC)
    print(f"loaded {SRC} shape={df.shape}")
    print("predictability: mean=%.3f median=%.3f min=%.3f max=%.3f" % (
        df.predictability.mean(), df.predictability.median(),
        df.predictability.min(), df.predictability.max()))
    print(f"n held-gene observations = {len(df)}  (across {df.dataset.nunique()} datasets x 3 holdouts)")
    print()

    # ---- factor ranking ----
    rank = []
    for fac, (lab, fam, sign) in FACTORS.items():
        rho, lo, hi, p, n = spearman_ci(df[fac].to_numpy(float), df.predictability.to_numpy(float))
        prho, pp, pn = partial_spearman_within_dataset(df, fac)
        rank.append(dict(factor=fac, label=lab, family=fam, sign=sign,
                         rho=rho, lo=lo, hi=hi, p=p, n=n,
                         partial_rho=prho, partial_p=pp, partial_n=pn,
                         abs_rho=abs(rho) if np.isfinite(rho) else 0.0))
    rk = pd.DataFrame(rank).sort_values("abs_rho", ascending=False).reset_index(drop=True)
    print("=== FACTOR RANKING (per-gene predictability = floor Pearson-Δ) ===")
    for _, r in rk.iterrows():
        print(f"  {r.label:42s} pooled rho={r.rho:+.3f} [{r.lo:+.3f},{r.hi:+.3f}] p={r.p:.1e} "
              f"| within-dataset rho={r.partial_rho:+.3f} p={r.partial_p:.1e}  n={r.n}")
    rk.to_csv(ROOT / "results/C3/predictability_factor_ranking.csv", index=False)
    print(f"\nwrote results/C3/predictability_factor_ranking.csv")

    top = rk.iloc[0]
    print(f"\nSTRONGEST FACTOR: {top.label}  (|rho|={top.abs_rho:.3f})")

    # =================== FIGURE ===================
    fig = plt.figure(figsize=(11.0, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.15], wspace=0.32,
                          left=0.075, right=0.985, top=0.84, bottom=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ---- (a) scatter: predictability vs strongest factor ----
    fac = top.factor
    x = df[fac].to_numpy(float); y = df.predictability.to_numpy(float)
    ds_list = sorted(df.dataset.unique())
    ramp = ["#0F4C75", "#3E6E92", "#6E97B4", "#9E5A3C", "#C28C6F"]
    ds_col = {d: ramp[i % len(ramp)] for i, d in enumerate(ds_list)}
    mk = {"10": "o", "25": "s", "50": "^"}
    for d in ds_list:
        for h in ["10", "25", "50"]:
            sub = df[(df.dataset == d) & (df.hold.astype(str) == h)]
            axA.scatter(sub[fac], sub.predictability, s=34, color=ds_col[d],
                        marker=mk[h], edgecolor="white", linewidth=0.5, alpha=0.9, zorder=3)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() > 2:
        b1, b0 = np.polyfit(x[m], y[m], 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        axA.plot(xs, b0 + b1 * xs, color=NAVY_DARK, lw=1.6, zorder=4)
    axA.text(0.04, 0.96, _u(f"Spearman ρ = {top.rho:+.2f}\n(within-dataset ρ = {top.partial_rho:+.2f})"),
             transform=axA.transAxes, va="top", ha="left", fontsize=7.4, color=NAVY_DARK,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=LEGEND_EC, lw=0.6))
    axA.set_xlabel(_u(top.label + "  →"), fontsize=7.8)
    axA.set_ylabel(_u("per-gene predictability\n(floor response-direction Pearson-Δ)  ↑"), fontsize=7.8)
    axA.axhline(0, color="#ccc", lw=0.6, zorder=0)
    panel_title(axA, "a", "What makes an unseen gene predictable?",
                sub=f"strongest single factor: {FAM_LAB[top.family]}", x_letter=-0.13)
    despine(axA)

    # ---- (b) factor-ranking bars ----
    rk2 = rk.sort_values("abs_rho").reset_index(drop=True)
    yy = np.arange(len(rk2))
    for y0, (_, r) in zip(yy, rk2.iterrows()):
        col = FAM_COL[r.family]
        axB.barh(y0, r.abs_rho, height=0.62, color=col, edgecolor=INK, linewidth=0.6,
                 alpha=0.92, zorder=3)
        # CI on |rho|: use abs of pooled CI ends conservatively
        lo, hi = abs(r.lo), abs(r.hi)
        clo, chi = min(lo, hi), max(lo, hi)
        axB.plot([clo, chi], [y0, y0], color=INK, lw=0.9, zorder=4)
        sig = "*" if (np.isfinite(r.p) and r.p < 0.05) else ""
        axB.text(r.abs_rho + 0.012, y0, _u(f"{r.rho:+.2f}{sig}"), va="center", ha="left",
                 fontsize=6.9, color=INK)
    axB.set_yticks(yy)
    axB.set_yticklabels([_u(r.label) for _, r in rk2.iterrows()], fontsize=6.9)
    axB.set_xlim(0, max(0.05, rk2.abs_rho.max() * 1.28))
    axB.set_xlabel(_u("|Spearman ρ| with per-gene predictability"), fontsize=7.8)
    panel_title(axB, "b", "Response-profile factors rank first; representation distance last",
                sub="bars = |ρ|; signed ρ labelled; * p<0.05", x_letter=-0.30)
    despine(axB)

    # family legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=FAM_COL[k], edgecolor=INK, lw=0.6, label=FAM_LAB[k])
               for k in ["effect", "repr", "overlap"]]
    # dataset legend for panel a
    ds_handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=ds_col[d], mec="white",
                             ms=6, label=d) for d in ds_list]
    hold_handles = [plt.Line2D([0], [0], marker=mk[h], ls="", mfc="#777", mec="white",
                               ms=6, label=f"{h}% holdout") for h in ["10", "25", "50"]]
    leg1 = fig.legend(handles=ds_handles + hold_handles, loc="lower left",
                      bbox_to_anchor=(0.075, -0.01), ncol=4, fontsize=6.3, frameon=True,
                      handlelength=1.0, columnspacing=1.0, title="panel a")
    leg1.get_title().set_fontsize(6.3)
    style_legend(leg1)
    leg2 = fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.985, -0.01),
                      ncol=1, fontsize=6.5, frameon=True, handlelength=1.0, title="factor family (panel b)")
    leg2.get_title().set_fontsize(6.3)
    style_legend(leg2)

    cap = _u(
        f"T3 leave-one-gene-out, {len(df)} held-gene observations (5 CRISPR datasets x 10/25/50% holdout). "
        "Per-gene predictability = the floor (best of cell-mean / donor-shift / linear-PCA) per-stratum "
        "downstream-only Pearson-Δ, recomputed exactly from the deposited cells; legitimate because NO heavy "
        "model beats this floor on any T3 cell (deposited mean heavy−floor gap −0.03, 0/15). Explanatory "
        "variables computed from the same cells: effect size/SNR from observed Δ; GO-Jaccard / co-expression "
        "distance to the nearest TRAINING perturbed gene; top-50 response-gene overlap with training. "
        "ρ = Spearman over held genes; within-dataset ρ removes cross-dataset offsets. The two top factors "
        "(response overlap, effect size) describe WHAT the held gene's response looks like; both abstract "
        "representation distances (GO, co-expression to nearest train gene) are flat, so unseen-gene "
        "predictability is set by response/effect structure, not by gene-embedding proximity.")
    fig.text(0.075, 0.005, cap, fontsize=5.6, color=GREY_MID, ha="left", va="bottom", wrap=True)

    out_png = PAPER / "figS_c3_predictability_probe.png"
    out_pdf = PAPER / "figS_c3_predictability_probe.pdf"
    fig.savefig(out_png, dpi=350)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"\n[fig] wrote {out_png}")
    print(f"[fig] wrote {out_pdf}")


if __name__ == "__main__":
    main()
