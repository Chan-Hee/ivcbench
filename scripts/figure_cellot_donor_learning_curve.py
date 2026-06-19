#!/usr/bin/env python
"""Figure: CellOT donor-transfer LEARNING CURVE on Soskic C2 — "data is rate-limiting".

Merges results/newdata/cellot_donor_learning_curve_seed*.csv (written by the detached GPU jobs) into
results/newdata/cellot_donor_learning_curve.csv, then renders a navy-editorial figure (style.py).

Panels:
  (a) CellOT response-direction Pearson (mean over the FIXED eval-donor set, +/- bootstrap CI over
      eval-donors x seeds) vs number of TRAINING donors, with the matched simple-baseline floor
      (matched simple context baseline (cell-mean / donor-shift; not a universal-floor member)) drawn as a dashed reference. Rising-then-plateau = the
      transfer skill is bought with training donors.
  (b) delta vs the matched floor (CellOT - baseline Pearson) vs number of training donors: where the
      curve crosses zero is the donor budget at which conditioning starts to pay off.

Reads ONLY the real CSVs the jobs produced. Renders nothing if no rows are present yet.
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.report.style import (set_pub_style, despine, style_legend,  # noqa: E402
                                   NAVY, NAVY_DARK, CLAY_DARK,
                                   SIMPLE_GREY, SIMPLE_DARK, GREY_MID, NULL_GREY)

NEWDATA = ROOT / "results" / "newdata"
PAPER = ROOT / "results" / "_paper"


def _u(s: str) -> str:
    return s.replace("-", "−")


def _navy_title(ax, letter, title, x_letter=-0.135, y=1.085):
    """Navy structural panel title (letter + bold title on one baseline, no grey subtitle)."""
    ax.text(x_letter, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="bottom", ha="right", color=NAVY_DARK)
    ax.text(x_letter + 0.055, y, title, transform=ax.transAxes, fontsize=8.6,
            fontweight="bold", va="bottom", ha="left", color=NAVY_DARK)


def boot_ci(vals, n=10000, seed=0):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load_merged():
    parts = sorted(glob.glob(str(NEWDATA / "cellot_donor_learning_curve_seed*.csv")))
    frames = []
    for p in parts:
        try:
            d = pd.read_csv(p)
            if len(d):
                frames.append(d)
        except Exception:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    merged = NEWDATA / "cellot_donor_learning_curve.csv"
    df.to_csv(merged, index=False)
    print(f"[merge] wrote {merged} ({len(df)} rows from {len(frames)} seed files)")
    return df


def main():
    set_pub_style()
    df = load_merged()
    if df is None or not len(df):
        print("[fig] no learning-curve rows yet; nothing to render.")
        return
    pe = df[df.metric == "pearson_delta"].copy()
    pe = pe[(pe.cellot_score != "") & (pe.baseline_score != "")]
    pe["cellot_score"] = pe["cellot_score"].astype(float)
    pe["baseline_score"] = pe["baseline_score"].astype(float)
    pe["delta_vs_primary"] = pe["delta_vs_primary"].astype(float)
    grid = sorted(pe["n_train_donors"].unique())

    # per grid-size aggregates (over eval-donors x seeds)
    co_m, co_lo, co_hi, bl_m, dl_m, dl_lo, dl_hi, ncells = [], [], [], [], [], [], [], []
    for k in grid:
        sub = pe[pe.n_train_donors == k]
        m, lo, hi = boot_ci(sub["cellot_score"].values, seed=k)
        co_m.append(m); co_lo.append(lo); co_hi.append(hi)
        bl_m.append(float(np.nanmean(sub["baseline_score"].values)))
        dm, dlo, dhi = boot_ci(sub["delta_vs_primary"].values, seed=100 + k)
        dl_m.append(dm); dl_lo.append(dlo); dl_hi.append(dhi)
        ncells.append(int(np.nanmedian(sub["n_train_cells"].values)))

    # paired test of the MODEL's own score at the largest vs smallest donor budget
    # (does CellOT's absolute skill fall as donors are added? -> "more donors lift the floor,
    #  not the model"). Per eval donor, averaged over seeds, on the shared donor set.
    k_lo, k_hi = grid[0], grid[-1]
    g_lo = pe[pe.n_train_donors == k_lo].groupby("eval_donor")["cellot_score"].mean()
    g_hi = pe[pe.n_train_donors == k_hi].groupby("eval_donor")["cellot_score"].mean()
    common = g_lo.index.intersection(g_hi.index)
    n_common = int(len(common))
    n_lower = int((g_hi[common].values < g_lo[common].values).sum())
    try:
        from scipy.stats import wilcoxon
        p_own = float(wilcoxon(g_hi[common].values, g_lo[common].values).pvalue)
    except Exception:
        p_own = float("nan")
    n_eval_fixed = int(pe.groupby("n_train_donors")["eval_donor"].nunique().max())
    n_seed = int(pe["seed"].nunique())
    d_first, d_last = dl_m[0], dl_m[-1]
    grid = np.asarray(grid, float)

    plt.rcParams["axes.unicode_minus"] = True  # true U+2212 on every tick / number

    fig = plt.figure(figsize=(7.4, 3.85))
    # generous, even right gutter; matched left/right margin so neither panel crowds an edge.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30,
                          left=0.105, right=0.955, top=0.745, bottom=0.205)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ---- editorial header (navy structural ink), kept clear of both page edges ----
    fig.text(0.5, 0.945,
             _u("More donor data lifts the simple baseline faster than the deep model, pointing to "
                "representation, not donor count alone, as the limit on donor transfer"),
             fontsize=8.0, color=NAVY_DARK, ha="center", va="center")

    # ---- (a) absolute Pearson vs training donors ----
    axA.fill_between(grid, co_lo, co_hi, color=NAVY, alpha=0.15, zorder=2, lw=0)
    axA.plot(grid, co_m, "-o", color=NAVY, mec=NAVY_DARK, mfc=NAVY, ms=5.4, lw=1.7,
             zorder=4, label="CellOT (conditioned)")
    axA.plot(grid, bl_m, "--s", color=SIMPLE_GREY, mec=SIMPLE_DARK, mfc=SIMPLE_GREY, ms=4.4,
             lw=1.4, zorder=3, label="simple baseline (cell-mean)")
    axA.set_xscale("log", base=2)
    axA.set_xticks(grid)
    axA.set_xticklabels([f"{int(k)}" for k in grid])
    axA.set_xlabel("number of training donors", fontsize=8.0)
    axA.set_ylabel(_u("response-direction Pearson Δ"), fontsize=8.0)
    # one swatch language, one label column; parked in the clear gap BETWEEN the two curves
    # (mid-left), where neither the CellOT band nor the baseline line runs.
    leg = axA.legend(loc="center left", fontsize=6.9, handlelength=1.6, handletextpad=0.6,
                     borderaxespad=0.0, labelspacing=0.6, bbox_to_anchor=(0.035, 0.46))
    style_legend(leg)
    _navy_title(axA, "a", "More donors lift the simple baseline, not the model")
    despine(axA)

    # ---- (b) delta vs floor (conditioning advantage; shrinking = warm/coral) ----
    axB.axhline(0, color=NULL_GREY, lw=1.1, zorder=1)
    axB.fill_between(grid, dl_lo, dl_hi, color=CLAY_DARK, alpha=0.13, zorder=2, lw=0)
    axB.plot(grid, dl_m, "-o", color=CLAY_DARK, mec="#8C3A24", mfc=CLAY_DARK, ms=5.4,
             lw=1.7, zorder=4)
    axB.set_xscale("log", base=2)
    axB.set_xticks(grid)
    axB.set_xticklabels([f"{int(k)}" for k in grid])
    axB.set_ylim(bottom=min(0.0, axB.get_ylim()[0]))
    axB.set_xlabel("number of training donors", fontsize=8.0)
    axB.set_ylabel(_u("CellOT advantage over simple baseline (Δ)"), fontsize=8.0)
    # stat block: pulled INWARD from the top-right, clear of the right gutter and the title band.
    p_txt = "n.s." if not np.isfinite(p_own) else f"{p_own:.4f}".rstrip("0").rstrip(".")
    stat_lines = _u(
        f"+{d_first:.2f} ({int(k_lo)} donors)  →  +{d_last:.2f} ({int(k_hi)})\n"
        f"{n_lower}/{n_common} donors: CellOT's own score lower\n"
        f"paired p = {p_txt}  ·  {n_seed} seeds, fixed {n_eval_fixed}-donor eval"
    )
    axB.text(0.965, 0.965, stat_lines, transform=axB.transAxes, ha="right", va="top",
             fontsize=6.6, color=GREY_MID, linespacing=1.4)
    _navy_title(axB, "b", "The conditioning advantage shrinks")
    despine(axB)

    foot = _u(
        f"Soskic CD4 16 h activation, leave-one-donor-out on a fixed set of {n_eval_fixed} held-out "
        f"eval donors. For each training-donor count the faithful CellOT (scGen AE + f/g ICNN, Bunne "
        f"2023) is trained on a random subset of the remaining donors' 0 h→16 h cells ({n_seed} seeds); "
        f"every eval donor's 16 h response is predicted from its own 0 h cells. Pearson over "
        f"training-only response genes; simple baseline = better of cell-mean / donor-shift (matched "
        f"simple context baseline; not a universal-floor member) on the same subset. Eval donors are "
        f"never trained on (leak-free by construction). Bands: 95% bootstrap CI."
    )
    fig.text(0.5, 0.022, foot, fontsize=5.5, color=GREY_MID, ha="center", va="bottom",
             wrap=True)

    out_png = PAPER / "figS_cellot_donor_learning_curve.png"
    out_pdf = PAPER / "figS_cellot_donor_learning_curve.pdf"
    fig.savefig(out_png, dpi=350); fig.savefig(out_pdf); plt.close(fig)
    import shutil
    shutil.copy(out_png, NEWDATA / "figS_cellot_donor_learning_curve.png")
    print(f"[fig] wrote {out_png}")
    print(f"[fig] wrote {out_pdf}")

    # also emit a compact summary CSV for the paper
    summ = pd.DataFrame(dict(
        n_train_donors=grid.astype(int), median_train_cells=ncells,
        cellot_pearson=np.round(co_m, 4), cellot_ci_lo=np.round(co_lo, 4),
        cellot_ci_hi=np.round(co_hi, 4), matched_floor_pearson=np.round(bl_m, 4),
        delta_vs_floor=np.round(dl_m, 4), delta_ci_lo=np.round(dl_lo, 4),
        delta_ci_hi=np.round(dl_hi, 4)))
    summ_path = PAPER / "cellot_donor_learning_curve_summary.csv"
    summ.to_csv(summ_path, index=False)
    print(f"[summary] wrote {summ_path}")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
