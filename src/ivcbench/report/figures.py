"""Per-cluster benchmark figure (Figs 3–7), publication-grade and driven generically by the
results table. Renders the 4-axis framework across a cluster's leak-proof splits:
  (a) Response-direction (Pearson-Δ, ↑) with 95% bootstrap CI
  (b) Distributional (energy distance, ↓) with 95% bootstrap CI
  (c) Immune-program (AUCell-Δ correlation, ↑)
  (d) Headline leaderboard (mean Pearson-Δ over applicable splits), family-colored
Floored cells (not headline-eligible under applicability gating) are drawn hatched + desaturated.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import (BASELINE_ORDER, FAMILY_COLORS, SPLIT_COLORS, despine, panel_label,
                    set_pub_style)

CLUSTER_TITLES = {
    "C1": "Cytokine-response prediction", "C2": "Donor-aware activation dynamics",
    "C3": "Gene-intervention prediction", "C4": "Complex-context prediction",
    "C5": "Small-molecule perturbation prediction",
}
_METRICS = ["pearson_delta", "e_distance", "aucell_program_corr",
            "pearson_delta_lo", "pearson_delta_hi", "e_distance_lo", "e_distance_hi"]


def _short(split: str, cluster: str) -> str:
    return split.replace(f"{cluster}_", "").replace("_", " ")


def _grouped_bars(ax, summ, cluster, metric, title, ylabel, higher_better=True):
    baselines = [b for b in BASELINE_ORDER if b in set(summ["baseline"])]
    splits = list(dict.fromkeys(summ["split"]))
    n = max(1, len(splits))
    w = 0.8 / n
    x = np.arange(len(baselines))
    has_ci = f"{metric}_lo" in summ.columns
    for i, s in enumerate(splits):
        color = SPLIT_COLORS[i % len(SPLIT_COLORS)]
        for j, b in enumerate(baselines):
            row = summ[(summ["split"] == s) & (summ["baseline"] == b)]
            if row.empty:
                continue
            m = float(row[metric].iloc[0])
            floored = not bool(row["headline_eligible"].iloc[0])
            yerr = None
            if has_ci:
                lo, hi = float(row[f"{metric}_lo"].iloc[0]), float(row[f"{metric}_hi"].iloc[0])
                yerr = [[max(0, m - lo)], [max(0, hi - m)]]
            ax.bar(x[j] + i * w, m, w, color=color, alpha=0.40 if floored else 0.92,
                   hatch="////" if floored else "", edgecolor=color, linewidth=0.8, zorder=2,
                   yerr=yerr, error_kw=dict(lw=0.8, capsize=2, ecolor="#333", zorder=3))
    ax.axhline(0, color="#333", lw=0.6, zorder=1)
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(baselines, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel + ("  ↑" if higher_better else "  ↓"))
    despine(ax)


def cluster_figure(raw_df: pd.DataFrame, cluster: str, out_path: str | Path) -> Path:
    set_pub_style()
    df = raw_df[raw_df["ran"]].copy()
    if "cluster" not in df.columns:
        df["cluster"] = cluster
    keys = ["cluster", "split", "baseline", "family", "action", "headline_eligible"]
    present = [m for m in _METRICS if m in df.columns]
    summ = df.groupby(keys, as_index=False)[present].mean()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    title = CLUSTER_TITLES.get(cluster, cluster)
    fig.suptitle(f"{cluster}  ·  {title}", fontsize=10.5, fontweight="bold", x=0.5, y=1.02)

    _grouped_bars(axes[0, 0], summ, cluster, "pearson_delta",
                  "Response-direction", "Pearson-Δ", True)
    _grouped_bars(axes[0, 1], summ, cluster, "e_distance",
                  "Distributional", "energy distance", False)
    _grouped_bars(axes[1, 0], summ, cluster, "aucell_program_corr",
                  "Immune-program", "AUCell-Δ corr", True)

    # (c) graceful handling: the program metric can be (i) UNDEFINED — no curated gene set for this
    # dataset, so every value is NaN (np.nanmax of all-NaN is NaN, which the old `< 1e-6` test missed,
    # leaving a blank panel); or (ii) structurally ~0 for constant-profile baselines. Show the right note.
    _au = summ["aucell_program_corr"].to_numpy(dtype=float)
    if _au.size and np.all(np.isnan(_au)):
        ax = axes[1, 0]
        ax.set_ylim(-0.05, 1.0)
        ax.text(0.5, 0.5,
                "AUCell-Δ not available — the immune-program\ngene set is not yet curated for this "
                "dataset\n(Axis 3 pending: the loader registers the program\nkey but supplies no genes).",
                transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color="#7a5a3a",
                bbox=dict(boxstyle="round,pad=0.5", fc="#fff6f0", ec="#e0a070", lw=0.8))
    elif _au.size and np.isfinite(np.nanmax(np.abs(_au))) and np.nanmax(np.abs(_au)) < 1e-6:
        ax = axes[1, 0]
        ax.set_ylim(-0.05, 1.0)
        ax.text(0.5, 0.55,
                "AUCell-Δ correlation is undefined for\nconstant-profile baselines (one predicted\n"
                "profile per split → zero cross-stratum\nvariance). Populated once conditioning\n"
                "models (scGen, CPA, scGPT, …) are added.",
                transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color="#555",
                bbox=dict(boxstyle="round,pad=0.5", fc="#f6f6f6", ec="#cccccc", lw=0.8))

    # (d) headline leaderboard — mean Pearson-Δ over applicable (headline-eligible) splits
    ax = axes[1, 1]
    head = summ[summ["headline_eligible"]]
    lead = (head.groupby(["baseline", "family"], as_index=False)["pearson_delta"].mean()
                .sort_values("pearson_delta"))
    nsplit = head.groupby("baseline")["split"].nunique()   # baselines are eligible on different #splits
    colors = [FAMILY_COLORS.get(f, "#888") for f in lead["family"]]
    ax.barh(range(len(lead)), lead["pearson_delta"], color=colors, edgecolor="white",
            linewidth=0.6, zorder=2)
    # put #applicable-splits on the y-label (e.g. "FP-ridge (n=2)") — the average spans DIFFERENT split
    # sets per baseline (a chemistry model eligible on 2, simple baselines floored to 1), so the bars
    # are not strictly head-to-head; n disambiguates. (On the left axis → no clash with the legend.)
    ax.set_yticks(range(len(lead)))
    ax.set_yticklabels([f"{b}  (n={int(nsplit.get(b, 0))})" for b in lead["baseline"]])
    ax.set_title("Headline leaderboard")
    ax.set_xlabel("mean Pearson-Δ over applicable splits  ↑")
    ax.axvline(0, color="#333", lw=0.6, zorder=1)
    despine(ax)
    fams = list(dict.fromkeys(lead["family"]))
    ax.legend(handles=[mpatches.Patch(color=FAMILY_COLORS.get(f, "#888"), label=f) for f in fams],
              title="family", loc="lower right", title_fontsize=7)

    for ax, lab in zip(axes.ravel(), "abcd"):
        panel_label(ax, lab)

    # shared split legend + floor key across panels a–c, placed under the figure
    splits = list(dict.fromkeys(summ["split"]))
    handles = [mpatches.Patch(color=SPLIT_COLORS[i % len(SPLIT_COLORS)], label=_short(s, cluster))
               for i, s in enumerate(splits)]
    handles.append(mpatches.Patch(facecolor="#bbbbbb", hatch="////", edgecolor="#888",
                                  label="floored (not headline-eligible)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.04),
               fontsize=7)
    # the panel-(d) leaderboard averages each baseline over the splits where it is headline-eligible,
    # which differ across baselines (n shown) and differ in difficulty — so bars are not strictly
    # head-to-head; the per-split table in the draft is the authoritative comparison.
    fig.text(0.5, -0.075, "Panel d averages each baseline over its applicable splits (n per bar; splits "
             "differ in difficulty) — see the per-split table for head-to-head values.",
             ha="center", fontsize=6.2, color="#666")

    fig.tight_layout(rect=[0, 0.02, 1, 0.99], w_pad=2.0, h_pad=2.4)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")                       # raster (PNG, 350 dpi)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")   # vector (PDF) for submission
    plt.close(fig)
    return out_path


def c1_figure(raw_df: pd.DataFrame, cluster: str, out_path: str | Path) -> Path:
    """C1 cytokine-response — Kang IFN-β cross-cell-type ANCHOR (OnePager Figure 3's lineage-level
    reproduction reference). The full Figure 3 (Oesinghaus 90-cytokine resolution + similarity gradient,
    Cano-Gamez naive→memory state transfer) needs those two datasets, which are not on disk — so this
    renders the Kang anchor honestly: (a) baseline × held-lineage Pearson-Δ heatmap, (b) best-latent vs
    best-simple per lineage (does a conditioned model beat the simple floor?), (c) cell-type difficulty.
    Only LOCT (C1_loct_*) splits → falls back to the generic cluster_figure otherwise."""
    set_pub_style()
    out_path = Path(out_path)
    df = raw_df[raw_df["ran"]].copy()
    loct = df[df["split"].astype(str).str.startswith("C1_loct")]
    if loct.empty:
        return cluster_figure(raw_df, cluster, out_path)             # synthetic / non-Kang → generic
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    lins = sorted(loct["split"].unique())
    lin_lbl = [s.replace("C1_loct_", "").replace("_", " ") for s in lins]
    rows = [b for b in BASELINE_ORDER if b in set(loct["baseline"])] + \
           [b for b in sorted(loct["baseline"].unique()) if b not in BASELINE_ORDER]
    SIMPLE = {"ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"}
    LATENT = [b for b in rows if b not in SIMPLE]

    def _v(b, s):
        r = loct[(loct["baseline"] == b) & (loct["split"] == s)]["pearson_delta"]
        return float(r.mean()) if len(r) else np.nan

    fig = plt.figure(figsize=(13.0, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 0.9], wspace=0.5,
                          left=0.10, right=0.965, top=0.82, bottom=0.2)
    axA, axB, axC = (fig.add_subplot(gs[0, k]) for k in range(3))

    # (a) baseline × lineage Pearson-Δ heatmap (IFN-β is strong → values 0.5–0.9)
    M = np.array([[_v(b, s) for s in lins] for b in rows])
    imA = _heat_clean(axA, M, rows, lin_lbl, cmap="RdBu_r", vmin=-0.9, vmax=0.9, annot=True,
                      bold_rows=("cell-mean", "donor-shift"))
    cax = make_axes_locatable(axA).append_axes("right", size="4%", pad=0.06)
    cb = fig.colorbar(imA, cax=cax, ticks=[-0.9, 0, 0.9]); cb.set_label("Pearson-Δ", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.5)
    axA.set_title("IFN-β response by held-out lineage\n(downstream Pearson-Δ)", fontsize=8.5,
                  weight="bold", pad=6)

    # (b) best-latent vs best-simple per lineage — does a conditioned model beat the simple floor?
    x = np.arange(len(lins)); w = 0.38
    simple_best = [np.nanmax([_v(b, s) for b in SIMPLE if b in rows]) for s in lins]
    latent_best = [np.nanmax([_v(b, s) for b in LATENT]) if LATENT else np.nan for s in lins]
    axB.bar(x - w / 2, simple_best, w, label="best simple", color="#999999", edgecolor="white", lw=0.5)
    axB.bar(x + w / 2, latent_best, w, label="best latent (scGen/CPA)", color="#CC79A7",
            edgecolor="white", lw=0.5)
    axB.set_xticks(x); axB.set_xticklabels(lin_lbl, fontsize=6.5, rotation=30, ha="right")
    axB.set_ylabel("Pearson-Δ  ↑"); axB.set_ylim(0, 1.0)
    axB.legend(fontsize=6.5, frameon=False, loc="lower right")
    axB.set_title("Conditioned model vs simple floor\n(per held-out lineage)", fontsize=8.5,
                  weight="bold", pad=6)
    despine(axB)

    # (c) cell-type difficulty: per-lineage mean Pearson-Δ over headline-eligible baselines (sorted)
    he = loct[loct["headline_eligible"] == True]                     # noqa: E712
    per = he.groupby("split")["pearson_delta"].mean().reindex(lins).sort_values()
    od_lbl = [s.replace("C1_loct_", "").replace("_", " ") for s in per.index]
    axC.barh(range(len(per)), per.values, color="#56B4E9", edgecolor="white", lw=0.5)
    axC.set_yticks(range(len(per))); axC.set_yticklabels(od_lbl, fontsize=7)
    axC.invert_yaxis()                               # hardest (lowest Δ) at top → matches the title
    axC.set_xlabel("mean Pearson-Δ (ranked baselines)  ↑")
    axC.set_title("Cell-type difficulty\n(top = hardest)", fontsize=8.5, weight="bold", pad=6)
    axC.set_xlim(0, 1.0); despine(axC)

    fig.suptitle(f"{cluster}  ·  Cytokine-response — Kang IFN-β cross-cell-type anchor "
                 "(Figure 3 reproduction reference)", fontsize=9.5, fontweight="bold", y=0.96)
    for ax, lab in zip((axA, axB, axC), "abc"):
        ax.text(-0.04, 1.08, lab, transform=ax.transAxes, fontsize=12, fontweight="bold",
                va="bottom", ha="right")
    fig.text(0.5, 0.02, "Kang = the OnePager's lineage-level IFN-β anchor. The full Figure 3 "
             "(Oesinghaus 90-cytokine resolution & similarity gradient; Cano-Gamez naive→memory state "
             "transfer) requires those datasets, not yet on disk. AUCell-Δ omitted: degenerate for a "
             "single seen cytokine (one tiled profile per model → no cross-stratum variance).",
             ha="center", fontsize=6.0, color="#666")
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def c4_figure(raw_df: pd.DataFrame, cluster: str, out_path: str | Path) -> Path:
    """C4 Axis-2 — RNA vs protein modality recoverability of an unseen KO (Frangieh). Two panels:
    (a) per-baseline Pearson-Δ in RNA vs the 20-marker proteome (paired), (b) the distributional
    (energy-distance) view by modality. Falls back to cluster_figure if the modality datasets are absent."""
    set_pub_style()
    out_path = Path(out_path)
    df = raw_df[raw_df["ran"]].copy()
    if not {"frangieh_RNA", "frangieh_protein"}.issubset(set(df.get("dataset", pd.Series([])))):
        return cluster_figure(raw_df, cluster, out_path)
    fifty = df[df["split"].astype(str).str.endswith("_50")]
    bls = [b for b in BASELINE_ORDER if b in set(fifty["baseline"])]
    mods = [("frangieh_RNA", "RNA (2000 genes)", "#0072B2"), ("frangieh_protein", "protein (20 CITE)", "#E69F00")]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 4.2))
    x = np.arange(len(bls)); w = 0.38
    for k, (ds, lab, col) in enumerate(mods):
        vals = [float(fifty[(fifty.dataset == ds) & (fifty.baseline == b)]["pearson_delta"].mean()) for b in bls]
        axA.bar(x + (k - 0.5) * w, vals, w, label=lab, color=col, edgecolor="white", lw=0.5)
    axA.set_xticks(x); axA.set_xticklabels(bls, fontsize=7, rotation=25, ha="right")
    axA.set_ylabel("Pearson-Δ  ↑"); axA.axhline(0, color="#bbb", lw=0.6)
    axA.legend(fontsize=7, frameon=False, loc="upper right")
    axA.set_title("Unseen-KO recoverability by modality\n(leave-one-KO-out, 50%)", fontsize=8.5, weight="bold", pad=6)
    despine(axA)
    for k, (ds, lab, col) in enumerate(mods):
        vals = [float(fifty[(fifty.dataset == ds) & (fifty.baseline == b)]["e_distance"].mean()) for b in bls]
        axB.bar(x + (k - 0.5) * w, vals, w, label=lab, color=col, edgecolor="white", lw=0.5)
    axB.set_xticks(x); axB.set_xticklabels(bls, fontsize=7, rotation=25, ha="right")
    axB.set_ylabel("energy distance  ↓")
    axB.legend(fontsize=7, frameon=False, loc="upper right")
    axB.set_title("Distributional fidelity by modality", fontsize=8.5, weight="bold", pad=6)
    despine(axB)
    fig.suptitle(f"{cluster}  ·  RNA vs protein modality (Frangieh CITE-seq, melanoma CRISPR-KO)",
                 fontsize=9.5, fontweight="bold", y=1.0)
    for ax, lab in zip((axA, axB), "ab"):
        ax.text(-0.05, 1.08, lab, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")
    fig.text(0.5, -0.04, "Same leave-one-KO-out split scored on matched RNA and 20-marker surface-proteome "
             "readouts of the same cells (IFNγ condition). No conditioned model on this axis — simple-only floor.",
             ha="center", fontsize=6.0, color="#666")
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def c5_figure(raw_df: pd.DataFrame, cluster: str, out_path: str | Path) -> Path:
    """C5 / OnePager Figure 7 — small-molecule, 2×2 matching the planning design:
      (a) O1: per-compound Pearson-Δ vs nearest-train Tanimoto distance (chemistry-difficulty axis),
      (b) O1: per-baseline Tanimoto-robustness slope (Δ-per-unit-distance) + R²,
      (c) O2: cross-cell-type LOCT Pearson-Δ heatmap (baseline × lineage),
      (d) O2: cell-type specificity (per-lineage mean Pearson-Δ over ranked baselines; NK hardest).
    Panels a–b need results/C5/tanimoto_percompound.csv (scripts/c5_tanimoto.py); if absent, falls back
    to the generic cluster_figure."""
    set_pub_style()
    out_path = Path(out_path)
    tani_path = out_path.parent / "tanimoto_percompound.csv"
    if not tani_path.exists():
        return cluster_figure(raw_df, cluster, out_path)             # graceful fallback
    tani = pd.read_csv(tani_path)
    df = raw_df[raw_df["ran"]].copy()

    fig = plt.figure(figsize=(12.4, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34, left=0.085, right=0.965, top=0.9, bottom=0.1)
    axA, axB, axC, axD = (fig.add_subplot(gs[i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)])

    # linear fit helper: slope + R² of y on x
    def _fit(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) < 3 or np.ptp(x) < 1e-9:
            return float("nan"), float("nan")
        b1, b0 = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        return float(b1), float(r * r)

    # (a) per-compound Pearson-Δ vs Tanimoto distance, for the chemistry models + the cell-mean floor
    show = [b for b in ["FP-ridge", "CPA", "cell-mean"] if b in set(tani["baseline"])]
    cmap_a = {"FP-ridge": "#0072B2", "CPA": "#D55E00", "cell-mean": "#999999"}
    lbl_a = {"FP-ridge": "FP-ridge", "CPA": "chemCPA", "cell-mean": "cell-mean (floor)"}
    for b in show:
        s = tani[tani["baseline"] == b]
        axA.scatter(s["tanimoto_dist"], s["pearson_delta"], s=16, alpha=0.6,
                    color=cmap_a.get(b, "#444"), edgecolor="none", label=lbl_a.get(b, b), zorder=2)
        b1, r2 = _fit(s["tanimoto_dist"], s["pearson_delta"])
        if b1 == b1:
            xs = np.array([s["tanimoto_dist"].min(), s["tanimoto_dist"].max()])
            b0 = s["pearson_delta"].mean() - b1 * s["tanimoto_dist"].mean()
            axA.plot(xs, b1 * xs + b0, color=cmap_a.get(b, "#444"), lw=1.6, zorder=3)
    axA.axhline(0, color="#bbb", lw=0.6, zorder=0)
    axA.set_xlabel("Tanimoto distance to nearest training compound  →  (chemically far)")
    axA.set_ylabel("per-compound Pearson-Δ  ↑")
    # honest title: in these data the fit is flat (no chemical-distance difficulty effect) — the slope
    # panel (b) quantifies it. Do NOT assert "error rises with distance" (the OnePager mock expected it
    # but the real OP3 result is a null).
    axA.set_title("Per-compound Pearson-Δ vs chemical distance\n(flat — no Tanimoto difficulty effect)",
                  fontsize=8.5, weight="bold", pad=6)
    axA.legend(fontsize=6.5, frameon=False, loc="upper right")
    despine(axA)

    # (b) per-baseline Tanimoto-robustness: slope of Pearson-Δ vs distance (≈0 = distance-insensitive)
    order_b = [b for b in BASELINE_ORDER if b in set(tani["baseline"])] + \
              [b for b in sorted(tani["baseline"].unique()) if b not in BASELINE_ORDER]
    slopes, r2s = {}, {}
    for b in order_b:
        s = tani[tani["baseline"] == b]
        slopes[b], r2s[b] = _fit(s["tanimoto_dist"], s["pearson_delta"])
    order_b = [b for b in order_b if slopes[b] == slopes[b]]
    fam = df.drop_duplicates("baseline").set_index("baseline")["family"].to_dict()
    yvals = [slopes[b] for b in order_b]
    colors_b = [FAMILY_COLORS.get(fam.get(b, ""), "#888") for b in order_b]
    labs = [("chemCPA" if b == "CPA" else b) for b in order_b]
    axB.barh(range(len(order_b)), yvals, color=colors_b, edgecolor="white", linewidth=0.5, zorder=2)
    axB.set_yticks(range(len(order_b))); axB.set_yticklabels(labs, fontsize=7)
    for i, b in enumerate(order_b):
        axB.text(yvals[i] + (0.01 if yvals[i] >= 0 else -0.01), i, f"R²={r2s[b]:.2f}",
                 va="center", ha="left" if yvals[i] >= 0 else "right", fontsize=5.6, color="#555")
    axB.axvline(0, color="#333", lw=0.6, zorder=1)
    axB.set_xlabel("Δ(Pearson-Δ) per unit Tanimoto distance  (slope; <0 = degrades when far)")
    axB.set_title("Per-baseline Tanimoto robustness", fontsize=8.5, weight="bold", pad=6)
    xpad = max(0.02, max(abs(v) for v in yvals) * 0.35)
    axB.set_xlim(min(yvals) - xpad, max(yvals) + xpad); despine(axB)

    # (c) cross-cell-type LOCT heatmap: baseline (rows) × lineage (cols), Pearson-Δ
    loct = df[df["split"].astype(str).str.startswith("C5_loct")]
    lins = sorted(loct["split"].unique())
    lin_lbl = [s.replace("C5_loct_", "").replace("_", " ") for s in lins]
    rows_c = [b for b in BASELINE_ORDER if b in set(loct["baseline"])] + \
             [b for b in sorted(loct["baseline"].unique()) if b not in BASELINE_ORDER]
    M = np.full((len(rows_c), len(lins)), np.nan)
    for i, b in enumerate(rows_c):
        for j, s in enumerate(lins):
            v = loct[(loct["baseline"] == b) & (loct["split"] == s)]["pearson_delta"]
            if len(v):
                M[i, j] = float(v.mean())
    row_c_lbl = [("chemCPA" if b == "CPA" else b) for b in rows_c]
    _heat_clean(axC, M, row_c_lbl, lin_lbl, cmap="RdBu_r", vmin=-0.5, vmax=0.5, annot=True)
    cax = make_axes_locatable_safe(axC)
    if cax is not None:
        cb = fig.colorbar(axC.images[0], cax=cax, ticks=[-0.5, 0, 0.5]); cb.set_label("Pearson-Δ", fontsize=7)
        cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_linewidth(0.5)
    axC.set_title("Cross-cell-type transfer (LOCT)\nbaseline × held-out lineage", fontsize=8.5,
                  weight="bold", pad=6)

    # (d) cell-type specificity: per-lineage mean Pearson-Δ over headline-eligible baselines, NK hardest
    he = loct[loct["headline_eligible"] == True]                     # noqa: E712
    per_lin = he.groupby("split")["pearson_delta"].agg(["mean", "std"]).reindex(lins)
    order_d = per_lin["mean"].sort_values().index                    # hardest (lowest) first
    od_lbl = [s.replace("C5_loct_", "").replace("_", " ") for s in order_d]
    axD.bar(range(len(order_d)), per_lin.loc[order_d, "mean"], yerr=per_lin.loc[order_d, "std"],
            color="#56B4E9", edgecolor="white", linewidth=0.6, capsize=3, zorder=2)
    axD.set_xticks(range(len(order_d))); axD.set_xticklabels(od_lbl, fontsize=7.5)
    axD.set_ylabel("mean Pearson-Δ over ranked baselines  ↑")
    axD.axhline(0, color="#bbb", lw=0.6, zorder=0)
    axD.set_title("Cell-type specificity\n(hardest → most robust lineage)", fontsize=8.5, weight="bold", pad=6)
    if len(order_d):
        axD.annotate(f"{od_lbl[0]} hardest", xy=(0, float(per_lin.loc[order_d[0], "mean"])),
                     xytext=(0.4, float(per_lin["mean"].max()) * 0.9), fontsize=6.4, color="#444",
                     arrowprops=dict(arrowstyle="-|>", color="#888", lw=0.8))
    despine(axD)

    fig.suptitle(f"{cluster}  ·  Small-molecule perturbation prediction (OP3 / Szałata 2024) — "
                 "O1 chemical difficulty (a,b) · O2 cross-cell-type transfer (c,d)",
                 fontsize=9.5, fontweight="bold", y=0.965)
    for ax, lab in zip((axA, axB, axC, axD), "abcd"):
        ax.text(-0.06, 1.06, lab, transform=ax.transAxes, fontsize=12, fontweight="bold",
                va="bottom", ha="right")
    fig.text(0.5, 0.045, "Tanimoto distance is a post-hoc difficulty axis (never a model input). Unseen-"
             "compound prediction is defined only for chemistry-aware models (FP-ridge, chemCPA); CINEMA-OT "
             "is perturbation-agnostic. Per-split values in the draft table; energy-distance/AUCell in Supp S5.",
             ha="center", fontsize=6.1, color="#666")
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_axes_locatable_safe(ax):
    try:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        return make_axes_locatable(ax).append_axes("right", size="4.5%", pad=0.06)
    except Exception:
        return None


def _heat_clean(ax, M, row_labels, col_labels, *, cmap, vmin, vmax, annot=True,
                annot_thresh=0.0, bold_rows=()):
    """Render one publication-grade heatmap cell: white inter-cell separators, no spines/ticks,
    optional value annotations with auto-contrast text. Returns the AxesImage for the colorbar."""
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(M.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(M.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(col_labels, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(row_labels, fontsize=7.5)
    for t, b in zip(ax.get_yticklabels(), row_labels):
        if b in bold_rows:
            t.set_fontweight("bold")
    if annot:
        span = max(vmax - vmin, 1e-9)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v) or abs(v) < annot_thresh:
                    continue
                ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=6.4,
                        color="white" if abs(v) / span > 0.32 else "#1a1a1a")
    return im


def c3_figure(raw_df: pd.DataFrame, cluster: str, out_path: str | Path) -> Path:
    """C3 Q1 (O1), publication-grade: (a) baseline×dataset downstream-only Pearson-Δ heatmap at 50%
    LO-gene; (b) baseline×immune-program AUCell-Δ heatmap (same row order); (c) robustness across
    10/25/50% holdout with the simple cell-mean drawn as the reference floor the conditioned models
    fail to clear. Rows are sorted by 50%-LO Pearson-Δ so the finding reads top-down."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    set_pub_style()
    df = raw_df[raw_df["ran"]].copy()
    s = df.groupby(["dataset", "modality", "split", "baseline"], as_index=False)["pearson_delta"].mean()
    present = set(s["baseline"])
    h50 = s[s["split"].str.endswith("_50")]
    rank_val = h50.groupby("baseline")["pearson_delta"].mean()
    elig = df.drop_duplicates("baseline").set_index("baseline")["headline_eligible"]
    is_elig = {b: bool(elig.get(b, False)) for b in present}
    # row order: headline-eligible block on top (sorted by 50%-LO Pearson-Δ), then a rule, then the
    # NOT-ranked reference block (run_floor / adapted*) also sorted — so a high floor (e.g. CINEMA-OT
    # ≈ donor-shift) can never read as a top method just by sitting high in a performance sort.
    def _byval(bl):
        return sorted(bl, key=lambda b: float(rank_val.get(b, -np.inf)), reverse=True)
    elig_bl = _byval([b for b in BASELINE_ORDER if b in present and is_elig[b]])
    ref_bl = _byval([b for b in BASELINE_ORDER if b in present and not is_elig[b]])
    baselines = elig_bl + ref_bl
    n_elig = len(elig_bl)
    datasets = list(dict.fromkeys(zip(df["dataset"], df["modality"])))
    SIMPLE = {"ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"}
    # tag the rows that are NOT headline-eligible so a reader never mistakes a high floor (e.g.
    # CINEMA-OT ≈ cell-mean) for a winning method: † not-defined floor, * adapted (gene-conditioned).
    _TAG = {"CINEMA-OT": "CINEMA-OT †", "CellOT": "CellOT †", "UCE": "UCE †",
            "scGen": "scGen *", "CPA": "CPA *"}
    row_lab = [_TAG.get(b, b) for b in baselines]

    fig = plt.figure(figsize=(13.6, 5.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 0.96, 1.12], wspace=0.62,
                          left=0.10, right=0.975, top=0.85, bottom=0.22)
    axA, axB, axC = (fig.add_subplot(gs[0, k]) for k in range(3))
    ds_lab = [f"{d.replace('mccutcheon_', 'McC ').replace('_', ' ')}\n({m})" for d, m in datasets]

    # (a) Pearson-Δ heatmap (50% LO-gene), rows sorted by performance
    M = np.full((len(baselines), len(datasets)), np.nan)
    for j, (ds, _m) in enumerate(datasets):
        for i, b in enumerate(baselines):
            v = h50[(h50["dataset"] == ds) & (h50["baseline"] == b)]["pearson_delta"]
            if len(v):
                M[i, j] = v.iloc[0]
    imA = _heat_clean(axA, M, row_lab, ds_lab, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                      annot=True, bold_rows=("cell-mean", "donor-shift"))
    caxA = make_axes_locatable(axA).append_axes("right", size="4.5%", pad=0.06)
    cbA = fig.colorbar(imA, cax=caxA, ticks=[-0.5, 0, 0.5]); cbA.set_label("Pearson-Δ", fontsize=7)
    cbA.ax.tick_params(labelsize=6.5, length=2); cbA.outline.set_linewidth(0.5)
    axA.set_title("Unseen-gene prediction\n(downstream-only Pearson-Δ, 50% LO)", fontsize=8.5,
                  weight="bold", pad=7)

    # (b) immune-program AUCell-Δ heatmap (same row order); near-0 for constant-profile baselines
    pcols = sorted(c for c in df.columns if c.startswith("aucell::"))
    df50 = df[df["split"].str.endswith("_50")]
    _PROG_ABBR = {"IL2_STAT5": "IL2/STAT5", "TCR_activation": "TCR act.",
                  "Treg_exhaustion": "Treg exh.", "effector_cytokine": "eff. cyto.",
                  "proliferation": "prolif."}
    progs = [_PROG_ABBR.get(c.split("::", 1)[1], c.split("::", 1)[1].replace("_", " ")) for c in pcols]
    Mp = np.array([[df50[df50["baseline"] == b][c].mean() if (df50["baseline"] == b).any() else np.nan
                    for c in pcols] for b in baselines]) if (pcols and baselines) else np.zeros((0, 0))
    vmaxB = max(0.05, float(np.nanmax(np.abs(np.nan_to_num(Mp)))) if Mp.size else 0.05)
    # no per-cell numbers here (unlike the headline panel a): (b)'s message is the engagement PATTERN
    # — mostly ≈0, a few faint cells — so colour + colorbar carry it consistently; exact values live in
    # the AUCell-Δ table (Supp Table S3). Annotating only large cells looked arbitrary.
    imB = _heat_clean(axB, Mp, baselines, progs, cmap="PuOr_r", vmin=-vmaxB, vmax=vmaxB, annot=False)
    axB.set_yticklabels([])                                   # rows identical to (a), shown there
    caxB = make_axes_locatable(axB).append_axes("right", size="4.5%", pad=0.06)
    cbB = fig.colorbar(imB, cax=caxB, ticks=[-round(vmaxB, 2), 0, round(vmaxB, 2)])
    cbB.set_label("AUCell-Δ", fontsize=7); cbB.ax.tick_params(labelsize=6.5, length=2)
    cbB.outline.set_linewidth(0.5)
    axB.set_title("Immune-program engagement\n(AUCell-Δ, 50% LO)", fontsize=8.5, weight="bold", pad=7)
    if 0 < n_elig < len(baselines):              # rule: ranked block (above) vs reference block (below)
        for _ax in (axA, axB):
            _ax.axhline(n_elig - 0.5, color="#222", lw=1.4, zorder=6)

    # (c) robustness — cell-mean is the reference floor; conditioned models sit below it
    fracs = [(10, "_10"), (25, "_25"), (50, "_50")]
    def _curve(b):
        xs, ys = [], []
        for pct, suf in fracs:
            v = s[(s["baseline"] == b) & (s["split"].str.endswith(suf))]["pearson_delta"]
            if len(v):
                xs.append(pct); ys.append(float(v.mean()))
        return xs, ys
    MODEL_COLORS = {"GEARS": "#009E73", "AttentionPert": "#117733", "scGPT": "#E69F00",
                    "scGen": "#CC79A7", "CPA": "#AA4499", "STATE": "#D55E00"}
    models = [b for b in baselines if b in MODEL_COLORS]
    handles = []
    # control floor (≈0) and linear-PCA (simple shrinkage) as light grey references
    if "ctrl-pred" in present:
        xs, ys = _curve("ctrl-pred"); h, = axC.plot(xs, ys, ls=":", lw=1.2, color="#bdbdbd",
                                                    marker="", zorder=1, label="ctrl-pred (no effect)")
        handles.append(h)
    if "linear-PCA" in present:
        xs, ys = _curve("linear-PCA"); h, = axC.plot(xs, ys, ls="-", lw=1.4, color="#9e9e9e",
                                                    marker="o", ms=3, zorder=2, label="linear-PCA (simple)")
        handles.append(h)
    if "CINEMA-OT" in present:
        xs, ys = _curve("CINEMA-OT"); h, = axC.plot(xs, ys, ls="--", lw=1.4, color="#4C72B0",
                                                    marker="^", ms=3.5, zorder=2, label="CINEMA-OT (OT floor†)")
        handles.append(h)
    for b in models:                                          # conditioned models under test
        xs, ys = _curve(b); h, = axC.plot(xs, ys, ls="-", lw=1.5, color=MODEL_COLORS[b],
                                          marker="o", ms=3.5, zorder=3, label=b)
        handles.append(h)
    # the reference floor LAST and boldest: cell-mean ≡ donor-shift
    xs, ys = _curve("cell-mean")
    hcm, = axC.plot(xs, ys, ls="-", lw=2.6, color="#111111", marker="o", ms=4.5, zorder=5,
                    label="cell-mean ≡ donor-shift (floor)")
    ytop = max(ys) * 1.12
    axC.set_ylim(min(-0.06, axC.get_ylim()[0]), ytop)
    axC.fill_between(xs, ys, ytop, color="#111111", alpha=0.05, zorder=0)
    axC.annotate("conditioned models stay\nbelow the simple floor", xy=(47, ys[2]),
                 xytext=(30.5, ytop * 0.62), fontsize=6.6, color="#444", ha="center", va="center",
                 arrowprops=dict(arrowstyle="-|>", color="#999", lw=0.8,
                                 connectionstyle="arc3,rad=-0.25"))
    axC.axhline(0, color="#bbb", lw=0.6, zorder=0)
    axC.set_xticks([10, 25, 50]); axC.set_xlim(6, 54)
    axC.set_xlabel("held-out target genes (%)"); axC.set_ylabel("mean Pearson-Δ across datasets  ↑")
    axC.set_title("Robustness to holdout fraction", fontsize=8.5, weight="bold", pad=7)
    despine(axC)

    # one shared legend for panel (c)'s curves, below the panels (heatmap rows are self-labeled)
    order = [hcm] + handles
    fig.legend(handles=order, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=5, fontsize=7, frameon=False, handlelength=1.8, columnspacing=1.5)
    fig.text(0.5, 0.005, "Rows above the rule in (a, b) are headline-ranked; below it are reference "
             "methods, NOT ranked:  * adapted (gene-embedding-conditioned)   † not-defined for an unseen "
             "gene (no per-gene transport mechanism / no decoder).  CINEMA-OT† is perturbation-agnostic "
             "(one global OT shift for every gene) ≈ donor-shift.", ha="center", fontsize=6.1, color="#666")
    fig.suptitle(f"{cluster}  ·  Gene-intervention prediction in primary human T cells "
                 "(true leave-one-gene-out)", fontsize=10, fontweight="bold", y=0.975)
    for ax, lab in zip((axA, axB, axC), "abc"):
        ax.text(-0.02, 1.10, lab, transform=ax.transAxes, fontsize=12, fontweight="bold",
                va="bottom", ha="right")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path
