#!/usr/bin/env python
"""Figure 6: donor-axis evidence after adding Soskic LODO.

Outputs results/_paper/figure_donor_decision.{png,pdf}. Panel a keeps the Kang random-vs-LODO
optimism control; panel b is the Soskic per-donor forest (scGen / CellOT minus the pre-specified
primary baseline); panel c summarizes axis-level conditioned-minus-baseline decisions.

Redesigned to the Nature-publication bar: restrained navy editorial palette, generous whitespace,
shared panel grammar, direct inline labelling (no parked decode-legend), and tidy number
formatting with no negative-zero artefacts. Every plotted value is identical to the prior version.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ivcbench.report.style import (
    set_pub_style, despine, panel_title, INK, GREY_MID,
    NAVY, NAVY_DARK, SLATE_BAND, NAVY_RAMP,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = RESULTS / "_paper"

# --- navy editorial roles for this figure (one concept = one colour) -----------------------------
POS = NAVY            # conditioning helps / above floor (the author's signature navy)
NEG = SLATE_BAND      # conditioning does not help / below floor — the one calm slate-grey
NULL = "#9DB8CC"      # light navy fill for the null point-cloud (scGen, latent) — muted, recedes
DOT_NULL = "#7F9CB4"  # slightly deeper edge of the same hue for the null dots


def donor_inflation() -> dict[str, np.ndarray]:
    d1 = pd.read_csv(RESULTS / "C1" / "results_raw.csv")
    d1 = d1[d1.ran == True]  # noqa: E712
    lo = d1[d1.split.str.startswith("C1_lodo")].copy()
    rn = d1[d1.split.str.startswith("C1_randsplit")].copy()
    lo["fold"] = lo.split.str.replace("C1_lodo_", "", regex=False)
    rn["fold"] = rn.split.str.replace("C1_randsplit_f", "", regex=False)
    out = {}
    for b in ["cell-mean", "donor-shift", "linear-PCA"]:
        m = (lo[lo.baseline == b][["fold", "pearson_delta"]]
             .merge(rn[rn.baseline == b][["fold", "pearson_delta"]], on="fold",
                    suffixes=("_lo", "_rn")))
        if len(m):
            out[b] = (m.pearson_delta_rn - m.pearson_delta_lo).to_numpy(float)
    return out


def kang_cluster_summary(infl: dict[str, np.ndarray]) -> dict:
    p = PAPER / "defensive_stats.json"
    if p.exists():
        js = json.loads(p.read_text())["donor_inflation"]
        return dict(mean=js["mean"], lo=js["cluster_ci"][0], hi=js["cluster_ci"][1],
                    pos=js["boot_pos_frac"], n=js["n_donors"])
    per = np.vstack([infl[k] for k in infl]).mean(0)
    rng = np.random.default_rng(0)
    boot = np.array([per[rng.integers(0, len(per), len(per))].mean() for _ in range(10000)])
    return dict(mean=float(per.mean()), lo=float(np.percentile(boot, 2.5)),
                hi=float(np.percentile(boot, 97.5)), pos=float((boot > 0).mean()), n=len(per))


def soskic_forest() -> tuple[pd.DataFrame, dict]:
    forest = ROOT / "figures" / "source_data" / "soskic_donor_forest.csv"
    summ = RESULTS / "C2" / "soskic_donor_bootstrap_summary.csv"
    if not forest.exists() or not summ.exists():
        raise SystemExit("Missing Soskic source data. Run scripts/soskic_donor_postprocess.py first.")
    f = pd.read_csv(forest)
    s = pd.read_csv(summ)
    pear = s[s.metric == "pearson_delta"].iloc[0].to_dict()
    return f, pear


def cellot_soskic() -> tuple[pd.DataFrame, dict]:
    """CellOT per-donor delta (vs the same primary baseline) and its donor-bootstrap summary."""
    raw = ROOT / "outputs" / "additional_models" / "cellot_soskic_raw.csv"
    summ = ROOT / "outputs" / "additional_models" / "cellot_summary.csv"
    d = pd.read_csv(raw)
    d = d[d.metric == "pearson_delta"][["donor", "delta_vs_primary"]].rename(
        columns={"delta_vs_primary": "delta"})
    s = pd.read_csv(summ)
    r = s[(s.dataset == "soskic2022") & (s.metric == "pearson_delta")].iloc[0].to_dict()
    co_s = dict(mean=float(r["delta_vs_primary_baseline"]), lo=float(r["CI_low"]),
                hi=float(r["CI_high"]), pct_positive=float(r["percent_positive_units"]) / 100.0,
                n=int(r["n_units"]))
    return d, co_s


_MINUS = "−"  # true unicode minus


def fmt(v: float, nd: int = 3) -> str:
    """Signed fixed-point with a true minus sign and NO negative-zero artefact.

    A value that rounds to zero is always printed as a clean unsigned ``0.00`` — never the
    ``+0.00``/``−0.00`` tokens a naive ``{:+.2f}`` would emit for tiny ±values.
    """
    r = round(float(v), nd)
    if r == 0.0:
        return f"{0.0:.{nd}f}"           # e.g. "0.000" — unsigned, no spurious + / −
    return f"{r:+.{nd}f}".replace("-", _MINUS)


def main() -> None:
    set_pub_style()
    PAPER.mkdir(parents=True, exist_ok=True)
    infl = donor_inflation()
    kang = kang_cluster_summary(infl)
    forest, sos = soskic_forest()
    co_f, co_s = cellot_soskic()

    fig = plt.figure(figsize=(7.2, 5.4))
    # Balanced grid: the dense ECDF panel (b) gets the larger width share so the sparse strip-plot
    # (a) does not look half-empty next to it; a normal inter-row gutter (hspace) replaces the old
    # full-width white cavity so a, b, c read as one tidy plate.
    gs = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.5], height_ratios=[1.0, 0.92],
                          left=0.105, right=0.965, top=0.91, bottom=0.115,
                          hspace=0.34, wspace=0.34)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    # (a) Kang random-split optimism control --------------------------------------------------
    order = list(infl)
    rng = np.random.default_rng(1)
    for i, b in enumerate(order):
        pts = infl[b]
        x = i + (rng.random(len(pts)) - 0.5) * 0.22
        axA.scatter(x, pts, s=17, color=GREY_MID, edgecolor="white", linewidth=0.4, alpha=0.80,
                    zorder=2)
        axA.plot([i - 0.24, i + 0.24], [pts.mean(), pts.mean()], color=INK, lw=1.6, zorder=3,
                 solid_capstyle="round")
    axA.axhline(0, color=GREY_MID, lw=0.7, zorder=1)
    axA.set_xticks(range(len(order)))
    axA.set_xticklabels(["cell\nmean", "donor\nshift", "linear\nPCA"], fontsize=7)
    axA.set_xlim(-0.55, len(order) - 0.45)
    axA.set_ylabel("random " + _MINUS + " LODO Pearson-Δ", fontsize=8)
    panel_title(axA, "a", "Kang random-split optimism control",
                sub="random split inflates donor transfer", x_letter=-0.20)
    # Tight y-limits so the strip cloud fills the frame (no empty upper-right / bottom band),
    # with just enough headroom for the 3-line stats box to sit in clear white space.
    a_all = np.concatenate([infl[k] for k in order])
    a_lo, a_hi = float(a_all.min()), float(a_all.max())
    axA.set_ylim(a_lo - 0.006, a_hi + 0.030)
    axA.text(0.03, 0.975,
             f"bootstrap mean {fmt(kang['mean'], 3)}\n"
             f"95% CI [{fmt(kang['lo'], 3)}, {fmt(kang['hi'], 3)}]\n"
             f"n = {kang['n']} donors",
             transform=axA.transAxes, va="top", ha="left", fontsize=6.8, color=INK,
             linespacing=1.45,
             bbox=dict(fc="white", ec="#cccccc", lw=0.6, boxstyle="round,pad=0.42"))
    despine(axA)

    # (b) Soskic donor LODO: latent-shift scGen (below floor) vs opt-transport CellOT (above) --
    sg = np.sort(forest["delta_conditioned_minus_primary"].to_numpy())
    co = np.sort(co_f["delta"].to_numpy())
    ysg = np.arange(len(sg))
    yco = np.arange(len(co))
    axB.scatter(sg, ysg, s=11, color=NULL, edgecolor=DOT_NULL, linewidth=0.25, alpha=0.95, zorder=2)
    axB.scatter(co, yco, s=11, color=POS, edgecolor="white", linewidth=0.25, alpha=0.95, zorder=3)
    axB.axvline(0, color=INK, lw=0.9, zorder=1)
    axB.axvline(sos["mean"], color=GREY_MID, lw=1.3, ls=(0, (4, 2)), zorder=1)
    axB.axvline(co_s["mean"], color=NAVY_DARK, lw=1.5, zorder=4)
    axB.set_ylim(-3, max(len(sg), len(co)) + 9)
    # Generous interior padding so the extreme navy CellOT point never touches the right margin
    # and the 'CellOT' inline label sits fully inside the axes with clearance from the spines.
    b_lo = float(min(sg.min(), co.min()))
    b_hi = float(max(sg.max(), co.max()))
    b_pad = 0.06 * (b_hi - b_lo)
    axB.set_xlim(b_lo - b_pad, b_hi + 0.16 * (b_hi - b_lo))
    axB.set_yticks([])
    axB.set_xlabel("conditioned " + _MINUS + " primary baseline Pearson-Δ", fontsize=8)
    panel_title(axB, "b", "Soskic 0h → 16h leave-one-donor-out",
                sub="optimal transport beats the latent shift", x_letter=-0.075)
    # Direct inline labelling — each point cloud is named at its own crest, so no parked
    # decode-legend is needed (matches the reference plate's direct-labelling grammar). Each
    # label sits in clear white space and its leader points to the cloud, never crossing the
    # navy CellOT mean line.
    n = len(sg)
    # Direct labelling at each cloud's own side: CellOT named over its navy cloud (upper-right
    # open space, short leader straight down to a high navy point — never crossing the other
    # cloud or a mean line); scGen named over its light cloud (left).
    axB.annotate("CellOT (optimal transport)", xy=(co[int(0.86 * (n - 1))], 0.86 * (n - 1)),
                 xytext=(0.985, 0.95), textcoords="axes fraction", fontsize=6.8, color=NAVY_DARK,
                 weight="bold", ha="right", va="center",
                 arrowprops=dict(arrowstyle="-", color=NAVY_DARK, lw=0.7,
                                 shrinkA=2, shrinkB=3))
    axB.annotate("scGen (latent shift)", xy=(sg[int(0.30 * (n - 1))], 0.30 * (n - 1)),
                 xytext=(0.045, 0.50), textcoords="axes fraction", fontsize=6.8, color=DOT_NULL,
                 weight="bold", ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=DOT_NULL, lw=0.7,
                                 shrinkA=2, shrinkB=2))
    axB.text(0.97, 0.045,
             f"n = {int(sos['n'])} donors\n"
             f"scGen {fmt(sos['mean'], 3)}  ({100 * sos['pct_positive']:.0f}% positive)\n"
             f"CellOT {fmt(co_s['mean'], 3)}  ({100 * co_s['pct_positive']:.0f}% positive)",
             transform=axB.transAxes, va="bottom", ha="right", fontsize=6.8, color=INK, zorder=6,
             linespacing=1.45,
             bbox=dict(fc="white", ec="#cccccc", lw=0.6, boxstyle="round,pad=0.42"))
    despine(axB)

    # (c) Axis-level decision summary ---------------------------------------------------------
    labels = ["cell-context\n(OP3)", "perturbation\n(CRISPR)", "modality\n(CITE protein)",
              "donor\n(Soskic, CellOT)"]
    vals = [0.119, -0.241, -0.590, float(co_s["mean"])]
    lo = [0.10, -0.29, np.nan, float(co_s["lo"])]
    hi = [0.14, -0.19, np.nan, float(co_s["hi"])]
    x = np.arange(len(labels))
    colors = [POS if v > 0 else NEG for v in vals]
    axC.axhline(0, color=INK, lw=0.9, zorder=3)
    axC.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6, width=0.56, zorder=2)
    for xi, v, l, h in zip(x, vals, lo, hi):
        has_ci = np.isfinite(l) and np.isfinite(h)
        if has_ci:
            axC.plot([xi, xi], [l, h], color=INK, lw=1.1, zorder=4, solid_capstyle="round")
            axC.plot([xi - 0.07, xi + 0.07], [l, l], color=INK, lw=1.1, zorder=4)
            axC.plot([xi - 0.07, xi + 0.07], [h, h], color=INK, lw=1.1, zorder=4)
        # anchor the value label clear of the whole error bar (not just the bar top),
        # so a wide CI whisker never strikes through the label (seen: perturbation −0.241)
        if v >= 0:
            ytext, va = (h if has_ci else v) + 0.035, "bottom"
        else:
            ytext, va = (l if has_ci else v) - 0.035, "top"
        axC.text(xi, ytext, fmt(v, 3), ha="center", va=va, fontsize=7.2, weight="bold", color=INK)
    axC.set_xticks(x)
    axC.set_xticklabels(labels, fontsize=7)
    axC.set_xlim(-0.62, len(labels) - 0.38)
    axC.set_ylabel("conditioned " + _MINUS + " primary baseline", fontsize=8)
    panel_title(axC, "c", "Axis-level decisions use biological-unit resampling",
                sub="navy = conditioning helps;  slate = simpler baseline wins", x_letter=-0.082)
    # Tight y-limits: just enough room for the value labels above the tallest (+0.119) bar and
    # below the deepest (−0.590) bar, so c does not carry large empty headroom and stops dwarfing
    # the top row.
    c_top = max(vals + [v for v in hi if np.isfinite(v)])
    c_bot = min(vals + [v for v in lo if np.isfinite(v)])
    axC.set_ylim(c_bot - 0.085, c_top + 0.075)
    despine(axC)

    out = PAPER / "figure_donor_decision.png"
    fig.savefig(out, dpi=400, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
