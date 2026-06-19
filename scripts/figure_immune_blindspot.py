#!/usr/bin/env python
"""Figure 7 — Immune blind-spot map (v2 §5), the immune-specific figure for the restructured Results.

Four pre-specified immune blind spots, all reported descriptively from the deposited per-unit tables
(`results/_paper/immune_novelty/T1`–`T3`, `results/C3/program_null.csv`, `results/C5/results_raw.csv`):

(a) Per-surface-marker recovery (C4 Frangieh CITE, cell-mean global protein shift,
    `immune_novelty/T1_C4_per_marker_protein_recovery.csv`): the checkpoint RECEPTOR PD-1 (CD279) is the
    single largest, best-recovered marker (sign-match ~0.99, effect-z ~1.9); its LIGAND PD-L1 (CD274) is
    near-zero and SIGN-UNSTABLE across holdouts (sign-match ~0.46 ≈ chance) — the corrected reading is
    "no stable surface effect to recover", NOT "wrong-sign reversal".

(b) Per-immune-program AUCell map (`immune_novelty/T2_per_program_AUCell_map.csv` +
    `results/C3/program_null.csv`): of the curated immune programs, ONLY type-I IFN transfers
    (C5 cell-context AUCell-Δ corr 0.79, recovered); inflammatory-NF-κB is weak, effector-lymphocyte is NO,
    and all five C3 T-cell programs sit at the permutation null.

(c) Direction ≠ magnitude (`results/C5/results_raw.csv`, C5 LOCT): AUCell rank concordance on the type-I IFN
    gene set (x) vs response-magnitude Pearson-Δ (y) per family — the conditioned families order the program
    correctly (~0.75–0.79) yet only the chemistry-side FP-ridge clears the magnitude floor (0.27 → 0.39);
    latent/hybrid recover the direction but not the magnitude.

(d) Per-lineage predictability (`immune_novelty/T3_per_lineage_predictability.csv`): on C1 (Kang) scGen beats
    the best simple baseline on ONLY ONE of eight lineages — CD14⁺ monocytes (single-draw gap +0.098), whose
    three-seed mean (+0.065, n=3 t-CI crossing zero, from `multiseed_scgen_summary.csv`) shows the win is
    under-powered; on C5 (OP3) the chemistry FP-ridge wins on ALL FOUR coarse lineages.

Layout: 2×2. (a) top-left, (b) top-right, (c) bottom-left, (d) bottom-right. No numeric value is hardcoded;
every datum is read from the deposited CSVs. Mirrors scripts/figure_cellcontext.py conventions; rebuilt via
scripts/normalize_plate.py.

Nature-grade redesign: generous whitespace, the shared navy editorial system, row-groups labelled DIRECTLY
with coloured accent bars (no tall decode-legend), one tidy bottom legend per panel, clean Liberation-Sans
typography, and number formatting with no negative-zero artefacts. Every plotted value is identical to the
prior version — only composition, layout, colour, spacing, labels and typography change.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ivcbench.report.style import (set_pub_style, despine, panel_title, style_legend,  # noqa: E402
                                   CONDITIONED, CONDITIONED_DARK, SIMPLE_GREY,
                                   CHEMISTRY, HYBRID, INK as TOK_INK, CITE_COLORS,
                                   NAVY, NAVY_DARK, GREY_MID, GREY_LITE, NULL_GREY, LEGEND_EC)

RESULTS = Path(__file__).resolve().parents[1] / "results"
SIMPLE = ["ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"]

# --- semantic palette, consumed from the shared style tokens (one colour = one concept, plate-wide) ---
C_FLOOR = SIMPLE_GREY          # simple-floor family — neutral grey
C_COND = CONDITIONED           # conditioned-deep / latent family (scGen, CPA) — purple
C_COND_DARK = CONDITIONED_DARK # legible-on-white conditioned text/callout
C_STATE = HYBRID               # hybrid family (STATE) — green-teal
C_WIN = CHEMISTRY              # FP-ridge, the chemistry-side winner — sky-blue
C_NULL = NULL_GREY             # permutation/structural null — background grey
C_INK = TOK_INK                # near-black axis/annotation text
C_RECEPTOR = CITE_COLORS["PD-1"]   # PD-1 (CD279), the recovered checkpoint receptor — teal-green
C_LIGAND = CITE_COLORS["PD-L1"]    # PD-L1 (CD274), the un-recovered checkpoint ligand — rose

MINUS = "−"               # U+2212 true unicode minus
FS_LAB = 6.4                   # axis tick / row label size
FS_AXTITLE = 6.8               # axis-title size
FS_NOTE = 6.0                  # ONE size for ALL on-plot text at the annotation tier: explanatory
                               # callouts AND inline data-point / model-name labels (so no same-level
                               # label towers over another — FP-ridge no longer larger than scGen, etc.)
FS_GROUP = 6.2                 # accent-bar group-label size
FS_LEGEND = 5.7                # per-panel bottom-legend text — one size across all panels


def _fmt(v, nd=2):
    """Fixed-decimal format with a true unicode minus and NO negative-zero token (−0.00 -> 0.00)."""
    s = f"{v:.{nd}f}"
    if float(s) == 0.0:                       # collapse any signed/negative zero to a clean 0.00
        s = f"{0.0:.{nd}f}"
    return s.replace("-", MINUS)


def _accent_bar(ax, y0, y1, color, label, *, x, lw=3.4, label_color=None):
    """A short vertical coloured accent bar in the left gutter that labels a contiguous row-group
    DIRECTLY (the reference-figure grammar), replacing a separate colour-decode legend. `x` is in
    axes-fraction units (negative = left of the spine) so it clears the y-tick labels."""
    label_color = label_color or color
    ax.plot([x, x], [y0, y1], color=color, lw=lw, solid_capstyle="round",
            transform=ax.get_yaxis_transform(), clip_on=False, zorder=5)
    ax.text(x - 0.020, (y0 + y1) / 2, label, transform=ax.get_yaxis_transform(),
            rotation=90, ha="right", va="center", fontsize=FS_GROUP, color=label_color,
            fontweight="bold", clip_on=False, zorder=5)


def main():
    set_pub_style()
    # 2×2 plate. Wider design canvas + generous inter-panel gaps so every panel breathes and nothing
    # runs to the edge; near-unity shrink keeps body text legible at print size.
    fig = plt.figure(figsize=(7.4, 7.0))
    # Symmetric outer margins (left/right) and a WIDE central gutter so the two left panels'
    # right content and the two right panels' left content (accent bars + tick labels) keep
    # equal whitespace from the figure midline — no panel hugs the centre.
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0],
                          wspace=0.78, hspace=0.66, left=0.175, right=0.955, top=0.895, bottom=0.090)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # ===================== (a) per-surface-marker protein recovery (C4 CITE) =====================
    # T1 holds two holdout fractions (25%, 50%) per marker; use the 50% holdout (the larger-n, more
    # stringent draw) as the per-marker point, exactly as deposited. Lollipop of obsΔ (observed surface
    # delta) with the predicted cell-mean shift overlaid; markers are ranked by |obsΔ| so the strongly-
    # perturbed checkpoint receptor PD-1 sits at the top and the near-zero/sign-unstable cluster sinks.
    t1 = pd.read_csv(RESULTS / "_paper" / "immune_novelty" / "T1_C4_per_marker_protein_recovery.csv")
    t1 = t1[t1.held_frac_pct == 50].copy()
    t1["lab"] = t1["alias"].map(lambda a: a.split(" (")[0])
    t1["absobs"] = t1["obsDelta_mean"].abs()
    t1 = t1.sort_values("absobs").reset_index(drop=True)   # smallest |obsΔ| at bottom, largest at top
    yA = np.arange(len(t1))

    def mcolor(lab):
        if lab == "PD-1":
            return C_RECEPTOR
        if lab == "PD-L1":
            return C_LIGAND
        return C_FLOOR

    axA.axvline(0.0, color=GREY_MID, lw=0.6, zorder=1)
    for yi, r in zip(yA, t1.itertuples()):
        col = mcolor(r.lab)
        is_chk = r.lab in ("PD-1", "PD-L1")
        axA.plot([0.0, r.obsDelta_mean], [yi, yi], color=col, lw=2.2 if is_chk else 1.2,
                 alpha=0.95 if is_chk else 0.5, solid_capstyle="round", zorder=2)
        axA.plot(r.obsDelta_mean, yi, "o", color=col, ms=5.2 if is_chk else 3.2, mec="white",
                 mew=0.7, zorder=4)
        axA.plot(r.predDelta, yi, "D", color="white", mec=col, mew=1.0,
                 ms=4.4 if is_chk else 2.8, zorder=3)

    # the un-recoverable cluster (sign-match ≈ chance) flagged with ONE faint band at the bottom rows
    no_rows = [i for i, r in enumerate(t1.itertuples()) if str(r.recoverable).startswith("NO")]
    if no_rows:
        axA.axhspan(min(no_rows) - 0.5, max(no_rows) + 0.5, color=C_NULL, alpha=0.14, zorder=0)

    axA.set_yticks(yA)
    axA.set_yticklabels(t1["lab"], fontsize=FS_LAB)
    for tick, lab in zip(axA.get_yticklabels(), t1["lab"]):
        if lab == "PD-1":
            tick.set_color(C_RECEPTOR); tick.set_fontweight("bold")
        elif lab == "PD-L1":
            tick.set_color(C_LIGAND); tick.set_fontweight("bold")
    axA.set_xlim(-0.62, 0.42)
    axA.set_ylim(-0.9, len(t1) - 0.4)
    axA.set_xlabel(r"surface-protein response  (obs $\Delta$, CLR)", fontsize=FS_AXTITLE, color=C_INK)
    xa = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4]
    axA.set_xticks(xa)
    axA.set_xticklabels([_fmt(t, 1) for t in xa], fontsize=FS_LAB)
    axA.tick_params(colors=C_INK, labelsize=FS_LAB)
    despine(axA)

    # direct, short callouts on the two checkpoint partners (the load-bearing contrast) — no boxes,
    # no bent leaders; parked in the open whitespace adjacent to each marker.
    rec = t1[t1.lab == "PD-1"].iloc[0]
    lig = t1[t1.lab == "PD-L1"].iloc[0]
    yrec = int(np.where(t1["lab"].values == "PD-1")[0][0])
    ylig = int(np.where(t1["lab"].values == "PD-L1")[0][0])
    # Both callouts sit FULLY inside the data box (within xlim −0.62..0.42) and clear of the spines
    # and the y-tick labels; regular weight so colour alone carries the emphasis (title > data > note).
    axA.annotate(f"receptor recovered\nsign-match {rec.sign_match_frac:.2f}",
                 xy=(rec.obsDelta_mean, yrec), xytext=(0.36, yrec - 1.25),
                 fontsize=FS_NOTE, color=C_RECEPTOR, ha="right", va="center", weight="normal",
                 linespacing=1.05, zorder=6,
                 arrowprops=dict(arrowstyle="-", color=C_RECEPTOR, lw=0.5,
                                 connectionstyle="angle,angleA=0,angleB=-90,rad=3",
                                 shrinkA=1.0, shrinkB=3.0))
    axA.annotate(f"ligand near-zero,\nsign-unstable ({lig.sign_match_frac:.2f})",
                 xy=(lig.obsDelta_mean, ylig), xytext=(0.025, ylig - 2.05),
                 fontsize=FS_NOTE, color=C_LIGAND, ha="left", va="center", weight="normal",
                 linespacing=1.05, zorder=6,
                 arrowprops=dict(arrowstyle="-", color=C_LIGAND, lw=0.5,
                                 connectionstyle="angle,angleA=0,angleB=90,rad=3",
                                 shrinkA=2.0, shrinkB=3.0))

    leg_a = [Line2D([0], [0], marker="o", color=C_FLOOR, lw=0, ms=4.2, mec="white", mew=0.7,
                    label="observed Δ"),
             Line2D([0], [0], marker="D", color="white", mec=C_FLOOR, mew=1.0, lw=0, ms=3.8,
                    label="predicted shift")]
    lega = axA.legend(handles=leg_a, fontsize=FS_LEGEND, frameon=False, loc="upper center",
                      bbox_to_anchor=(0.5, -0.150), ncol=2, handletextpad=0.4, columnspacing=1.4,
                      borderaxespad=0.0)
    lega.set_zorder(8)

    # ===================== (b) per-immune-program AUCell map =====================
    # T2 holds the per-program best-family AUCell-Δ recovery (C5 cell-context + C3 unseen-pert), and
    # program_null.csv certifies the C3 programs sit at a permutation null. ONE ordered lollipop with the
    # two split blocks contiguous (C5 on top, C3 below); colour carries recovery status, a left accent
    # bar labels each split DIRECTLY (no decode-legend), and a faint null band marks |corr|≈0.
    t2 = pd.read_csv(RESULTS / "_paper" / "immune_novelty" / "T2_per_program_AUCell_map.csv")
    t2 = t2[t2["best_corr"].notna()].copy()          # drop the C1 NOT-ESTIMABLE row (no numeric corr)
    PROG_LABEL = {"type_I_IFN": "type-I IFN", "inflammatory_NFkB": "inflammatory NF-κB",
                  "effector_lymphocyte": "effector lymphocyte", "TCR_activation": "TCR activation",
                  "IL2_STAT5": "IL2–STAT5", "proliferation": "proliferation",
                  "effector_cytokine": "effector cytokine", "Treg_exhaustion": "Treg / exhaustion"}
    t2["plab"] = t2["program"].map(lambda p: PROG_LABEL.get(p, p))

    blocks = []
    for clu in ("C5", "C3"):
        sub = t2[t2["cluster"] == clu].sort_values("best_corr", ascending=True)
        blocks.append((clu, sub))
    ordered = []
    y = 0.0
    yblock = {}
    for clu, sub in reversed(blocks):                 # C3 first at the bottom, then C5 on top
        ys = []
        for r in sub.itertuples():
            ordered.append((y, r)); ys.append(y); y += 1.0
        yblock[clu] = ys
        y += 1.1                                       # block gap

    def prog_color(stat):
        if stat == "yes":
            return C_WIN
        if stat == "weak":
            return C_STATE
        return C_NULL

    no_mask = t2["recovered"] == "NO"
    null_hw = float(t2.loc[no_mask, "best_corr"].abs().max()) if no_mask.any() else 0.06
    axB.axvspan(-null_hw, null_hw, color=C_NULL, alpha=0.18, zorder=0)
    axB.axvline(0.0, color=GREY_MID, lw=0.6, zorder=1)
    yticks, ylabs = [], []
    for yi, r in ordered:
        col = prog_color(r.recovered)
        big = r.recovered == "yes"
        axB.plot([0.0, r.best_corr], [yi, yi], color=col, lw=2.4 if big else 1.4,
                 alpha=0.95 if big else 0.7, solid_capstyle="round", zorder=2)
        axB.plot(r.best_corr, yi, marker="o", color=col, ms=6.0 if big else 3.8, mec="white",
                 mew=0.7, ls="", zorder=4)
        yticks.append(yi); ylabs.append(r.plab)

    axB.set_yticks(yticks)
    axB.set_yticklabels(ylabs, fontsize=FS_LAB)
    axB.set_xlim(-0.16, 0.92)
    axB.set_ylim(-0.7, y - 1.1 + 0.5)
    xb = [0.0, 0.2, 0.4, 0.6, 0.8]
    axB.set_xticks(xb)
    axB.set_xticklabels([_fmt(t, 1) for t in xb], fontsize=FS_LAB)
    axB.set_xlabel(r"program recovery  (AUCell-$\Delta$ corr.)", fontsize=FS_AXTITLE, color=C_INK)
    axB.tick_params(colors=C_INK, labelsize=FS_LAB)
    despine(axB)

    # DIRECT split labels via left accent bars (the reference grammar) — no per-row tags, no decode key.
    # Parked in the outer gutter (x in axes-fraction) so the rotated labels clear the long program names.
    _accent_bar(axB, min(yblock["C5"]) - 0.35, max(yblock["C5"]) + 0.35, NAVY, "T5 · cell-context",
                x=-0.475)
    _accent_bar(axB, min(yblock["C3"]) - 0.35, max(yblock["C3"]) + 0.35, GREY_MID, "T3 · unseen-pert",
                x=-0.475)

    # callout on the lone recovered program (type-I IFN, the topmost C5 mark)
    rIFN = t2[t2["program"] == "type_I_IFN"].iloc[0]
    yIFN = next(yi for yi, r in ordered if r.program == "type_I_IFN")
    axB.annotate("only program\nrecovered", xy=(rIFN.best_corr, yIFN),
                 xytext=(rIFN.best_corr - 0.04, yIFN - 1.05),
                 fontsize=FS_NOTE, color=C_WIN, ha="center", va="center", weight="normal",
                 linespacing=1.05, zorder=6,
                 arrowprops=dict(arrowstyle="-", color=C_WIN, lw=0.5,
                                 connectionstyle="angle,angleA=0,angleB=-90,rad=3",
                                 shrinkA=1.0, shrinkB=4.0))

    leg_b = [Line2D([0], [0], marker="o", color=C_WIN, lw=0, ms=4.6, mec="white", mew=0.7,
                    label="recovered"),
             Line2D([0], [0], marker="o", color=C_STATE, lw=0, ms=3.8, mec="white", mew=0.7,
                    label="weak"),
             Line2D([0], [0], marker="o", color=C_NULL, lw=0, ms=3.8, mec="white", mew=0.7,
                    label="at null")]
    legb = axB.legend(handles=leg_b, fontsize=FS_LEGEND, frameon=False, loc="upper center",
                      bbox_to_anchor=(0.5, -0.150), ncol=3, handletextpad=0.4, columnspacing=1.0,
                      borderaxespad=0.0)
    legb.set_zorder(8)

    # ===================== (c) direction vs magnitude =====================
    # C5 LOCT per-family means: AUCell rank concordance on the type-I IFN gene set (x) vs response-magnitude
    # Pearson-Δ (y). Conditioned families order the program well (~0.75–0.79) yet fall below the linear-PCA
    # magnitude floor; only FP-ridge clears it. Simple/OT floors pile at x≈0.
    d5 = pd.read_csv(RESULTS / "C5" / "results_raw.csv"); d5 = d5[d5.ran == True]  # noqa: E712
    loc5 = d5[d5.split.str.startswith("C5_loct")]
    g = loc5.groupby("baseline").agg(ifn=("aucell::type_I_IFN", "mean"),
                                     bulk=("pearson_delta", "mean"),
                                     fam=("family", "first")).reset_index()
    floor = g[g.baseline.isin(SIMPLE)].bulk.max()

    XLO, XHI, YLO, YHI = -0.13, 1.02, -0.05, 0.46
    THR = 0.5
    axC.set_xlim(XLO, XHI)
    axC.set_ylim(YLO, YHI)
    # "recover direction AND magnitude" gate: right of the rank-concordance threshold and above the floor
    axC.add_patch(Rectangle((THR, floor), XHI - THR, YHI - floor, facecolor="#EAF0F4",
                            edgecolor="none", lw=0.0, zorder=0, clip_on=True))
    axC.axhline(floor, color=C_FLOOR, ls="--", lw=0.9, zorder=1)
    axC.axvline(THR, color=C_FLOOR, ls=":", lw=0.8, zorder=1)
    axC.text(0.985, YHI - 0.018, "direction + magnitude", fontsize=FS_NOTE, color=NAVY_DARK,
             style="italic", weight="normal", ha="right", va="top", zorder=3)
    # floor-line caption parked in the open mid-x band just BELOW the dashed line, clear of the
    # linear/OT circle cluster that sits at the left end of the same line.
    axC.text(0.22, floor - 0.012, f"magnitude floor = {_fmt(floor)}", color=C_INK,
             fontsize=FS_NOTE, ha="left", va="top", zorder=5)

    MODEL_STYLE = {"FP-ridge": (C_WIN, "*"), "scGen": (C_COND, "o"),
                   "CPA": (C_COND, "s"), "STATE": (C_STATE, "^")}
    gi = g.set_index("baseline")
    # the simple/OT floors all sit at ifn=0; fan them apart with a small DISPLAY jitter (x leftward +
    # y up/down) purely so the three coincident circles read as three, then label the whole cluster with
    # ONE bracketed group caption (no per-dot leaders) — the reference's direct-label grammar.
    floor_stack = [b for b in ("linear-PCA", "donor-shift", "CINEMA-OT") if b in gi.index]
    floor_stack = sorted(floor_stack, key=lambda b: -float(gi.loc[b, "bulk"]))
    jit = {b: dx for b, dx in zip(floor_stack, (-0.012, -0.050, -0.088))}
    jit_y = {b: dy for b, dy in zip(floor_stack, (0.016, 0.0, -0.016))}  # vertical fan, display-only
    yjit = {"STATE": -0.012, "CPA": +0.012}   # separate the two near-coincident low-magnitude marks

    for _, r in g.iterrows():
        star = r.baseline == "FP-ridge"
        col, mk = MODEL_STYLE.get(r.baseline, (C_FLOOR, "o"))
        xpos = r.ifn + jit.get(r.baseline, 0.0)
        ypos = r.bulk + yjit.get(r.baseline, 0.0) + jit_y.get(r.baseline, 0.0)
        axC.scatter(xpos, ypos, s=185 if star else 48, marker=mk, color=col,
                    edgecolor=C_INK, lw=1.0 if star else 0.6, zorder=5)

    # ONE bracketed group caption for the tightly-clustered linear/OT pile — no per-dot leader lines.
    by = [float(gi.loc[b, "bulk"]) + jit_y[b] for b in floor_stack]
    bx = [float(gi.loc[b, "ifn"]) + jit[b] for b in floor_stack]
    axC.text(min(bx) - 0.010, max(by) + 0.052, "linear / OT\nbaselines", fontsize=FS_NOTE,
             color=C_FLOOR, ha="left", va="bottom", style="italic", linespacing=1.0, zorder=3)
    for b, dy, dx in [("cell-mean", 0.022, 0.016), ("ctrl-pred", -0.024, 0.030)]:
        if b in gi.index:
            axC.annotate(b, (gi.loc[b, "ifn"], gi.loc[b, "bulk"]),
                         (gi.loc[b, "ifn"] + dx, gi.loc[b, "bulk"] + dy),
                         fontsize=FS_NOTE, ha="left", va="center", color=C_FLOOR)

    def _leader(x, y, tx, ty, text, color, ha, va, fs, weight="normal"):
        axC.annotate(text, xy=(x, y), xytext=(tx, ty), ha=ha, va=va, fontsize=fs, color=color,
                     weight=weight, zorder=7,
                     arrowprops=dict(arrowstyle="-", color=color, lw=0.45,
                                     connectionstyle="angle,angleA=0,angleB=90,rad=2",
                                     shrinkA=0.5, shrinkB=2.5))
    # All four inline model-name labels are ONE hierarchy (a name pointing at its marker) → ONE size
    # (FS_NOTE). FP-ridge's emphasis is carried by its sky-blue colour + the large star marker, NOT by a
    # bigger label, so it no longer towers over scGen / STATE / CPA.
    _leader(gi.loc["FP-ridge", "ifn"], gi.loc["FP-ridge", "bulk"], THR - 0.020, 0.402,
            "FP-ridge", C_WIN, "right", "bottom", FS_NOTE)
    _leader(gi.loc["scGen", "ifn"], gi.loc["scGen", "bulk"], 0.600, 0.176,
            "scGen", C_COND_DARK, "right", "center", FS_NOTE)
    _leader(gi.loc["STATE", "ifn"], gi.loc["STATE", "bulk"] + yjit["STATE"], 0.855, 0.004,
            "STATE", C_STATE, "left", "center", FS_NOTE)
    # CPA label anchored ha="right" so the text grows LEFTWARD and its right edge stays well inside
    # the axes frame (XHI = 1.02); the leader runs up-right from the CPA marker to the label.
    _leader(gi.loc["CPA", "ifn"], gi.loc["CPA", "bulk"] + yjit["CPA"], 0.998, 0.130,
            "CPA", C_COND_DARK, "right", "center", FS_NOTE)

    axC.set_xlabel(r"direction: type-I IFN rank concordance  (AUCell-$\Delta$)",
                   fontsize=FS_AXTITLE, color=C_INK)
    axC.set_ylabel(r"magnitude: Pearson-$\Delta$  (higher better)", fontsize=FS_AXTITLE, color=C_INK)
    xc = [0.0, 0.25, 0.5, 0.75, 1.0]
    axC.set_xticks(xc)
    axC.set_xticklabels([_fmt(t, 2) for t in xc], fontsize=FS_LAB)
    yc = [0.0, 0.1, 0.2, 0.3, 0.4]
    axC.set_yticks(yc)
    axC.set_yticklabels([_fmt(t, 1) for t in yc], fontsize=FS_LAB)
    axC.tick_params(colors=C_INK, labelsize=FS_LAB)
    despine(axC)
    # NO bottom legend for panel c: all four method families are already labelled INLINE at their
    # markers (FP-ridge / scGen / STATE / CPA) via the direct leaders above — a separate decode key
    # would only duplicate those labels (the reference labels groups directly, never twice).

    # ===================== (d) per-lineage predictability (C1 + C5) =====================
    # T3: per-lineage scGen vs best simple baseline (C1) and FP-ridge vs best conditioned (C5). Grouped
    # horizontal dumbbells: C1 block (8 Kang lineages) shows the lone CD14⁺ monocyte win + the 3-seed
    # under-power note; C5 block (4 OP3 lineages) shows FP-ridge winning on all four. Left accent bars
    # label each block DIRECTLY (no boxed gutter text, no decode-legend).
    t3 = pd.read_csv(RESULTS / "_paper" / "immune_novelty" / "T3_per_lineage_predictability.csv")
    c1 = t3[t3.cluster == "C1"].copy().sort_values("gap_scGen_minus_simple").reset_index(drop=True)
    c5 = t3[t3.cluster == "C5"].copy().sort_values("best_conditioned_pearson_delta").reset_index(drop=True)
    ms = pd.read_csv(RESULTS / "_paper" / "multiseed_scgen_summary.csv")
    ms_c1 = ms[(ms.cluster == "C1") & (ms.n_seed >= 2)].set_index("lineage")

    LIN_LABEL = {"Mono_CD14": "CD14+ mono", "Mono_FCGR3A": "FCGR3A+ mono", "CD8T": "CD8 T",
                 "CD4T": "CD4 T", "B": "B", "NK": "NK", "Mk": "Mk", "DC": "DC",
                 "Mono": "monocyte", "T_cells": "T cells"}

    rowsD = []
    y = 0.0
    for r in c5.itertuples():           # C5 first -> lowest y
        rowsD.append(dict(block="C5", lin=r.lineage, simple=r.best_simple_pearson_delta,
                          cond=r.best_conditioned_pearson_delta, model="FP-ridge", y=y,
                          win=True)); y += 1.0
    y += 1.2                            # block gap
    for r in c1.itertuples():           # C1 on top
        rowsD.append(dict(block="C1", lin=r.lineage, simple=r.best_simple_pearson_delta,
                          cond=r.scGen_pearson_delta, model="scGen", y=y,
                          win=(r.winner == "scGen"))); y += 1.0
    ymax = y

    for d in rowsD:
        col_cond = C_WIN if d["model"] == "FP-ridge" else C_COND
        lo, hi = sorted([d["simple"], d["cond"]])
        axD.plot([lo, hi], [d["y"], d["y"]], color=GREY_LITE, lw=1.2, solid_capstyle="round", zorder=1)
        axD.plot(d["simple"], d["y"], "o", color=C_FLOOR, ms=4.2, mec="white", mew=0.7, zorder=3)
        axD.plot(d["cond"], d["y"], "o", color=col_cond, ms=4.8 if d["win"] else 4.2,
                 mec="white", mew=0.7, zorder=4)
        if d["win"]:
            axD.text(d["cond"], d["y"], "*", ha="center", va="center", color="white", fontsize=5.6,
                     weight="bold", zorder=6)

    # CD14 monocyte 3-seed under-power: seed-mean + n=3 t-CI as a thin whisker on a half-row above
    _T2 = 4.302653  # t(0.975, df=2)
    cd14_row = next((d for d in rowsD if d["block"] == "C1" and d["lin"] == "Mono_CD14"), None)
    if cd14_row is not None and "Mono_CD14" in ms_c1.index:
        m = float(ms_c1.loc["Mono_CD14", "pearson_mean"]); sd = float(ms_c1.loc["Mono_CD14", "pearson_sd"])
        se = sd / (3 ** 0.5)
        lo3, hi3 = m - _T2 * se, m + _T2 * se
        yy = cd14_row["y"]
        axD.plot([lo3, hi3], [yy + 0.32] * 2, color=C_COND_DARK, lw=1.0, alpha=0.85, zorder=2)
        axD.plot([lo3, lo3], [yy + 0.32 - 0.12, yy + 0.32 + 0.12], color=C_COND_DARK, lw=1.0, zorder=2)
        axD.plot([hi3, hi3], [yy + 0.32 - 0.12, yy + 0.32 + 0.12], color=C_COND_DARK, lw=1.0, zorder=2)
        axD.plot(m, yy + 0.32, "o", color=C_COND_DARK, ms=3.0, zorder=3)
        axD.annotate("3-seed mean,\nCI → 0\n(under-powered)", xy=(lo3, yy + 0.32),
                     xytext=(0.085, yy - 1.30), fontsize=FS_NOTE, color=C_COND_DARK, ha="left", va="center",
                     linespacing=1.05, weight="normal", zorder=6,
                     arrowprops=dict(arrowstyle="-", color=C_COND_DARK, lw=0.5,
                                     connectionstyle="angle,angleA=90,angleB=0,rad=3",
                                     shrinkA=1.0, shrinkB=2.5))

    yticks = [d["y"] for d in rowsD]
    ylabs = [LIN_LABEL.get(d["lin"], d["lin"]) for d in rowsD]
    axD.set_yticks(yticks)
    axD.set_yticklabels(ylabs, fontsize=FS_LAB)
    for tick, d in zip(axD.get_yticklabels(), rowsD):
        if d["block"] == "C1" and d["win"]:
            tick.set_color(C_COND_DARK); tick.set_fontweight("bold")
        elif d["block"] == "C5":
            tick.set_color(C_WIN)
    # Right margin: the CD14 3-seed CI whisker tops out near 0.99; carry the data box past 1.00 so the
    # whisker end-cap keeps clear whitespace from the right spine (no mark flush to the border).
    axD.set_xlim(0.0, 1.06)
    axD.set_ylim(-0.7, ymax - 0.2)
    xd = [0.0, 0.25, 0.5, 0.75, 1.0]
    axD.set_xticks(xd)
    axD.set_xticklabels([_fmt(t, 2) for t in xd], fontsize=FS_LAB)
    axD.set_xlabel(r"Pearson-$\Delta$, response direction  (higher better)",
                   fontsize=FS_AXTITLE, color=C_INK)
    axD.tick_params(colors=C_INK, labelsize=FS_LAB)
    despine(axD)

    # DIRECT block labels via left accent bars (replaces the three boxed gutter callouts)
    c5_ys = [d["y"] for d in rowsD if d["block"] == "C5"]
    c1_ys = [d["y"] for d in rowsD if d["block"] == "C1"]
    _accent_bar(axD, min(c5_ys) - 0.35, max(c5_ys) + 0.35, C_WIN, "T5 · all 4", x=-0.42)
    _accent_bar(axD, min(c1_ys) - 0.35, max(c1_ys) + 0.35, C_COND, "T1 · 1 of 8", x=-0.42)

    # ONE compact single-row legend; the "beats baseline" star is the SAME small size/weight as the
    # dot keys (matplotlib star marker, not a heavy text glyph) so all four keys read at equal weight.
    # The per-task tags "(T1)"/"(T5)" are dropped from the scGen / FP-ridge keys — the left accent
    # bars ("T1 · 1 of 8" / "T5 · all 4") already label the two blocks and each conditioned mark appears
    # ONLY in its own block, so the tag would label the same thing twice. Dropping it (and "best" →
    # "simple baseline") narrows the four-key row to fit INSIDE panel d (no run past the panel/canvas).
    leg_d = [Line2D([0], [0], marker="o", color=C_FLOOR, lw=0, ms=4.2, mec="white", mew=0.7,
                    label="simple baseline"),
             Line2D([0], [0], marker="o", color=C_COND, lw=0, ms=4.2, mec="white", mew=0.7,
                    label="scGen"),
             Line2D([0], [0], marker="o", color=C_WIN, lw=0, ms=4.2, mec="white", mew=0.7,
                    label="FP-ridge"),
             Line2D([0], [0], marker="*", color=C_INK, lw=0, ms=5.0, mec="white", mew=0.5,
                    label="beats baseline")]
    legd = axD.legend(handles=leg_d, fontsize=FS_LEGEND, frameon=False, loc="upper center",
                      bbox_to_anchor=(0.5, -0.165), ncol=4, handletextpad=0.25, columnspacing=0.7,
                      borderaxespad=0.0)
    legd.set_zorder(8)

    # ===================== suptitle, panel letters + concise titles =====================
    fig.text(0.012, 0.986, "The immune blind spots", fontsize=11.0, fontweight="bold",
             color=NAVY_DARK, ha="left", va="top")

    FS_HEAD = 8.0
    FS_LETTER = 9.7

    def head(ax, letter, title):
        panel_title(ax, letter, title, sub=None, fs_title=FS_HEAD, fs_letter=FS_LETTER)

    head(axA, "a", "Surface-protein recovery")
    head(axB, "b", "Program recovery")
    head(axC, "c", "Direction vs magnitude")
    head(axD, "d", "Per-lineage advantage")

    out = RESULTS / "_paper" / "figure_immune_blindspot.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
