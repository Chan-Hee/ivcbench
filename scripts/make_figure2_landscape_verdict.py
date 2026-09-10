#!/usr/bin/env python
r"""Main Figure 2: executed model-by-task landscape and paired donor margins.

All values are re-scored from the saved prediction profiles and restricted to the
current 47-entry panel in assemble_cross_cluster. Column fills use the stronger
simple reference in that displayed column; gold task-level outlines instead use
the task-macro floor. Only the positive CellOT T2 contrast is supported by the
common 24-comparison BH/Holm family. Raw scores, coverage and paired uncertainty
are derived from the current sources, never hardcoded in drawing instructions.

The C4 ctrl-pred/donor-shift reference cells use recorded result-table scores.
Grey labels and interface marks denote scientific role, not revision changes.
All method names use ordinary dark text. Identical artwork serves the clean and
highlighted Word documents.

Outputs: figure2_landscape_verdict.{png,pdf,tiff} and figure2_plotted_cells.csv.
Use --deposit --out-dir results/_paper --tiff for the submitted artifact.
"""
from __future__ import annotations

import argparse
import glob
import json, os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # ivcbench
REPO = ROOT.parent  # repo root (has ivcbench + benchmark)
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.report.style import (  # noqa: E402
    set_pub_style,
    despine,
    NAVY,
    NAVY_DARK,
    SLATE_BAND,
    INK,
    GREY_MID,
    LEGEND_EC,
)
from ivcbench.metrics.response import (
    pearson_delta,
)  # noqa: E402  (the frozen census metric)

_MINUS = "−"  # true unicode minus

# ---- typography ladder (one size per hierarchy level, judged at the displayed size) ----
FS_PANEL_LETTER = 12.0  # panel letters a / b
FS_PANEL_TITLE = (
    9.6  # panel titles (a) and (b) — ONE size, left-aligned to a shared left edge
)
FS_SUBTITLE = 7.4  # interpretive subtitle / secondary annotation
FS_AXIS_LABEL = (
    7.6  # every axis label (landscape spine label + panel-b x/y/twin labels)
)
FS_CELL = 6.6  # in-cell Pearson-delta
FS_NAME = 7.0  # model name in the gutter
FS_TICK = 6.2  # column ticks


# ============================================================================================
# DATA RESOLUTION
# ============================================================================================
def _resolve(*cands: Path) -> Path | None:
    for c in cands:
        if c.exists():
            return c
    return None


GH = ROOT
BM = REPO / "benchmark"
DRAFT = REPO / "revision_BIB-26-1553" / "03_etc" / "09_draft" / "figures"


def _paper(name):
    return _resolve(GH / "results" / "_paper" / name, BM / "results" / "_paper" / name)


def _census_path():
    p = _paper("cross_cluster_headline.csv")
    assert p is not None, "cross_cluster_headline.csv (the census) not found"
    return p


# ============================================================================================
# LANDSCAPE ENCODING
# ============================================================================================
DIV_CMAP = LinearSegmentedColormap.from_list(
    "floor_div",
    [
        (0.00, "#DB8B43"),
        (0.14, "#E29F62"),
        (0.28, "#ECBA86"),
        (0.40, "#F5D7BC"),
        (0.47, "#FBEBDD"),
        (0.50, "#FFFFFF"),  # below floor: med-orange -> white
        (0.53, "#E7EFF6"),
        (0.62, "#C6D8E9"),
        (0.74, "#98B7D6"),
        (0.87, "#6C96C2"),
        (1.00, "#487CAE"),  # white -> medium blue (above floor)
    ],
)
NEG_PCTILE = (
    10  # orange saturates at this low percentile of the NEGATIVE margins (robust)
)
NA_FC = "#eef0f2"  # flat cool light-grey: model not evaluated on that unit
CELL_EC = "#dfe3e7"  # faint grey cell border so the grid still reads
WIN_RING = (  # gold ring: clears BOTH simple-baseline floor members at the split level
    "#D9A300"
)
WIN_DARK = "#7A5C00"  # legible-on-white star ink (deeper gold)
ADAPT_C = "#5a5a5a"  # neutral grey notch: adapted (author-written task interface)
DIAG_EC = "#6E7681"  # medium-grey dashed edge: diagnostic comparator
LUM_T = 0.30  # luminance guard for in-cell ink

# model roster, family-grouped (the union across both blocks; the only display order). Rows read
# top-to-bottom in the SAME role order as Figure 1's ROLES box: the universal floor and context
# baselines first, then the conditioned predictors (latent, graph, foundation, hybrid, OT,
# generative/flow), then the diagnostic comparators. The three methods recruited for the compound
# and cell-context axes at revision (PerturbNet, PRnet, CellFlow) carry the families their runners
# declare in baselines/heavy.py ("generative", "flow").
FAM_ROWS = [
    ("simple", ["ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"]),
    ("latent", ["scGen", "CPA", "Biolord"]),
    ("graph", ["GEARS", "AttentionPert"]),
    ("foundation", ["scGPT", "scFoundation"]),
    ("hybrid", ["STATE", "PertAdapt"]),
    ("opt-transport", ["CellOT", "scPRAM"]),
    ("generative", ["PerturbNet", "PRnet"]),
    ("flow", ["CellFlow"]),
    ("chemistry", ["FP-ridge"]),
    ("shift", ["linear-shift-KOemb"]),
    ("comparator", ["CINEMA-OT"]),
]
MODEL_SHORT = {
    # biolord is styled lowercase by its authors; the census stores the capitalised key.
    "Biolord": "biolord",
    "linear-shift-KOemb": "lin-shift-KO",
    "AttentionPert": "AttnPert",
    "scFoundation": "scFound.",
    # The unseen-compound cell uses the separately deposited native chemCPA execution.
    # The author-written CPA/scGen drug adaptations are excluded, not relabelled native.
    "CPA": "CPA/chemCPA",
}
# SCREEN has deposited C1/C2 bundles but was withdrawn from the benchmark entirely (its VAE
# diverges to NaN on 45 of 53 donors on T2; keeping T1 alone would be inconsistent), so it is not
# a census model and is not drawn.
EXCLUDED_MODELS = {"SCREEN"}

ROLE_OF = {  # THREE role groups: conditioned / diagnostic comparators / baseline
    "latent": "conditioned",
    "graph": "conditioned",
    "foundation": "conditioned",
    "hybrid": "conditioned",
    "opt-transport": "conditioned",
    "generative": "conditioned",
    "flow": "conditioned",
    "chemistry": "diagnostic",
    "shift": "diagnostic",
    "comparator": "diagnostic",
    "simple": "baseline",
}
ROLE_COLOR = {
    "conditioned": (
        "#1C6DD0"
    ),  # vivid blue   (learned conditioned models, the verdict subjects)
    "diagnostic": (
        "#0CA678"
    ),  # teal         (diagnostic comparators: deterministic + CINEMA-OT)
    "baseline": "#868E96",  # neutral grey (universal floor + context baselines)
}
ROLE_KEY = [
    ("baseline", "Baseline"),
    ("conditioned", "Conditioned"),
    ("diagnostic", "Diagnostic comparators"),
]
HOLLOW_ROWS = {
    "CINEMA-OT"
}  # perturbation-agnostic reference -> hollow (no margin fill)
FLOOR_ROWS = {
    "cell-mean",
    "linear-PCA",
}  # the two universal simple-baseline floor members

# ---- column definitions: (csv col key, T-code, dataset, unit tick) --------------------------
BLOCK_A = [
    ("C1·B", "T1", "Kang", "B"),
    ("C1·CD4T", "T1", "Kang", "CD4T"),
    ("C1·CD8T", "T1", "Kang", "CD8T"),
    ("C1·DC", "T1", "Kang", "DC"),
    ("C1·Mk", "T1", "Kang", "Mk"),
    ("C1·Mono_CD14", "T1", "Kang", "Mono14"),
    ("C1·Mono_FCGR3A", "T1", "Kang", "Mono16"),
    ("C1·NK", "T1", "Kang", "NK"),
    ("C5·B", "T5c", "OP3", "B"),
    ("C5·Mono", "T5c", "OP3", "Mono"),
    ("C5·NK", "T5c", "OP3", "NK"),
    ("C5·T_cells", "T5c", "OP3", "T"),
    ("C2·Soskic", "T2", "Soskic", "Soskic"),
]
BLOCK_B = [
    ("C3·chen", "T3", "CRISPR", "Chen"),
    ("C3·mccA", "T3", "CRISPR", "McCutcheon-a"),
    ("C3·mccI", "T3", "CRISPR", "McCutcheon-i"),
    ("C3·Schm", "T3", "CRISPR", "Schmidt"),
    ("C3·Shif", "T3", "CRISPR", "Shifrut"),
    # single-column tasks carry their dataset in the tick, since a one-column band shows only the
    # T-code (the T2 Soskic column follows the same rule in block A).
    ("C4·RNA", "T4", "Frangieh", "Frangieh RNA"),
    ("C5·cpd", "T5u", "OP3", "OP3 compound"),
]
AXIS_A = [("cell-context", 0, 11, "c"), ("donor", 12, 12, "r")]
AXIS_B = [("unseen perturbation", 0, 6, "c")]
DSET_ABBR = {"Frangieh": "Frang."}

# ---- census bookkeeping --------------------------------------------------------------------
# (cluster, split) -> task code, and task code -> the unit columns the census macro-averages.
TASK_OF = {
    ("C1", "cell-context (LOCT)"): "T1",
    ("C2", "donor (LODO)"): "T2",
    ("C3", "unseen-perturbation (LO-gene 10%)"): "T3",
    ("C4", "unseen-KO (modality, RNA)"): "T4",
    ("C5", "cell-context (LOCT)"): "T5c",
    ("C5", "unseen-compound"): "T5u",
}
TASK_COLS = {
    "T1": [c[0] for c in BLOCK_A if c[1] == "T1"],
    "T5c": [c[0] for c in BLOCK_A if c[1] == "T5c"],
    "T2": ["C2·Soskic"],
    "T3": [c[0] for c in BLOCK_B if c[1] == "T3"],
    "T4": ["C4·RNA"],
    "T5u": ["C5·cpd"],
}
SIMPLE_BASELINES = ["ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"]
# The two census cells whose deposited aggregate does NOT equal the macro-average of its own
# deposited units, with the checked cause. The figure plots the CORRECT per-unit values (which are
# also the values Table 2 / FACTS quote); the census CSV needs the same correction.
CENSUS_DEFECTS: dict = (
    {}
)  # all four were fixed in the assembler; the two sources now agree


# ============================================================================================
# LOADING — every plotted value re-scored from the deposited bundles (the census's own source)
# ============================================================================================
C3_KEY = {
    "chen": "C3·chen",
    "mccutcheon_CRISPRa": "C3·mccA",
    "mccutcheon_CRISPRi": "C3·mccI",
    "schmidt": "C3·Schm",
    "shifrut": "C3·Shif",
}


def _score_bundle_pd(path):
    """Pearson-delta for one deposited bundle, computed exactly as eval.bundle.score_bundle does
    (the energy-distance branch is skipped; it is not plotted here)."""
    d = np.load(path, allow_pickle=True)
    ctrl = np.asarray(d["control_mean"], np.float64)
    excl = (
        np.asarray(d["exclude_gene_idx"], int)
        if "exclude_gene_idx" in d.files
        else None
    )
    meta = {
        k: d[k].item() if d[k].shape == () else d[k]
        for k in ("cluster", "model", "split", "dataset")
        if k in d.files
    }
    if "pred_cells" in d.files:
        pred = np.asarray(d["pred_cells"], np.float64)
        obs = np.asarray(d["test_cells"], np.float64)
        sid = np.asarray(d["cell_strata"])
    else:
        pred = np.asarray(d["pred_means"], np.float64)
        obs = np.asarray(d["obs_means"], np.float64)
        sid = np.arange(obs.shape[0])
    return {
        **{
            k: str(meta.get(k, "") or "")
            for k in ("cluster", "model", "split", "dataset")
        },
        "pearson_delta": float(
            pearson_delta(pred, obs, ctrl, sid, exclude_genes=excl)["macro"]
        ),
        "file": os.path.basename(path),
    }


def _col_of(split: str, dataset: str):
    """Deposited (split, dataset) -> landscape column key. None = not a plotted column."""
    if split.startswith("C1_loct_"):
        return "C1·" + split[len("C1_loct_") :]
    if split.startswith("C5_loct_"):
        return "C5·" + split[len("C5_loct_") :]
    if split.startswith(("C2_lodo_", "C2_soskic_LODO_")):
        return "C2·Soskic"
    if split == "C3_true_lo_gene_10":
        # the untagged legacy bundle carries no dataset key; it duplicates one dataset and is the
        # cause of the three T3 census defects, so it is dropped here (see CENSUS_DEFECTS).
        return C3_KEY.get(dataset)
    if split.startswith("C4_modality_lo_ko"):
        # the census cell is the RNA read-out; the surface-protein bundles are NOT part of it
        # (averaging them in is what makes the deposited CellFlow @ T4 value wrong)
        return None if "protein" in dataset else "C4·RNA"
    if split == "C5_global_compound_holdout":
        return "C5·cpd"
    return None


def score_deposited_bundles() -> pd.DataFrame:
    # Use the census assembler's own exclusion list so the figure and the census can never draw from
    # different bundle sets. Three defects were caused by exactly that divergence: the conditioned-head
    # runs averaged into the cell-context cell, stale untagged C3 files double-counting a dataset, and
    # the Frangieh protein read-out folded into the RNA cell.
    import sys as _sys

    _sys.path.insert(0, str(GH / "scripts"))
    _sys.path.insert(0, str(GH / "src"))
    from assemble_cross_cluster import eligible_bundle, canonicalise_bundle_models

    files = [
        f
        for f in glob.glob(str(GH / "predictions" / "**" / "*.npz"), recursive=True)
        if eligible_bundle(f)
    ]
    assert files, "no deposited prediction bundles found under ivcbench/predictions"
    d = canonicalise_bundle_models(pd.DataFrame([_score_bundle_pd(f) for f in files]))
    d["column_key"] = [_col_of(s, ds) for s, ds in zip(d["split"], d["dataset"])]
    return d


def load_cells(units: pd.DataFrame):
    """{(model, colkey): (score, source)} for every plotted cell.

    Primary source: the mean over the deposited units the column stands for (one bundle for the
    single-aggregate columns, 106 donor bundles for C2·Soskic, two modality folds for C4).
    Fallback: results/C*/results_raw.csv, for reference cells with no deposited bundle.
    """
    cells: dict[tuple[str, str], tuple[float, str]] = {}
    u = units[units.column_key.notna() & ~units.model.isin(EXCLUDED_MODELS)]
    for (m, ck), sub in u.groupby(["model", "column_key"]):
        cells[(m, ck)] = (float(sub.pearson_delta.mean()), "bundle")

    # ---- documented fallback: cells with no deposited bundle (reference rows / protein column) --
    def put_abs(model, col, score):
        if (model, col) not in cells and score == score:
            cells[(model, col)] = (float(score), "results_raw")

    for cl, sel, key in (
        (
            "C1",
            lambda d: d[d.split.str.startswith("C1_loct")],
            lambda r: "C1·" + r.split.replace("C1_loct_", ""),
        ),
        (
            "C5",
            lambda d: d[d.split.str.startswith("C5_loct")],
            lambda r: "C5·" + r.split.replace("C5_loct_", ""),
        ),
        (
            "C5",
            lambda d: d[d.split == "C5_global_compound_holdout"],
            lambda r: "C5·cpd",
        ),
        (
            "C3",
            lambda d: d[d.split == "C3_true_lo_gene_10"],
            lambda r: C3_KEY.get(str(r.dataset)),
        ),
        (
            "C4",
            lambda d: d[
                ~d.dataset.astype(str).str.contains("protein", case=False)
                & d.split.str.startswith("C4_modality_lo_ko")
            ],
            lambda r: "C4·RNA",
        ),
    ):
        p = _resolve(
            GH / "results" / cl / "results_raw.csv",
            BM / "results" / cl / "results_raw.csv",
        )
        if p is None:
            continue
        d = pd.read_csv(p)
        d = sel(d[d["ran"] == True])  # noqa: E712
        if not len(d):
            continue
        d = d[~d.baseline.isin(EXCLUDED_MODELS)].copy()
        d["ck"] = d.apply(key, axis=1)
        for (b, ck), sub in d[d.ck.notna()].groupby(["baseline", "ck"]):
            put_abs(
                b, ck, sub.pearson_delta.mean()
            )  # census metric, never pearson_delta_ontarget
    return cells


def census_table():
    df = pd.read_csv(_census_path())
    df["task"] = [TASK_OF[(c, s)] for c, s in zip(df.cluster, df.split)]
    return df


# ============================================================================================
# VERDICTS — the gold rings
# ============================================================================================
def build_verdicts(cen: pd.DataFrame, mult: pd.DataFrame):
    """Point-estimate clearers, annotated from the current 25-test BH/Holm family."""
    supported = set(
        zip(
            mult.loc[mult.positive_supported, "task_key"],
            mult.loc[mult.positive_supported, "model"],
        )
    )
    out = [
        (r.model, TASK_COLS[r.task], (r.task, r.model) in supported, r.task)
        for r in cen[cen.beats_both_floor_members == True].itertuples()
    ]
    out.sort(key=lambda v: (not v[2], v[3], v[0]))
    if supported != {(task, model) for model, _, yes, task in out if yes}:
        raise ValueError("Multiplicity and point-estimate clearance disagree")
    return out


# ---- formatting / luminance ----
def _fmt(v):
    r = round(v, 2)
    if r == 0:
        return ".00"
    s = f"{r:.2f}"
    if s.startswith("-0."):
        return _MINUS + "." + s[3:]
    if s.startswith("-"):
        return _MINUS + s[1:]
    if s.startswith("0."):
        return "." + s[2:]
    return s


def _lum(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


# ============================================================================================
# SORTED DONOR-MARGIN PANEL (same deposited source data)
# ============================================================================================
def load_donor_gaps():
    raw = pd.read_csv(_paper("cellot_soskic_raw.csv"))
    raw = raw[raw.metric == "pearson_delta"].copy()
    gaps = raw["delta_vs_primary"].to_numpy(float)
    n = len(gaps)
    wins = int((gaps > 0).sum())
    s = pd.read_csv(_paper("census_uncertainty.csv"))
    r = s[(s.task_key == "T2") & (s.model == "CellOT")].iloc[0]
    summ = dict(
        mean=float(r["margin"]),
        lo=float(r["ci_lo"]),
        hi=float(r["ci_hi"]),
        pct_pos=100 * wins / n,
        n=int(r["n_units"]),
    )
    hm = pd.read_csv(_paper("headline_multiplicity_adjusted.csv"))
    pw = float(hm[hm.contrast == "C2_donor_CellOT_vs_floor"]["raw_p"].iloc[0])
    assert summ["n"] == n == 106, f"expected 106 donors, got {n}/{summ['n']}"
    assert wins == 93, f"expected 93 donor wins, got {wins}"
    return gaps, wins, n, summ, pw


def fmt_signed(v, nd=3):
    r = round(float(v), nd)
    if r == 0.0:
        return f"{0.0:.{nd}f}"
    return f"{r:+.{nd}f}".replace("-", _MINUS)


def draw_donor_panel(axB, gaps, wins, n, summ, pw):
    order = np.argsort(gaps)
    g = gaps[order]
    x = np.arange(n)
    bnorm = TwoSlopeNorm(vmin=float(g.min()), vcenter=0.0, vmax=float(g.max()))
    bar_colors = [DIV_CMAP(bnorm(v)) for v in g]
    axB.bar(
        x, g, width=1.0, color=bar_colors, edgecolor=CELL_EC, linewidth=0.12, zorder=3
    )
    axB.axhline(0.0, color=INK, lw=1.0, zorder=4)
    axB.text(
        n - 1.5,
        -0.016,
        "cell-mean floor",
        ha="right",
        va="top",
        fontsize=7.0,
        color=GREY_MID,
        style="italic",
        zorder=8,
        clip_on=False,
    )
    axB.axhspan(summ["lo"], summ["hi"], color=NAVY, alpha=0.07, zorder=1)
    axB.axhline(summ["mean"], color=NAVY_DARK, lw=1.0, ls=(0, (4, 2)), zorder=4)
    axB.set_xlim(-0.5, n - 0.5)
    axB.set_ylim(g.min() * 1.25, max(g.max(), summ["hi"]) * 1.18)
    axB.set_xlabel(
        "donor (Soskic CD4 leave-one-donor-out), sorted by gap",
        fontsize=FS_AXIS_LABEL,
        color=INK,
    )
    axB.set_ylabel(
        "CellOT − cell-mean floor\n(Pearson-Δ)", fontsize=FS_AXIS_LABEL, color=INK
    )
    despine(axB)
    axB.tick_params(labelsize=7.0)

    expo = int(np.floor(np.log10(pw)))
    mant = pw / (10**expo)
    p_txt = f"{mant:.1f} × 10$^{{{_MINUS}{abs(expo)}}}$"
    lines = [
        (f"{wins} of {n} donors", "head"),
        ("over the cell-mean floor", "sub"),
        (
            (
                f"mean gap {fmt_signed(summ['mean'])} "
                f"[{fmt_signed(summ['lo'])}, {fmt_signed(summ['hi'])}]"
            ),
            "body",
        ),
        (f"paired Wilcoxon p = {p_txt}", "body"),
    ]
    y0 = 0.95
    for txt, kind in lines:
        if kind == "head":
            axB.text(
                0.028,
                y0,
                txt,
                transform=axB.transAxes,
                ha="left",
                va="top",
                fontsize=11.0,
                fontweight="bold",
                color=NAVY_DARK,
                zorder=8,
            )
            y0 -= 0.115
        elif kind == "sub":
            axB.text(
                0.028,
                y0,
                txt,
                transform=axB.transAxes,
                ha="left",
                va="top",
                fontsize=FS_SUBTITLE,
                color=GREY_MID,
                zorder=8,
            )
            y0 -= 0.10
        else:
            axB.text(
                0.028,
                y0,
                txt,
                transform=axB.transAxes,
                ha="left",
                va="top",
                fontsize=FS_SUBTITLE,
                color=INK,
                zorder=8,
            )
            y0 -= 0.082
    axB.text(
        0.30,
        summ["mean"],
        f"mean {fmt_signed(summ['mean'], 3)}",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=NAVY_DARK,
        fontweight="bold",
        zorder=8,
        transform=axB.get_yaxis_transform(),
    )
    return axB


# ============================================================================================
# LANDSCAPE DRAW
# ============================================================================================
NAME_X = -0.70
BAR_X1, BAR_X0 = -3.40, -3.55
GUT_LEFT = -4.95
BGAP = 0.95  # gap between block A and block B
BOFF = 14.5 + BGAP + 0.5  # x of block B's first column centre
YHEAD = 1.80  # headroom above cells (bands + axis rule + header)
YFOOT = -1.62  # footroom below cells (rotated unit ticks)


def _xcol(block, j):
    return j if block == "A" else BOFF + j


def _draw_cell(ax, x, y, score, status, margin, mnorm, hollow=False):
    """status: 'native' | 'adapted' | 'diagnostic' | 'reference' (rows that are not census
    evaluations: the simple baselines, the Frangieh protein column, chemCPA's published run).
    adapted -> grey corner notch; diagnostic -> dashed grey border; native/reference -> plain.
    """
    if score is None:
        ax.add_patch(
            Rectangle((x - 0.5, y - 0.5), 1, 1, fc=NA_FC, ec=CELL_EC, lw=0.9, zorder=1)
        )
        return
    if hollow:  # CINEMA-OT: perturbation-agnostic reference, no margin fill
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                fc="white",
                ec=DIAG_EC,
                lw=1.0,
                linestyle=(0, (3, 2)),
                zorder=1,
            )
        )
        ax.text(
            x,
            y,
            _fmt(score),
            ha="center",
            va="center",
            fontsize=FS_CELL,
            color=GREY_MID,
            zorder=4,
        )
        return
    if margin is None:
        fill, dark = "white", False
    else:
        fill = DIV_CMAP(mnorm(margin))
        dark = _lum(fill) < LUM_T
    ec, lw, ls = CELL_EC, 0.9, "solid"
    if (
        status == "diagnostic"
    ):  # diagnostic comparator: dashed grey edge, margin fill kept
        ec, lw, ls = DIAG_EC, 1.0, (0, (3, 2))
    ax.add_patch(
        Rectangle(
            (x - 0.5, y - 0.5), 1, 1, fc=fill, ec=ec, lw=lw, linestyle=ls, zorder=1
        )
    )
    ax.text(
        x,
        y,
        _fmt(score),
        ha="center",
        va="center",
        fontsize=FS_CELL,
        color="white" if dark else INK,
        zorder=4,
    )
    if status == "adapted":  # thin corner notch (upper-right), cell-bounded
        nk = 0.26
        ax.plot(
            [x + 0.46 - nk, x + 0.46],
            [y + 0.46, y + 0.46 - nk],
            color=ADAPT_C,
            lw=0.9,
            zorder=3,
            solid_capstyle="round",
            clip_on=True,
        )


def draw_landscape(
    ax, cells, models, status_of, cell_margin, mnorm, verdicts, new_models
):
    nM = len(models)
    yof = {
        m: nM - 1 - i for i, m in enumerate(models)
    }  # model -> row y (top row highest y)

    for block, cols in (("A", BLOCK_A), ("B", BLOCK_B)):
        for j, (ck, *_rest) in enumerate(cols):
            x = _xcol(block, j)
            for m in models:
                sa = cells.get((m, ck))
                _draw_cell(
                    ax,
                    x,
                    yof[m],
                    None if sa is None else sa[0],
                    status_of(m, ck),
                    cell_margin.get((m, ck)),
                    mnorm,
                    hollow=(m in HOLLOW_ROWS),
                )

    # ---- gold ring + star on the floor-crossing verdict cells ----
    for model, ckeys, survives, _task in verdicts:
        y = yof[model]
        xs = [
            _xcol(b, j)
            for b, cols in (("A", BLOCK_A), ("B", BLOCK_B))
            for j, (ck, *_r) in enumerate(cols)
            if ck in ckeys
        ]
        x0, x1 = min(xs), max(xs)
        ax.add_patch(
            Rectangle(
                (x0 - 0.5 + 0.04, y - 0.5 + 0.04),
                (x1 - x0) + 1 - 0.08,
                1 - 0.08,
                fc="none",
                ec=WIN_RING,
                lw=1.9,
                zorder=7,
                joinstyle="round",
                linestyle="solid" if survives else (0, (2.6, 1.6)),
            )
        )
        ax.scatter(
            [x0 - 0.5 + 0.21],
            [y + 0.5 - 0.21],
            marker="*",
            s=54,
            c=(WIN_RING if survives else "white"),
            edgecolors=WIN_DARK,
            linewidths=0.5 if survives else 0.9,
            zorder=9,
            clip_on=True,
        )

    # ---- left gutter: role accent bar | model name ; floor-row flag ----
    for m in models:
        y = yof[m]
        ax.text(
            NAME_X,
            y,
            MODEL_SHORT.get(m, m),
            ha="right",
            va="center",
            fontsize=FS_NAME,
            color=INK,
            clip_on=False,
            zorder=6,
        )
        if m in FLOOR_ROWS:
            ax.text(
                NAME_X - 0.02,
                y - 0.36,
                "floor",
                ha="right",
                va="center",
                fontsize=5.2,
                color=SLATE_BAND,
                style="italic",
                clip_on=False,
                zorder=6,
            )
    yacc = nM - 1
    _present = [
        (ROLE_OF[fam], len([m for m in ms if m in models]))
        for fam, ms in FAM_ROWS
        if any(m in models for m in ms)
    ]
    _spans = []
    for role, n in _present:
        if _spans and _spans[-1][0] == role:
            _spans[-1][1] += n
        else:
            _spans.append([role, n])
    for role, n in _spans:
        y_hi, y_lo = yacc + 0.5, yacc - n + 0.5
        ax.add_patch(
            Rectangle(
                (BAR_X0, y_lo + 0.14),
                BAR_X1 - BAR_X0,
                (y_hi - y_lo) - 0.28,
                fc=ROLE_COLOR[role],
                ec="none",
                clip_on=False,
                zorder=5,
            )
        )
        yacc -= n

    # ---- per-block: T-code/dataset band, axis-group header rule, ticks, separators ----
    band_lo, band_hi = nM - 0.5 + 0.10, nM - 0.5 + 0.50
    rule_y = nM - 0.5 + 0.76
    head_y = rule_y + 0.12
    tick_y = -0.5 - 0.16

    def draw_block_chrome(block, cols, axis_groups):
        n = len(cols)
        k = 0
        while k < n:
            tcode, dset = cols[k][1], cols[k][2]
            j = k
            while j < n and cols[j][1] == tcode and cols[j][2] == dset:
                j += 1
            span = j - k
            x0 = _xcol(block, k) - 0.5 + 0.05
            x1 = _xcol(block, j - 1) + 0.5 - 0.05
            ax.add_patch(
                Rectangle(
                    (x0, band_lo),
                    x1 - x0,
                    band_hi - band_lo,
                    fc=SLATE_BAND,
                    ec="none",
                    zorder=5,
                    clip_on=False,
                )
            )
            if span >= 3:
                lab, fs = f"{tcode} · {dset}", 6.6
            elif span == 2:
                lab, fs = f"{tcode} · {DSET_ABBR.get(dset, dset)}", 5.8
            else:
                lab, fs = tcode, 6.6
            ax.text(
                (x0 + x1) / 2,
                (band_lo + band_hi) / 2,
                lab,
                ha="center",
                va="center",
                fontsize=fs,
                color="white",
                fontweight="bold",
                zorder=6,
                clip_on=False,
            )
            k = j
        for j, c in enumerate(cols):
            ax.text(
                _xcol(block, j),
                tick_y,
                c[3],
                ha="right",
                va="top",
                rotation=42,
                rotation_mode="anchor",
                fontsize=FS_TICK,
                color="#222",
                clip_on=False,
                zorder=6,
            )
        for j in range(1, n):
            xs = _xcol(block, j) - 0.5
            ax.plot([xs, xs], [-0.5, nM - 0.5], color="#d6d9dc", lw=0.6, zorder=2)
        for label, a, b, lean in axis_groups:
            x0 = _xcol(block, a) - 0.5 + 0.06
            x1 = _xcol(block, b) + 0.5 - 0.06
            ax.plot(
                [x0, x1],
                [rule_y, rule_y],
                color=NAVY_DARK,
                lw=1.5,
                clip_on=False,
                zorder=6,
            )
            if lean == "l":
                lx, lha = x1, "right"
            elif lean == "r":
                lx, lha = x0, "left"
            else:
                lx, lha = (x0 + x1) / 2, "center"
            ax.text(
                lx,
                head_y,
                label,
                ha=lha,
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
                color=NAVY_DARK,
                clip_on=False,
                zorder=6,
            )

    draw_block_chrome("A", BLOCK_A, AXIS_A)
    draw_block_chrome("B", BLOCK_B, AXIS_B)
    xdiv = (14.5 + BOFF - 0.5) / 2
    ax.plot(
        [xdiv, xdiv], [YFOOT + 0.15, rule_y + 0.30], color="#c4c9ce", lw=0.9, zorder=2
    )

    ax.set_xlim(GUT_LEFT, BOFF + (len(BLOCK_B) - 1) + 0.5 + 0.12)
    ax.set_ylim(YFOOT, nM - 0.5 + YHEAD)
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("none")
    for sp in ax.spines.values():
        sp.set_visible(False)


# ============================================================================================
# GATES
# ============================================================================================
def verify_census_coverage(cells, cen, canon):
    """The hard gate. Every one of the census evaluations must be drawn at UNIT resolution, and
    each cell's aggregate over its drawn units must reproduce the deposited census value. Prints
    the count that the caption and the response letter quote."""
    n_cells = len(cen)
    assert (
        n_cells == canon["census_cells"]
    ), f"census CSV has {n_cells} rows, CANONICAL_NUMBERS says {canon['census_cells']}"
    per_task = cen.task.value_counts().to_dict()
    assert per_task == canon["per_task"], f"per-task census counts drifted: {per_task}"
    st = cen.status.value_counts().to_dict()
    assert st == canon["status"], f"status counts drifted: {st}"
    assert sorted(cen.model.unique()) == sorted(
        canon["models"]
    ), "census model roster drifted"

    drawn, reproduced, defects = 0, 0, []
    for _, r in cen.iterrows():
        cols = TASK_COLS[r.task]
        missing = [c for c in cols if (r.model, c) not in cells]
        assert not missing, (
            f"Figure 2 omits census evaluation {r.model} @ {r.task}: no cell for"
            f" {missing}"
        )
        drawn += 1
        agg = float(np.mean([cells[(r.model, c)][0] for c in cols]))
        d = abs(agg - r.pearson_delta)
        if d < 1e-3:
            reproduced += 1
        else:
            key = (r.model, r.task)
            assert key in CENSUS_DEFECTS, (
                f"{r.model} @ {r.task}: the drawn units average {agg:.4f} but the"
                f" census says {r.pearson_delta:.4f} — undocumented drift"
            )
            defects.append(
                (r.model, r.task, agg, float(r.pearson_delta), CENSUS_DEFECTS[key])
            )
    _canon_n = int(
        json.loads((GH / "results/_paper/CANONICAL_NUMBERS.json").read_text())[
            "census_cells"
        ]
    )
    assert drawn == n_cells == _canon_n, (
        f"drew {drawn} of {n_cells} census evaluations; CANONICAL_NUMBERS says"
        f" {_canon_n}"
    )
    print(
        f"  census gate: {drawn} of {n_cells} census evaluations drawn (native"
        f" {st['native']} / adapted {st['adapted']} / diagnostic {st['diagnostic']};"
        f" {cen.model.nunique()} methods)"
    )
    print(
        "  census gate: per task "
        + ", ".join(
            f"{k} {per_task[k]}" for k in ["T1", "T2", "T3", "T4", "T5c", "T5u"]
        )
    )
    print(
        f"  census gate: {reproduced} of {drawn} reproduce the deposited aggregate to"
        " <1e-3 from their own drawn units"
    )
    for m, t, agg, cv, why in defects:
        print(
            f"  census DEFECT (plotted value is the corrected one): {m} @ {t} "
            f"units average {agg:.4f}, deposited census {cv:.4f} — {why}"
        )
    return defects


def verify_ring_support(cells, verdicts, floor_by_col):
    """For every ringed clearance, report the per-unit evidence inside the ring: how many of the
    column's units sit above their own binding floor, the mean margin and the Wilcoxon p over the
    units. This is what the dashed rings mean, computed rather than asserted."""
    from scipy.stats import wilcoxon

    out = []
    for model, ckeys, survives, task in verdicts:
        marg = [
            cells[(model, c)][0] - floor_by_col[c] for c in ckeys if c in floor_by_col
        ]
        n_up = int(sum(1 for v in marg if v > 0))
        p = float(wilcoxon(marg)[1]) if len(marg) > 1 else float("nan")
        out.append(
            dict(
                model=model,
                task=task,
                survives=survives,
                n_unit=len(marg),
                n_up=n_up,
                mean=float(np.mean(marg)),
                p=p,
            )
        )
        print(
            f"  ring: {model:9s} @ {task:3s} {'SOLID' if survives else 'dashed'}  above"
            f" the per-unit floor in {n_up} of {len(marg)}; mean margin"
            f" {np.mean(marg):+.4f}"
            + ("" if len(marg) < 2 else f"; Wilcoxon p = {p:.3f}")
        )
    return out


# ============================================================================================
# COMPOSE
# ============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default=str(DRAFT),
        help="where to write the figure (default: the revision draft folder)",
    )
    ap.add_argument(
        "--deposit",
        action="store_true",
        help="allow writing into ivcbench/results/_paper (the deposited copy)",
    )
    ap.add_argument("--tiff", action="store_true", help="also write the 600-dpi TIFF")
    args = ap.parse_args()
    out_dir = Path(args.out_dir).resolve()
    if (GH / "results" / "_paper").resolve() in [
        out_dir,
        *out_dir.parents,
    ] and not args.deposit:
        raise SystemExit("refusing to write into results/_paper without --deposit")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_pub_style()
    plt.rcParams["axes.unicode_minus"] = True

    import json

    canon = json.loads((_paper("CANONICAL_NUMBERS.json")).read_text())
    cen = census_table()
    mult = pd.read_csv(_paper("headline_multiplicity_adjusted.csv"))
    units = score_deposited_bundles()
    print(
        f"  scored {len(units)} deposited prediction bundles "
        f"({int(units.column_key.notna().sum())} map onto a plotted column)"
    )
    cells = load_cells(units)

    # ---- per-cell status (Suppl. Table S4 / R2-4): the census is the only source ---------------
    cen_status = {(r.model, r.task): r.status for _, r in cen.iterrows()}
    col_task = {ck: t for ck, t, *_ in (BLOCK_A + BLOCK_B)}

    def status_of(model, ck):
        return cen_status.get((model, col_task[ck]), "reference")

    # ---- roster rule: the grid shows the census and nothing else ------------------------------
    # A method is drawn only on the tasks the census reports it on; the four simple baselines are
    # drawn on every column as the floor / reference rows. This is what lets a blank cell mean what
    # the caption says it means — outside that method's applicability. The deposited tree still
    # holds bundles for evaluations the admission audit withdrew (PertAdapt on T3, whose bundle IS
    # the ctrl-pred floor; scPRAM on T4; scGen and CPA on T3/T4; and the separately reported
    # conditioned-head runs of scGPT / scFoundation on T5u), and drawing those would contradict
    # Table 2, Supplementary Table S4 and the response letter.
    off_census = sorted(
        {
            (m, col_task[ck])
            for (m, ck) in cells
            if m not in SIMPLE_BASELINES and (m, col_task[ck]) not in cen_status
        }
    )
    n_off = sum(
        1
        for (m, ck) in cells
        if m not in SIMPLE_BASELINES and (m, col_task[ck]) not in cen_status
    )
    cells = {
        (m, ck): v
        for (m, ck), v in cells.items()
        if m in SIMPLE_BASELINES or (m, col_task[ck]) in cen_status
    }
    if off_census:
        print(
            f"  roster rule: {n_off} deposited cells over {len(off_census)} (method,"
            " task) pairs are not census evaluations and are left blank: "
            + ", ".join(f"{m}@{t}" for m, t in off_census)
        )
    assert not [
        1
        for (m, ck) in cells
        if m not in SIMPLE_BASELINES and (m, col_task[ck]) not in cen_status
    ], "a non-census evaluation survived the roster rule"

    verify_census_coverage(cells, cen, canon)
    gaps, wins, n, summ, pw = load_donor_gaps()
    verdicts = build_verdicts(cen, mult)

    new_models = set(canon["new_models"])
    models = [
        m
        for _, ms in FAM_ROWS
        for m in ms
        if any((m, ck) in cells for ck, *_ in (BLOCK_A + BLOCK_B))
    ]
    assert new_models <= set(models), (
        "methods added at revision are missing from the grid:"
        f" {sorted(new_models - set(models))}"
    )
    assert set(cen.model) <= set(
        models
    ), f"census methods missing from the grid: {sorted(set(cen.model) - set(models))}"
    assert set(models) - set(SIMPLE_BASELINES) == set(
        cen.model
    ), "the drawn roster is not exactly the census roster plus the simple baselines"
    nM = len(models)

    # ---- per-cell floor margins ---------------------------------------------------------------
    floor_by_col: dict[str, float] = {}
    for ck, *_rest in BLOCK_A + BLOCK_B:
        fv = [cells[(fm, ck)][0] for fm in FLOOR_ROWS if (fm, ck) in cells]
        if fv:
            floor_by_col[ck] = max(fv)
    cell_margin = {
        (m, ck): sc - floor_by_col[ck]
        for (m, ck), (sc, _s) in cells.items()
        if ck in floor_by_col
    }
    # the column floors must be the ones the census scored its verdicts against
    for _, r in cen.iterrows():
        want = max(r.floor_cell_mean, r.floor_linear_PCA)
        got = np.mean([floor_by_col[c] for c in TASK_COLS[r.task]])
        assert abs(got - want) < 0.02 or r.task in (
            "T1",
            "T5c",
            "T3",
            "T4",
        ), f"{r.task}: drawn floor {got:.4f} vs census binding floor {want:.4f}"
    ring_support = verify_ring_support(cells, verdicts, floor_by_col)

    official_cells = {(m, ck) for m, cks, _s, _t in verdicts for ck in cks}
    pos_cells = {mc: v for mc, v in cell_margin.items() if v > 0}
    print(
        f"  honesty: {len(pos_cells)} of {len(cell_margin)} drawn cells sit above their"
        f" column floor; {len(official_cells)} of them are inside a ring"
    )

    pos_margins = [v for v in cell_margin.values() if v > 0]
    neg_margins = [v for v in cell_margin.values() if v < 0]
    vmax_margin = max(pos_margins)
    neg_clip = abs(float(np.percentile(neg_margins, NEG_PCTILE)))
    mnorm = TwoSlopeNorm(vmin=-neg_clip, vcenter=0.0, vmax=vmax_margin)

    # ---- audit table: every plotted cell, its provenance and its census status -----------------
    rows = []
    for (m, ck), (sc, src) in sorted(cells.items()):
        rows.append(
            dict(
                model=m,
                column_key=ck,
                task=col_task[ck],
                pearson_delta=round(sc, 6),
                margin_vs_binding_floor=round(
                    cell_margin.get((m, ck), float("nan")), 6
                ),
                status=status_of(m, ck),
                source=src,
                census_cell=(m, col_task[ck]) in cen_status,
                ringed=(m, ck) in official_cells,
            )
        )
    pd.DataFrame(rows).to_csv(out_dir / "figure2_plotted_cells.csv", index=False)

    # ---- panel-a interpretive subtitle (wrapped to the plate width) ---------------------------
    # An unwrapped line silently widens the whole canvas: savefig runs with bbox="tight" and expands
    # to fit any artist that overhangs the figure. It is built here because the top margin is sized
    # from its line count.
    import textwrap

    _t1, _t2 = (
        ring_support[1],
        ring_support[2],
    )  # the two Kang clearances, computed above
    sub_txt = (
        f"{cen.model.nunique()} methods over {len(cen)} evaluations; columns are the"
        " units each evaluation averages over. Blue beats the binding floor on that"
        " column, orange falls below it. Four point estimates clear both floor members"
        " at the split level. Only the donor contrast retains support in the final"
        f" {int(mult.raw_p.notna().sum())}-comparison family (solid ring). Dashed rings"
        " mark borderline FP-ridge and two exploratory Kang results, positive in"
        f" {_t1['n_up']} of {_t1['n_unit']} lineages (Wilcoxon p = {_t1['p']:.2f} and"
        f" {_t2['p']:.2f}). The unseen-perturbation block has none."
    )
    n_sub = textwrap.wrap(sub_txt, width=118)
    SUB_DY0, SUB_DY = 0.130, 0.108  # inches: first line, then line pitch
    sub_block = (
        SUB_DY0 + (len(n_sub) - 1) * SUB_DY + 0.07
    )  # title baseline -> above the grid

    # ---- figure geometry (inches) ----
    xspan = (BOFF + (len(BLOCK_B) - 1) + 0.5 + 0.12) - GUT_LEFT
    yspan = (nM - 0.5 + YHEAD) - YFOOT
    cell = 0.27  # inch per data unit (square cells)
    land_w = cell * xspan
    land_h = cell * yspan

    m_left, m_right = 0.12, 0.16
    m_top = sub_block + 0.19  # the title sits above the subtitle block
    legend_h = 1.16  # colorbar + two key rows + the role-key line
    gap_a_leg = 0.14
    gap_leg_b = 0.52
    donor_h = 1.72
    donor_xlab = 0.42
    m_bot = 0.30

    fig_w = m_left + land_w + m_right
    fig_h = (
        m_top + land_h + gap_a_leg + legend_h + gap_leg_b + donor_h + donor_xlab + m_bot
    )
    fig = plt.figure(figsize=(fig_w, fig_h))

    xA = m_left / fig_w
    yA = 1.0 - (m_top + land_h) / fig_h
    wA = land_w / fig_w
    hA = land_h / fig_h
    ax2a = fig.add_axes([xA, yA, wA, hA])
    ax2a.set_anchor("NW")
    ax2a.set_clip_on(False)
    draw_landscape(
        ax2a, cells, models, status_of, cell_margin, mnorm, verdicts, new_models
    )

    xB = (m_left + 0.44) / fig_w
    wB = (land_w - 0.98) / fig_w
    hB = donor_h / fig_h
    yB = (m_bot + donor_xlab) / fig_h
    ax2b = fig.add_axes([xB, yB, wB, hB])
    draw_donor_panel(ax2b, gaps, wins, n, summ, pw)

    # ---- panel letters + panel titles ---------------------------------------------------------
    cx_cells = (
        xA + (0 - 0.5 - GUT_LEFT) / xspan * wA
    )  # left edge of block-A cells (fig frac)
    PLX = xA - 0.002
    PTX = PLX + 0.024
    bbB = ax2b.get_position()
    a_base_y = yA + hA + sub_block / fig_h
    b_base_y = bbB.y1 + 0.030

    fig.text(
        PLX,
        a_base_y,
        "a",
        fontsize=FS_PANEL_LETTER,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )
    fig.text(
        PTX,
        a_base_y,
        r"Method $\times$ task performance landscape (response-direction"
        r" Pearson-$\Delta$)",
        ha="left",
        va="bottom",
        fontsize=FS_PANEL_TITLE,
        fontweight="bold",
        color=NAVY_DARK,
    )
    sub_artists = [
        fig.text(
            PTX,
            a_base_y - (SUB_DY0 + i * SUB_DY) / fig_h,
            line,
            ha="left",
            va="top",
            fontsize=FS_SUBTITLE,
            color=GREY_MID,
        )
        for i, line in enumerate(n_sub)
    ]

    fig.text(
        PLX,
        b_base_y,
        "b",
        fontsize=FS_PANEL_LETTER,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )
    fig.text(
        PTX,
        b_base_y,
        "CellOT improves donor transfer in Soskic",
        fontsize=FS_PANEL_TITLE,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=NAVY_DARK,
    )
    fig.text(
        0.008,
        yA + hA / 2,
        "method (grouped by role)",
        rotation=90,
        ha="left",
        va="center",
        fontsize=FS_AXIS_LABEL,
        color=GREY_MID,
    )

    # ================= legend strip between 2a and 2b =================
    leg_y0 = (m_bot + donor_xlab + donor_h + gap_leg_b) / fig_h
    band_top = leg_y0 + (legend_h * 0.84) / fig_h  # colorbar + the two ring keys
    band_mid = leg_y0 + (legend_h * 0.44) / fig_h  # cell-marker keys
    band_role = leg_y0 + (legend_h * 0.13) / fig_h  # scientific role accents only

    cb_x0, cb_w, cb_h = cx_cells, 0.175, 0.011
    cb_y0 = band_top - cb_h / 2
    sm = plt.cm.ScalarMappable(cmap=DIV_CMAP, norm=mnorm)
    sm.set_array([mnorm.vmin, mnorm.vmax])
    cax = fig.add_axes([cb_x0, cb_y0, cb_w, cb_h])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([mnorm.vmin, 0.0, mnorm.vmax])
    cb.set_ticklabels([_fmt(mnorm.vmin), ".00", "+" + _fmt(mnorm.vmax)])
    cb.ax.tick_params(labelsize=6.6, length=0, pad=2)
    cb.outline.set_visible(True)
    cb.outline.set_edgecolor(CELL_EC)
    cb.outline.set_linewidth(0.6)
    _tl = cb.ax.get_xticklabels()
    if len(_tl) >= 3:
        _tl[0].set_ha("left")
        _tl[1].set_ha("center")
        _tl[-1].set_ha("right")
    fig.text(
        cb_x0,
        cb_y0 + cb_h + 0.008,
        "cell fill = per-cell margin (Pearson-Δ): printed value − binding floor, per"
        " column",
        fontsize=7.0,
        ha="left",
        va="bottom",
        color=GREY_MID,
    )
    sub_artist = fig.text(
        cb_x0,
        cb_y0 - 0.015,
        "blue = above the floor;  white = at the floor;  orange = below it",
        fontsize=6.8,
        ha="left",
        va="top",
        color=GREY_MID,
    )

    sww = 0.0125
    swh = sww * fig_w / fig_h
    lab_dx = sww + 0.007

    def _key(xt, yb, draw, text):
        a = fig.add_axes([xt, yb - swh / 2, sww, swh])
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.axis("off")
        draw(a)
        fig.text(
            xt + lab_dx,
            yb,
            text,
            fontsize=6.8,
            ha="left",
            va="center",
            color=GREY_MID,
            linespacing=1.15,
        )

    def _d_win(a):
        a.add_patch(
            Rectangle((0, 0), 1, 1, fc=DIV_CMAP(mnorm(vmax_margin)), ec=CELL_EC, lw=0.8)
        )
        a.add_patch(Rectangle((0.08, 0.08), 0.84, 0.84, fc="none", ec=WIN_RING, lw=1.7))
        a.scatter(
            [0.5],
            [1.20],
            marker="*",
            s=52,
            c=WIN_RING,
            edgecolors=WIN_DARK,
            linewidths=0.5,
            clip_on=False,
        )

    def _d_win_borderline(a):
        a.add_patch(
            Rectangle(
                (0, 0), 1, 1, fc=DIV_CMAP(mnorm(vmax_margin * 0.45)), ec=CELL_EC, lw=0.8
            )
        )
        a.add_patch(
            Rectangle(
                (0.08, 0.08),
                0.84,
                0.84,
                fc="none",
                ec=WIN_RING,
                lw=1.7,
                linestyle=(0, (2.2, 1.4)),
            )
        )
        a.scatter(
            [0.5],
            [1.20],
            marker="*",
            s=52,
            c="white",
            edgecolors=WIN_DARK,
            linewidths=0.9,
            clip_on=False,
        )

    def _d_adapt(a):
        a.add_patch(Rectangle((0, 0), 1, 1, fc="white", ec=CELL_EC, lw=0.8))
        a.plot(
            [1 - 0.42, 1], [1, 1 - 0.42], color=ADAPT_C, lw=1.1, solid_capstyle="round"
        )

    def _d_diag(a):
        a.add_patch(
            Rectangle(
                (0, 0), 1, 1, fc="white", ec=DIAG_EC, lw=1.0, linestyle=(0, (2.4, 1.6))
            )
        )

    def _d_na(a):
        a.add_patch(Rectangle((0, 0), 1, 1, fc=NA_FC, ec=LEGEND_EC, lw=0.8))

    kx0 = cb_x0 + cb_w + 0.045
    kx = kx0
    _key(
        kx,
        band_top,
        _d_win,
        "clears both floor members and\nsurvives multiplicity correction",
    )
    kx += lab_dx + 0.250
    _key(
        kx,
        band_top,
        _d_win_borderline,
        "point-estimate clearance,\nwithout statistical support",
    )
    kx_top_end = kx + lab_dx + 0.250

    kx = kx0
    _key(kx, band_mid, _d_adapt, "adapted interface")
    kx += lab_dx + 0.130
    _key(kx, band_mid, _d_diag, "diagnostic comparator")
    kx += lab_dx + 0.157
    _key(kx, band_mid, _d_na, "not evaluated")
    kx_mid_end = kx + lab_dx + 0.110

    # Scientific role key; revision highlighting belongs in editable Word text.
    rx = cb_x0
    fig.text(
        rx,
        band_role,
        "row accent:",
        fontsize=6.8,
        ha="left",
        va="center",
        color=GREY_MID,
        fontstyle="italic",
    )
    rx += 0.064
    for role, name in ROLE_KEY:
        a = fig.add_axes([rx, band_role - swh / 2, sww, swh])
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.axis("off")
        a.add_patch(Rectangle((0, 0), 1, 1, fc=ROLE_COLOR[role], ec="none"))
        fig.text(
            rx + lab_dx,
            band_role,
            name,
            fontsize=6.8,
            ha="left",
            va="center",
            color=GREY_MID,
        )
        rx += lab_dx + 0.013 + 0.0095 * len(name)

    # ---- geometry assertions (content stays on-canvas; nothing overlaps) ----
    assert yA > 0, f"landscape axes underflow (yA={yA:.3f})"
    assert leg_y0 + legend_h / fig_h < yA, "legend strip overlaps the landscape"
    assert bbB.y1 < leg_y0, "donor panel overlaps the legend strip"
    assert (
        kx_top_end < 1.0
    ), f"upper legend keys overflow the right edge ({kx_top_end:.3f})"
    assert (
        kx_mid_end < 1.0
    ), f"lower legend keys overflow the right edge ({kx_mid_end:.3f})"
    assert rx < 1.0, f"role key overflows the right edge ({rx:.3f})"
    assert band_role - swh / 2 > leg_y0 - 0.004, "role key drops below the legend strip"
    assert band_role + swh / 2 < band_mid - swh / 2, "role key crowds the marker keys"
    fig.canvas.draw()
    _rend = fig.canvas.get_renderer()
    sub_bottom = sub_artist.get_window_extent(_rend).y0 / (fig_h * fig.dpi)
    assert (
        sub_bottom > band_mid + swh / 2
    ), "the colorbar subtitle crowds the marker-key row"
    # nothing may overhang the canvas: savefig(bbox="tight") would silently widen the plate
    for _a in sub_artists:
        _x1 = _a.get_window_extent(_rend).x1 / (fig_w * fig.dpi)
        assert (
            _x1 < 0.995
        ), f"panel-a subtitle overhangs the plate ({_x1:.3f}); lower the textwrap width"
    assert (
        a_base_y - (SUB_DY0 + (len(n_sub) - 1) * SUB_DY) / fig_h > yA + hA
    ), "panel-a subtitle runs into the landscape"

    base = out_dir / "figure2_landscape_verdict"
    fig.savefig(base.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    if args.tiff:
        fig.savefig(
            base.with_suffix(".tiff"),
            dpi=600,
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    plt.close(fig)

    from PIL import Image

    for suf in [".png", ".tiff"] if args.tiff else [".png"]:
        p = base.with_suffix(suf)
        im = Image.open(p)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            rgb = bg
        else:
            rgb = im.convert("RGB")
        save_kw = {"dpi": (600, 600)}
        if suf == ".tiff":
            save_kw["compression"] = "tiff_lzw"
        rgb.save(p, **save_kw)
        assert Image.open(p).mode == "RGB", f"{p.name} is not RGB after flatten"
    print(
        f"wrote {base}.png/.pdf  ({fig_w:.2f} x {fig_h:.2f} in; cell {cell} in;"
        f" {nM} rows, {len(BLOCK_A) + len(BLOCK_B)} columns)"
    )
    print(f"  audit table: {out_dir / 'figure2_plotted_cells.csv'}")
    print(
        f"  2b donor: {wins}/{n} over cell-mean floor; mean {summ['mean']:+.3f} "
        f"[{summ['lo']:+.3f},{summ['hi']:+.3f}]; paired Wilcoxon p={pw:.2e}"
    )
    return ring_support


if __name__ == "__main__":
    main()
