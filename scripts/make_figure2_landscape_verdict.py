#!/usr/bin/env python
r"""Main Figure 2 — the Method x task PERFORMANCE LANDSCAPE with the floor-crossing verdict, plus
the donor-barrier companion panel.

This promotes the comprehensive landscape (scripts/figure_ranking.py, deposited as Supplementary
Fig. S1) to the main text and annotates the §2 sparsity headline directly on it:

  2a  Method (rows, grouped by role) x task (columns, grouped by the split they test),
      drawn as TWO side-by-side heatmap blocks on ONE shared model roster (a central gutter lists
      every model once):
        left block  — CELL-CONTEXT / MODALITY / DONOR transfer (T1 Kang LOCT, T5 OP3 LOCT,
                      T4 Frangieh RNA->surface, T2 Soskic LODO);
        right block — UNSEEN PERTURBATION (T3 primary-T CRISPR LO-gene, T5 OP3 LO-compound).
      Cell fill = each cell's PER-CELL margin over the binding (larger) of the two universal
      simple-baseline floor members for that column (cell-mean shift, linear-PCA shift), i.e.
      margin = Pearson-delta - max(floor_cell_mean, floor_linear_PCA), on a SOFT, BALANCED diverging
      scale centred on the floor: a cell BELOW the floor (margin < 0) is MEDIUM ORANGE that deepens
      smoothly with how far below it sits; a cell AT the floor (margin = 0) is WHITE; a cell ABOVE
      the floor (margin > 0) is BLUE that deepens with the margin: light blue for a small (weak)
      win, medium blue for a large one. The map is a graded gradient, NOT two harsh poles: medium
      orange and medium blue at the extremes, light tints near the floor, so weak wins stay in the
      blue family and losses stay distinguishable by HOW FAR below the floor they sit. The orange
      bound is set from the DATA (a robust low percentile of the negative margins) so the bulk of
      losses spread across the orange range and a single far outlier (ctrl-pred) does not flatten
      everyone into one flat shade; the blue bound is the largest positive per-cell margin (which is
      an official winner). The printed in-cell number stays the raw Pearson-delta (the SAME quantity
      as Supplementary Fig. S1), kept deliberately distinct from the margin colour, and is near-black
      so it reads on every shade. A GOLD ring + star marks the OFFICIAL split-level verdicts: the
      (model x task) cells that clear BOTH floor members at the aggregated split level per
      results/_paper/cross_cluster_headline.csv: EXACTLY TWO, both on the cell-context/donor side,
      CellOT @ T2 Soskic donor and FP-ridge @ T5 OP3 cell-context (four lineage cells under one
      ring). Some non-ringed cells also read blue because a method can beat the binding floor on a
      SPECIFIC lineage/column without officially winning the aggregated split (e.g. CellOT and scGen
      on the Kang CD14 monocyte lineage); the gold rings keep the official winners unambiguous and
      the colorbar frames blue as a per-column statement. The entire unseen-perturbation block stays
      at or below the floor and carries no ring; that is the one-shape finding: conditioning crosses
      the cell-context and donor barriers, never the unseen-perturbation barrier.

  2b  The donor barrier in detail — per-donor CellOT minus the per-donor CELL-MEAN FLOOR on the
      Soskic CD4 leave-one-donor-out task (n = 106). CellOT clears the cell-mean floor in 93 of 106
      donors; mean gap +0.107 Pearson-delta [+0.085, +0.129]; paired Wilcoxon p = 1.6e-13.

Every plotted value is read from deposited source data; no hardcoded numbers. Vector primitives,
true print points, editable text (pdf.fonttype 42), unicode minus. Output: results/_paper/
figure2_landscape_verdict.{png,pdf,tiff}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]            # ivcbench
REPO = ROOT.parent                                     # repo root (has ivcbench + benchmark)
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.report.style import (  # noqa: E402
    set_pub_style, despine,
    NAVY, NAVY_DARK, NAVY_RAMP, SLATE_BAND, INK, GREY_MID, CLAY_DARK, LEGEND_EC, FAM_COLORS,
)

_MINUS = "−"  # true unicode minus

# ---- typography ladder (one size per hierarchy level, judged at the displayed size) ----
FS_PANEL_LETTER = 12.0   # panel letters a / b
FS_PANEL_TITLE = 9.6     # panel titles (a) and (b) — ONE size, left-aligned to a shared left edge
FS_SUBTITLE = 7.6        # interpretive subtitle / secondary annotation
FS_AXIS_LABEL = 7.6      # every axis label (landscape spine label + panel-b x/y/twin labels)


# ============================================================================================
# DATA RESOLUTION — prefer the local ivcbench copy; fall back to the benchmark working tree
# for the few artefacts the release omits (CellOT-by-lineage, chemCPA-compound, additional models).
# ============================================================================================
def _resolve(*cands: Path) -> Path | None:
    for c in cands:
        if c.exists():
            return c
    return None


GH = ROOT
BM = REPO / "benchmark"


def _results_csv(cluster: str) -> Path:
    p = _resolve(GH / "results" / cluster / "results_raw.csv",
                 BM / "results" / cluster / "results_raw.csv")
    assert p is not None, f"missing results_raw for {cluster}"
    return p


def _additional(name: str) -> Path | None:
    return _resolve(GH / "results" / "_paper" / name,
                    BM / "outputs" / "additional_models" / name,
                    GH / "outputs" / "additional_models" / name)


def _soskic_axis() -> Path | None:
    return _resolve(GH / "results" / "C2" / "soskic_donor_axis.csv",
                    GH / "results" / "soskic_donor_axis.csv",
                    BM / "results" / "soskic_donor_axis.csv")


# ============================================================================================
# LANDSCAPE GEOMETRY / ENCODING (mirrors figure_ranking.py grammar)
# ============================================================================================
# CELL FILL ENCODES EACH CELL'S PER-CELL FLOOR MARGIN, NOT THE ABSOLUTE Pearson-Δ.
# The universal simple-baseline floor for a column = the BINDING (larger) of its two floor members
# (cell-mean shift, linear-PCA shift). A cell's per-cell margin = Pearson-Δ − binding floor; it is
# computed for EVERY cell (no clamping) so a weak win keeps its small positive value. The fill is a
# SOFT, BALANCED diverging scale centred on the floor (margin = 0):
#   margin < 0  (below the floor)  -> MEDIUM ORANGE that deepens smoothly with how far below
#   margin = 0  (at the floor)     -> WHITE  (the diverging centre)
#   margin > 0  (above the floor)  -> BLUE that deepens with the margin (light blue = weak win)
# The extremes are MEDIUM (not deep navy / deep rust): the field reads as one graded gradient, weak
# wins stay in the blue family, and losses stay distinguishable by depth. A GOLD ring marks only the
# OFFICIAL split-level verdicts from cross_cluster_headline.csv (CellOT on the Soskic donor split,
# FP-ridge on the OP3 cell-context split). A non-ringed cell can still read blue: a method may beat
# the binding floor on a specific lineage/column without officially winning the aggregated split; the
# rings keep the official winners unambiguous. The printed number stays the raw Pearson-Δ, so colour
# (per-cell margin) and number (level) are deliberately distinct.
# soft diverging colormap: medium orange (below floor) -> white (at floor) -> medium blue (above it).
# Chroma lifts off white quickly enough on the blue side that even a small positive reads as a light
# BLUE tint (not white), and the orange side gradates from pale peach near the floor to medium orange
# far below, so losses are separable by how far below the floor they sit.
DIV_CMAP = LinearSegmentedColormap.from_list("floor_div", [
    (0.00, "#DB8B43"), (0.14, "#E29F62"), (0.28, "#ECBA86"), (0.40, "#F5D7BC"),
    (0.47, "#FBEBDD"), (0.50, "#FFFFFF"),                         # below floor: med-orange -> white
    (0.53, "#E7EFF6"), (0.62, "#C6D8E9"), (0.74, "#98B7D6"),
    (0.87, "#6C96C2"), (1.00, "#487CAE"),                         # white -> medium blue (above floor)
])
NEG_PCTILE = 10            # orange saturates at this low percentile of the NEGATIVE margins (robust):
#   the bulk of the 123 below-floor cells then spread across the orange gradient instead of being
#   flattened by the single far outlier (ctrl-pred, margin down to ≈ −0.88); the magnitude is read
#   from the data at draw time (≈ 0.56), not hardcoded.
NA_FC = "#eef0f2"          # flat cool light-grey: model not run / not applicable on that task
CELL_EC = "#dfe3e7"        # faint grey cell border so the grid still reads
WIN_RING = "#D9A300"       # gold ring: beats BOTH simple-baseline floor members (the §2 headline)
WIN_DARK = "#7A5C00"       # legible-on-white star ink (deeper gold)
ADAPT_C = "#5a5a5a"        # neutral grey notch: adapted (re-fit / non-native) run
LUM_T = 0.30               # luminance guard for in-cell ink: near-black on every shade in this soft
#   map (the deepest blue/orange stay light enough for black text); white only if a future deeper
#   colormap pushes a fill below this luminance.

# model roster, family-grouped (the union across both blocks; the only display order).
# Rows read top-to-bottom in the SAME role order as Figure 1's ROLES box: conditioned predictors
# first (latent, graph, foundation, hybrid, OT-CellOT), then deterministic predictors (chemistry,
# shift), then the comparator (CINEMA-OT), then the baselines (simple). The opt-transport pair is
# SPLIT so CellOT sits in the conditioned block (family label "OT") and CINEMA-OT becomes its own
# comparator row (family key "comparator", label "comp.").
FAM_ROWS = [
    ("simple", ["ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"]),
    ("latent", ["scGen", "CPA", "chemCPA"]),
    ("graph", ["GEARS", "AttentionPert"]),
    ("foundation", ["scGPT", "scFoundation"]),
    ("hybrid", ["STATE", "PertAdapt"]),
    ("opt-transport", ["CellOT", "scPRAM"]),
    ("chemistry", ["FP-ridge"]),
    ("shift", ["linear-shift-KOemb"]),
    ("comparator", ["CINEMA-OT"]),
]
FAM_KEY = {"simple": "simple", "latent": "latent", "graph": "graph", "foundation": "foundation",
           "hybrid": "hybrid", "chemistry": "chemistry", "shift": "shift",
           "opt-transport": "opt-transport", "comparator": "comparator"}
FAM_LABEL = {"simple": "simple", "latent": "latent", "graph": "graph", "foundation": "found.",
             "hybrid": "hybrid", "chemistry": "chem.", "shift": "shift", "opt-transport": "OT",
             "comparator": "comp."}
MODEL_SHORT = {"linear-shift-KOemb": "lin-shift-KO", "AttentionPert": "AttnPert",
               "scFoundation": "scFound."}

# ---- ROLE palette (Figure 1's exact hexes, so the two figures share a colour language) -----------
# Each family belongs to one of Figure 1's four ROLES-box roles; the gutter accent bar + family word
# are tinted by ROLE (not by family), mirroring Figure 1: conditioned = navy, deterministic =
# teal/green, comparator = grey, baseline = light blue / slate (Figure 1's Universal-floor / cell-
# context identity). FAM_COLORS (the per-family palette) is imported from the shared style module and
# is left untouched; these two local maps override the hue ONLY for the gutter accent + family word.
ROLE_OF = {   # THREE role groups: conditioned / diagnostic comparators / baseline
    "latent": "conditioned", "graph": "conditioned", "foundation": "conditioned",
    "hybrid": "conditioned", "opt-transport": "conditioned",
    "chemistry": "diagnostic", "shift": "diagnostic", "comparator": "diagnostic",
    "simple": "baseline",
}
ROLE_COLOR = {       # gutter strips + legend role key share these (three role groups)
    "conditioned": "#1C6DD0",    # vivid blue   (learned conditioned models, the verdict subjects)
    "diagnostic":  "#0CA678",    # teal         (diagnostic comparators: deterministic + CINEMA-OT)
    "baseline":    "#868E96",    # neutral grey (universal floor + context baselines)
}
ROLE_KEY = [  # compact one-line role key in the legend strip (left->right = top-to-bottom row order)
    ("baseline", "Baseline"), ("conditioned", "Conditioned"),
    ("diagnostic", "Diagnostic comparators"),
]
# cell-level marking so diagnostic comparators do not read as conditioned-model verdicts:
DIAG_EC = "#6E7681"                                  # medium-grey edge for diagnostic-comparator cells
DASHED_ROWS = {"FP-ridge", "linear-shift-KOemb"}     # deterministic side-info -> dashed grey border (fill kept)
HOLLOW_ROWS = {"CINEMA-OT"}                          # perturbation-agnostic reference -> hollow (no margin fill)

# the two row identities that ARE the universal simple-baseline floor (flagged in the gutter).
FLOOR_ROWS = {"cell-mean", "linear-PCA"}

# ---- column definitions: (csv col key, T-code, dataset, lineage tick) -----------------------
# left block: cell-context (Kang T1, OP3 T5) | modality (Frangieh T4) | donor (Soskic T2)
BLOCK_A = [
    ("C1·B", "T1", "Kang", "B"), ("C1·CD4T", "T1", "Kang", "CD4T"),
    ("C1·CD8T", "T1", "Kang", "CD8T"), ("C1·DC", "T1", "Kang", "DC"),
    ("C1·Mk", "T1", "Kang", "Mk"), ("C1·Mono_CD14", "T1", "Kang", "Mono14"),
    ("C1·Mono_FCGR3A", "T1", "Kang", "Mono16"), ("C1·NK", "T1", "Kang", "NK"),
    ("C5·B", "T5", "OP3", "B"), ("C5·Mono", "T5", "OP3", "Mono"),
    ("C5·NK", "T5", "OP3", "NK"), ("C5·T_cells", "T5", "OP3", "T"),
    ("C4·RNA", "T4", "Frangieh", "RNA"), ("C4·prot", "T4", "Frangieh", "protein"),
    ("C2·Soskic", "T2", "Soskic", "Soskic"),
]
# right block: unseen perturbation (CRISPR T3, compound T5). The two McCutcheon arms (CRISPRa,
# CRISPRi) are FIVE distinct primary-human-T CRISPR evaluation units alongside Chen/Schmidt/Shifrut;
# they carry distinct column keys (C3·mccA / C3·mccI) so neither arm is silently overwritten. Tick
# "-a" = CRISPRa, "-i" = CRISPRi, kept short to match the other single-token column labels.
BLOCK_B = [
    ("C3·chen", "T3", "CRISPR", "Chen"),
    ("C3·mccA", "T3", "CRISPR", "McCutcheon-a"), ("C3·mccI", "T3", "CRISPR", "McCutcheon-i"),
    ("C3·Schm", "T3", "CRISPR", "Schmidt"), ("C3·Shif", "T3", "CRISPR", "Shifrut"),
    ("C5·cpd", "T5", "OP3", "compound"),
]
# axis-group spans (for the header rules) per block: (label, first idx, last idx, lean).
# lean: "c" centred, "l" right-anchored to its own right edge (leans left/inward), "r" left-anchored
# to its own left edge — used to split the adjacent narrow modality/donor headers off the shared seam.
AXIS_A = [("cell-context", 0, 11, "c"), ("modality", 12, 13, "l"), ("donor", 14, 14, "r")]
AXIS_B = [("unseen perturbation", 0, 5, "c")]
DSET_ABBR = {"Frangieh": "Frang."}
# the two floor-crossing (model, set-of-colkeys) verdicts read from cross_cluster_headline.csv.
VERDICTS = [
    ("CellOT", ["C2·Soskic"]),
    ("FP-ridge", ["C5·B", "C5·Mono", "C5·NK", "C5·T_cells"]),
]


def _ran(cluster: str) -> pd.DataFrame:
    d = pd.read_csv(_results_csv(cluster))
    return d[d["ran"] == True]  # noqa: E712


def _norm_action(a):
    return {"run_floor": "floor", "run_adapted": "adapted"}.get(a, "native")


def _action_of(sub):
    acts = set(sub.get("action", []))
    return "floor" if "run_floor" in acts else ("adapted" if "run_adapted" in acts else "native")


def load_cells():
    """Return {(model, colkey): (score, action)} for every plotted cell. Same provenance as
    figure_ranking.long_table (results_raw + additional models), restricted to the columns above."""
    cells: dict[tuple[str, str], tuple[float, str]] = {}

    def put(model, col, score, action):
        if score == score:  # not NaN
            cells[(model, col)] = (float(score), action)

    # C1 cell-context (Kang LOCT). For scGen, prefer the deposited 3-SEED MEAN
    # (multiseed_scgen_summary.csv) over the seed-0 value in results_raw, so the printed Kang cells
    # match Table S9 / Note S3 (CD14 monocyte 0.884, not the seed-0 0.917). Other models / lineages
    # without a multiseed row keep their seed-0 results_raw value.
    ms_path = _resolve(GH / "results" / "_paper" / "multiseed_scgen_summary.csv",
                       BM / "results" / "_paper" / "multiseed_scgen_summary.csv")
    scgen_kang_mean = {}
    if ms_path is not None:
        _ms = pd.read_csv(ms_path)
        _ms = _ms[(_ms.cluster == "C1") & _ms.lineage.notna() & (_ms.n_seed >= 2)]
        scgen_kang_mean = {r.lineage: float(r.pearson_mean) for r in _ms.itertuples()}
    d1 = _ran("C1")
    for s in (x for x in d1.split.unique() if x.startswith("C1_loct")):
        lin = s.replace("C1_loct_", "")
        for _, r in d1[d1.split == s].iterrows():
            score = r.pearson_delta
            if r.baseline == "scGen" and lin in scgen_kang_mean:
                score = scgen_kang_mean[lin]      # 3-seed mean (deposited), not seed-0
            put(r.baseline, f"C1·{lin}", score, _norm_action(r.action))

    # C5 cell-context (OP3 LOCT) + C5 unseen-compound
    d5 = _ran("C5")
    for s in (x for x in d5.split.unique() if x.startswith("C5_loct")):
        lin = s.replace("C5_loct_", "")
        for _, r in d5[d5.split == s].iterrows():
            put(r.baseline, f"C5·{lin}", r.pearson_delta, _norm_action(r.action))
    for _, r in d5[d5.split == "C5_global_compound_holdout"].iterrows():
        put(r.baseline, "C5·cpd", r.pearson_delta, _norm_action(r.action))

    # C4 modality (Frangieh RNA / protein). Case-insensitive: the deposited dataset keys mix
    # "frangieh_RNA" (baselines) and "frangieh_rna" (scGen, linear-shift), so a case-sensitive
    # "RNA" in ds silently routed the lowercase RNA runs into the protein column.
    d4 = _ran("C4")
    for ds in d4.dataset.unique():
        mod = "RNA" if "rna" in str(ds).lower() else "prot"
        for b in d4[d4.dataset == ds].baseline.unique():
            sub = d4[(d4.dataset == ds) & (d4.baseline == b)]
            put(b, f"C4·{mod}", sub.pearson_delta_ontarget.mean(), _action_of(sub))

    # C3 unseen perturbation (CRISPR LO-gene). Explicit dataset->column-key map: the two McCutcheon
    # arms (CRISPRa, CRISPRi) MUST get distinct keys so neither overwrites the other (the prior
    # f"C3·{ds[:6]}" slug mapped both arms to "C3·mccutc" and put() silently dropped one). Five units.
    C3_KEY = {"chen": "C3·chen", "mccutcheon_CRISPRa": "C3·mccA",
              "mccutcheon_CRISPRi": "C3·mccI", "schmidt": "C3·Schm", "shifrut": "C3·Shif"}
    d3 = _ran("C3")
    d3 = d3[d3.split == "C3_true_lo_gene_10"]   # headline 10% LO-gene split only (match S2b census; not pooled 10/25/50%)
    for ds in d3.dataset.unique():
        key = C3_KEY[ds]
        for b in d3[d3.dataset == ds].baseline.unique():
            sub = d3[(d3.dataset == ds) & (d3.baseline == b)]
            put(b, key, sub.pearson_delta_ontarget.mean(), _action_of(sub))

    # C2 Soskic donor (simple + conditioned, from the donor-axis table)
    sp = _soskic_axis()
    if sp is not None:
        d2 = pd.read_csv(sp)
        for b in d2.model.unique():
            sub = d2[d2.model == b]
            put(b, "C2·Soskic", sub.pearson_delta.mean(), "native")

    # additional models
    ck = _additional("cellot_kang_by_lineage.csv")
    if ck is not None:
        for _, r in pd.read_csv(ck).query("metric == 'pearson_delta'").iterrows():
            put("CellOT", f"C1·{r.lineage}", r.cellot_score, "native")
    cs = _additional("cellot_summary.csv")
    if cs is not None:
        s = pd.read_csv(cs)
        so = s[(s.dataset == "soskic2022") & (s.metric == "pearson_delta")]
        if len(so):
            put("CellOT", "C2·Soskic", float(so.model_score.iloc[0]), "adapted")
    cc = _additional("chemcpa_op3_unseen_compound_summary.csv")
    if cc is not None:
        put("chemCPA", "C5·cpd", float(pd.read_csv(cc).chemCPA_score.iloc[0]), "native")

    # ---- completeness fills: every model that ran must appear (Table S2b / 14-model census) -------
    # The two single-aggregate columns (C4 modality, C2 donor) previously under-loaded. Six C4
    # models (CPA, CellOT, GEARS, AttentionPert, STATE, scPRAM) live in the C4 "fills" deposit, and
    # CPA/STATE/scPRAM on the donor axis are not in the soskic per-donor file; pull both so the grid
    # shows all 14 models and every aggregate cell matches cross_cluster_headline (= Suppl. Table
    # S2b = main text). put_abs never overwrites an already-loaded (validated) cell.
    def put_abs(model, col, score, action):
        if (model, col) not in cells and score == score:  # absent & not NaN
            cells[(model, col)] = (float(score), action)

    fc4 = _additional("results_raw_C4_fills_rewrapped.csv")
    if fc4 is not None:
        f = pd.read_csv(fc4)
        f = f[f["ran"] == True]  # noqa: E712
        for mdl, sub in f.groupby("baseline"):
            put_abs(mdl, "C4·RNA", sub["pearson_delta"].mean(), _action_of(sub))  # match census metric (pearson_delta), not _ontarget

    hp = _resolve(GH / "results" / "_paper" / "cross_cluster_headline.csv",
                  BM / "results" / "_paper" / "cross_cluster_headline.csv")
    if hp is not None:
        hh = pd.read_csv(hp)
        for _, r in hh[(hh.cluster == "C2") & hh.model.isin(["CPA", "STATE", "scPRAM"])].iterrows():
            put_abs(r.model, "C2·Soskic", r.pearson_delta, "adapted")  # donor models are per-fold re-fit
        for _, r in hh[hh.cluster == "C4"].iterrows():                 # safety net for any C4 gap
            put_abs(r.model, "C4·RNA", r.pearson_delta, "native")

    return cells


def verify_verdicts():
    """Read cross_cluster_headline.csv and assert EXACTLY the two beats-both-floor cells we annotate."""
    p = _resolve(GH / "results" / "_paper" / "cross_cluster_headline.csv",
                 BM / "results" / "_paper" / "cross_cluster_headline.csv")
    df = pd.read_csv(p)
    win = df[df.beats_both_floor_members == True]  # noqa: E712
    got = {(r.model, r.cluster, r.split) for _, r in win.iterrows()}
    expect = {("CellOT", "C2", "donor (LODO)"), ("FP-ridge", "C5", "cell-context (LOCT)")}
    assert got == expect, f"floor-crossing verdict drift: {got} != {expect}"
    return win


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
# DONOR-CDF PANEL (ported from figure_main2.py; same deposited source data)
# ============================================================================================
def _paper(name):
    return _resolve(GH / "results" / "_paper" / name, BM / "results" / "_paper" / name)


def load_donor_gaps():
    raw = pd.read_csv(_paper("cellot_soskic_raw.csv"))
    raw = raw[raw.metric == "pearson_delta"].copy()
    gaps = raw["delta_vs_primary"].to_numpy(float)
    n = len(gaps)
    wins = int((gaps > 0).sum())
    s = pd.read_csv(_paper("cellot_summary.csv"))
    r = s[(s.dataset == "soskic2022") & (s.metric == "pearson_delta")].iloc[0]
    summ = dict(mean=float(r["delta_vs_primary_baseline"]), lo=float(r["CI_low"]),
                hi=float(r["CI_high"]), pct_pos=float(r["percent_positive_units"]),
                n=int(r["n_units"]))
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
    # Colour each per-donor bar by its OWN gap with the SAME soft diverging map as panel (a):
    # soft orange below 0 (deepening with how negative), white near 0, soft blue above 0
    # (deepening with how positive). TwoSlopeNorm centred at 0 with bounds fit to THIS panel's
    # gap range, so the sorted bars read as one smooth orange -> blue gradient left to right,
    # visually consistent with the landscape's floor-margin palette. Faint grey edge (CELL_EC,
    # the same border the landscape cells carry) so near-zero near-white bars still read on white.
    bnorm = TwoSlopeNorm(vmin=float(g.min()), vcenter=0.0, vmax=float(g.max()))
    bar_colors = [DIV_CMAP(bnorm(v)) for v in g]
    axB.bar(x, g, width=1.0, color=bar_colors, edgecolor=CELL_EC, linewidth=0.12, zorder=3)
    axB.axhline(0.0, color=INK, lw=1.0, zorder=4)
    axB.text(n - 1.5, -0.016, "cell-mean floor", ha="right", va="top",
             fontsize=7.0, color=GREY_MID, style="italic", zorder=8, clip_on=False)
    axB.axhspan(summ["lo"], summ["hi"], color=NAVY, alpha=0.07, zorder=1)
    axB.axhline(summ["mean"], color=NAVY_DARK, lw=1.0, ls=(0, (4, 2)), zorder=4)
    axB.set_xlim(-0.5, n - 0.5)
    axB.set_ylim(g.min() * 1.25, max(g.max(), summ["hi"]) * 1.18)
    axB.set_xlabel("donor (Soskic CD4 leave-one-donor-out), sorted by gap", fontsize=7.6, color=INK)
    axB.set_ylabel("CellOT − cell-mean floor\n(Pearson-Δ)", fontsize=7.6, color=INK)
    despine(axB)
    axB.tick_params(labelsize=7.0)

    # CDF twin: cumulative donor fraction over the SORTED donors, plotted on the SAME donor-rank
    # x-axis as the bars (x = rank), so the curve aligns with the bars and reads the under-baseline
    # fraction exactly where the bars cross zero (~rank 13 -> ~0.12). (Plotting it against the gap
    # value collapses it to a spike at x=0, since every gap is small relative to the 0..n rank axis.)
    axT = axB.twinx()
    cdf = np.arange(1, n + 1) / n
    axT.step(np.arange(n), cdf, where="post", color=NAVY, lw=1.4, zorder=5)
    axT.set_ylim(0, 1.0)
    axT.set_ylabel("cumulative donor fraction", fontsize=7.6, color=NAVY_DARK, rotation=270,
                   labelpad=12)
    axT.tick_params(axis="y", labelsize=7.0, colors=NAVY_DARK, length=3, width=0.8)
    axT.spines["right"].set_visible(True)
    axT.spines["right"].set_color(NAVY_DARK)
    axT.spines["right"].set_linewidth(0.8)
    axT.spines["top"].set_visible(False)
    axT.set_zorder(axB.get_zorder() + 1)
    axT.patch.set_visible(False)

    expo = int(np.floor(np.log10(pw)))
    mant = pw / (10 ** expo)
    p_txt = f"{mant:.1f} × 10$^{{{_MINUS}{abs(expo)}}}$"
    lines = [(f"{wins} of {n} donors", "head"),
             ("over the cell-mean floor", "sub"),
             (f"mean gap {fmt_signed(summ['mean'])} "
              f"[{fmt_signed(summ['lo'])}, {fmt_signed(summ['hi'])}]", "body"),
             (f"paired Wilcoxon p = {p_txt}", "body")]
    y0 = 0.95
    for txt, kind in lines:
        if kind == "head":
            axB.text(0.028, y0, txt, transform=axB.transAxes, ha="left", va="top",
                     fontsize=11.0, fontweight="bold", color=NAVY_DARK, zorder=8)
            y0 -= 0.115
        elif kind == "sub":
            axB.text(0.028, y0, txt, transform=axB.transAxes, ha="left", va="top",
                     fontsize=7.6, color=GREY_MID, zorder=8)
            y0 -= 0.10
        else:
            axB.text(0.028, y0, txt, transform=axB.transAxes, ha="left", va="top",
                     fontsize=7.6, color=INK, zorder=8)
            y0 -= 0.082
    axB.text(0.30, summ["mean"], f"mean {fmt_signed(summ['mean'], 3)}", ha="center", va="bottom",
             fontsize=7.0, color=NAVY_DARK, fontweight="bold", zorder=8,
             transform=axB.get_yaxis_transform())
    return axT


# ============================================================================================
# LANDSCAPE DRAW
# ============================================================================================
# gutter geometry (data units, in the landscape axes). Read left->right:
#   [family word] [accent bar] [model name] | [block A cells] | [gap] | [block B cells]
NAME_X = -0.70
BAR_X1, BAR_X0 = -3.30, -3.45
FAMLAB_RX = -3.60
GUT_LEFT = -4.80
BGAP = 0.95                                  # gap between block A and block B
BOFF = 14.5 + BGAP + 0.5                      # x of block B's first column centre
YHEAD = 1.95                                 # headroom above cells (bands + axis rule + header)
YFOOT = -1.70                                # footroom below cells (2-line ticks)


def _xcol(block, j):
    return j if block == "A" else BOFF + j


def _draw_cell(ax, x, y, score, action, margin, mnorm, style="solid"):
    """margin = this cell's per-cell margin over the binding simple-baseline floor member (None only
    if the column carries no floor member). Soft diverging fill: medium orange below the floor, white
    at it, blue above it (light blue for a weak win). The printed number is always the raw
    Pearson-Δ. In-cell ink is near-black on every shade of this soft map (medium blue and medium
    orange both stay light enough for black text); white only if a fill falls below LUM_T.
    style: 'solid' (conditioned/baseline), 'dashed' (deterministic side-info: dashed grey edge,
    margin fill kept), 'hollow' (CINEMA-OT agnostic reference: no margin fill, outline + number only)."""
    if score is None:
        ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, fc=NA_FC, ec=CELL_EC, lw=1.0, zorder=1))
        return
    if style == "hollow":              # CINEMA-OT: agnostic reference, not a conditioned-model verdict
        ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, fc="white", ec=DIAG_EC, lw=1.1,
                               linestyle=(0, (3, 2)), zorder=1))
        ax.text(x, y, _fmt(score), ha="center", va="center", fontsize=7.0, color=GREY_MID, zorder=4)
        return
    if margin is None:
        fill, dark = "white", False
    else:
        fill = DIV_CMAP(mnorm(margin))
        dark = _lum(fill) < LUM_T      # near-black ink on every shade in this soft map
    ec, lw, ls = CELL_EC, 1.0, "solid"
    if style == "dashed":              # deterministic side-info predictor: dashed grey edge, fill kept
        ec, lw, ls = DIAG_EC, 1.1, (0, (3, 2))
    ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, fc=fill, ec=ec, lw=lw, linestyle=ls, zorder=1))
    txt_c = "white" if dark else INK
    ax.text(x, y, _fmt(score), ha="center", va="center", fontsize=7.0, color=txt_c, zorder=4)
    if action == "adapted":                  # thin corner notch (upper-right), cell-bounded
        nk = 0.26
        ax.plot([x + .46 - nk, x + .46], [y + .46, y + .46 - nk], color=ADAPT_C, lw=0.9,
                zorder=3, solid_capstyle="round", clip_on=True)


def draw_landscape(ax, cells, models, fam_of, cell_margin, mnorm):
    nM = len(models)
    yof = {m: nM - 1 - i for i, m in enumerate(models)}   # model -> row y (top row highest y)

    # ---- cells for both blocks ----
    for block, cols in (("A", BLOCK_A), ("B", BLOCK_B)):
        for j, (ck, *_rest) in enumerate(cols):
            x = _xcol(block, j)
            for m in models:
                y = yof[m]
                sa = cells.get((m, ck))
                _style = "dashed" if m in DASHED_ROWS else ("hollow" if m in HOLLOW_ROWS else "solid")
                _draw_cell(ax, x, y, None if sa is None else sa[0],
                           None if sa is None else sa[1], cell_margin.get((m, ck)), mnorm, _style)

    # ---- gold ring + star on the floor-crossing verdict cells ----
    # The star sits in the ring's TOP-LEFT interior corner (clear of the centred number and the
    # adapted notch), NOT above the ring — above would land on the neighbouring row's data cells.
    for model, ckeys in VERDICTS:
        y = yof[model]
        xs = []
        for block, cols in (("A", BLOCK_A), ("B", BLOCK_B)):
            for j, (ck, *_r) in enumerate(cols):
                if ck in ckeys:
                    xs.append(_xcol(block, j))
        x0, x1 = min(xs), max(xs)
        ax.add_patch(Rectangle((x0 - .5 + .04, y - .5 + .04), (x1 - x0) + 1 - .08, 1 - .08,
                               fc="none", ec=WIN_RING, lw=2.2, zorder=7, joinstyle="round"))
        ax.scatter([x0 - .5 + .19], [y + .5 - .19], marker="*", s=66, c=WIN_RING,
                   edgecolors=WIN_DARK, linewidths=0.5, zorder=9, clip_on=True)

    # ---- left gutter: family word | accent bar | model name ; floor-row flag ----
    for m in models:
        y = yof[m]
        ax.text(NAME_X, y, MODEL_SHORT.get(m, m), ha="right", va="center", fontsize=7.4,
                color=INK, clip_on=False, zorder=6)
        if m in FLOOR_ROWS:                   # mark the two universal floor members
            ax.text(NAME_X - 0.02, y - 0.34, "floor", ha="right", va="center", fontsize=5.4,
                    color=SLATE_BAND, style="italic", clip_on=False, zorder=6)
    # role accent: ONE bar per role group (Figure 1 palette). Role reads off colour + row order +
    # the single legend key; no family/role text in the gutter, so the only per-row label is the model.
    yacc = nM - 1
    _present = [(ROLE_OF[fam], len([m for m in ms if m in models]))
                for fam, ms in FAM_ROWS if any(m in models for m in ms)]
    _spans = []
    for role, n in _present:
        if _spans and _spans[-1][0] == role:
            _spans[-1][1] += n
        else:
            _spans.append([role, n])
    for role, n in _spans:
        y_hi, y_lo = yacc + .5, yacc - n + .5
        ax.add_patch(Rectangle((BAR_X0, y_lo + .14), BAR_X1 - BAR_X0, (y_hi - y_lo) - .28,
                               fc=ROLE_COLOR[role], ec="none", clip_on=False, zorder=5))
        yacc -= n

    # ---- per-block: T-code/dataset band, axis-group header rule, 2-line ticks, separators ----
    band_lo, band_hi = nM - .5 + 0.10, nM - .5 + 0.52
    rule_y = nM - .5 + 0.80
    head_y = rule_y + 0.14
    tick_y = -0.5 - 0.18

    def draw_block_chrome(block, cols, axis_groups):
        n = len(cols)
        # contiguous (T-code, dataset) bands. The band carries the dataset name ONLY where its span
        # is wide enough to hold it (>= 3 cols: Kang, OP3, CRISPR); narrow single/double-column tasks
        # carry the bare T-code (the dataset reads off the tick: Soskic / compound, or the caption),
        # so no band label ever overflows its slate.
        k = 0
        while k < n:
            tcode, dset = cols[k][1], cols[k][2]
            j = k
            while j < n and cols[j][1] == tcode and cols[j][2] == dset:
                j += 1
            span = j - k
            x0 = _xcol(block, k) - .5 + 0.05
            x1 = _xcol(block, j - 1) + .5 - 0.05
            ax.add_patch(Rectangle((x0, band_lo), x1 - x0, band_hi - band_lo, fc=SLATE_BAND,
                                   ec="none", zorder=5, clip_on=False))
            if span >= 3:
                lab, fs = f"{tcode} · {dset}", 6.8
            elif span == 2:
                lab, fs = f"{tcode} · {DSET_ABBR.get(dset, dset)}", 6.0
            else:
                lab, fs = tcode, 7.0
            ax.text((x0 + x1) / 2, (band_lo + band_hi) / 2, lab, ha="center", va="center",
                    fontsize=fs, color="white", fontweight="bold", zorder=6, clip_on=False)
            k = j
        # ticks (lineage / split) + thin column separators
        for j, c in enumerate(cols):
            ax.text(_xcol(block, j), tick_y, c[3], ha="right", va="top", rotation=42,
                    rotation_mode="anchor", fontsize=6.6, color="#222", clip_on=False, zorder=6)
        for j in range(1, n):
            xs = _xcol(block, j) - .5
            ax.plot([xs, xs], [-.5, nM - .5], color="#d6d9dc", lw=0.7, zorder=2)
        # axis-group header rule + bold label (lean adjacent narrow headers off the shared seam)
        for label, a, b, lean in axis_groups:
            x0 = _xcol(block, a) - .5 + 0.06
            x1 = _xcol(block, b) + .5 - 0.06
            ax.plot([x0, x1], [rule_y, rule_y], color=NAVY_DARK, lw=1.6, clip_on=False, zorder=6)
            if lean == "l":
                lx, lha = x1, "right"
            elif lean == "r":
                lx, lha = x0, "left"
            else:
                lx, lha = (x0 + x1) / 2, "center"
            ax.text(lx, head_y, label, ha=lha, va="bottom", fontsize=8.0, fontweight="bold",
                    color=NAVY_DARK, clip_on=False, zorder=6)

    draw_block_chrome("A", BLOCK_A, AXIS_A)
    draw_block_chrome("B", BLOCK_B, AXIS_B)

    # block divider rule between A and B
    xdiv = (14.5 + BOFF - 0.5) / 2
    ax.plot([xdiv, xdiv], [YFOOT + 0.15, rule_y + 0.30], color="#c4c9ce", lw=0.9, zorder=2)

    ax.set_xlim(GUT_LEFT, BOFF + (len(BLOCK_B) - 1) + 0.5 + 0.12)
    ax.set_ylim(YFOOT, nM - .5 + YHEAD)
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("none")
    for sp in ax.spines.values():
        sp.set_visible(False)


# ============================================================================================
# COMPOSE
# ============================================================================================
def verify_census_coverage(cells):
    """Hardened gate (the guard that would have caught the original Figure-2 under-population): assert
    every (model, task) evaluation in the cross_cluster_headline census appears in the grid, and that all
    14 distinct census models are drawn. Coverage, not equality (the grid may carry extra context cells,
    e.g. CellOT on Kang). Fails loudly on any census evaluation the grid silently omits."""
    p = _resolve(GH / "results" / "_paper" / "cross_cluster_headline.csv",
                 BM / "results" / "_paper" / "cross_cluster_headline.csv")
    if p is None:
        return
    df = pd.read_csv(p)
    drawn_cols = {ck for (_m, ck) in cells}
    drawn_models = {m for (m, _c) in cells}

    def expected(cluster, split):
        if cluster == "C2":
            return {"C2·Soskic"}
        if cluster == "C5" and "compound" in str(split):
            return {"C5·cpd"}
        if cluster == "C5":
            return {c for c in drawn_cols if c.startswith("C5·") and c != "C5·cpd"}
        return {c for c in drawn_cols if c.startswith(cluster + "·")}

    missing = [f"{r.model}@{r.cluster}/{r.split}" for _, r in df.iterrows()
               if not any((r.model, c) in cells for c in expected(r.cluster, r.split))]
    assert not missing, "Figure 2 omits census evaluations (silent under-population): " + ", ".join(missing)
    assert df.model.nunique() == 14, f"census has {df.model.nunique()} models, expected 14"
    assert set(df.model) <= drawn_models, f"census models not drawn: {sorted(set(df.model) - drawn_models)}"


def main():
    set_pub_style()
    plt.rcParams["axes.unicode_minus"] = True

    win = verify_verdicts()
    cells = load_cells()
    verify_census_coverage(cells)   # hardened gate: every census (model,task) must appear in the grid
    gaps, wins, n, summ, pw = load_donor_gaps()

    models = [m for _, ms in FAM_ROWS for m in ms
              if any((m, ck) in cells for ck, *_ in (BLOCK_A + BLOCK_B))]
    fam_of = {m: f for f, ms in FAM_ROWS for m in ms}
    nM = len(models)

    # ---- per-cell floor margins (soft diverging fill, centred on the floor) ----------------------
    # EVERY cell is coloured by its OWN per-cell margin over its column's binding floor, with NO
    # clamping: a weak win keeps its small positive value and reads light BLUE; a loss keeps its
    # negative value and reads orange that deepens with how far below the floor it sits. The binding
    # floor for a column = max(cell-mean shift, linear-PCA shift) Pearson-Δ.
    floor_by_col: dict[str, float] = {}
    for ck, *_rest in (BLOCK_A + BLOCK_B):
        fv = [cells[(fm, ck)][0] for fm in FLOOR_ROWS if (fm, ck) in cells]
        if fv:
            floor_by_col[ck] = max(fv)
    cell_margin: dict[tuple[str, str], float] = {}
    for (model, ck), (score, _a) in cells.items():
        if ck in floor_by_col:
            cell_margin[(model, ck)] = score - floor_by_col[ck]

    # OFFICIAL split-level winners = the gold-ringed (model, col) cells from cross_cluster_headline.csv
    # (CellOT on the Soskic donor split, FP-ridge on the four OP3 cell-context lineages). The old
    # "exactly 5 cells positive" assertion no longer holds once weak per-column wins are un-clamped;
    # self-verify on the OFFICIAL-WINNER count instead (two task-level verdicts, five ringed cells),
    # and confirm every ringed cell is genuinely above its column floor.
    official_cells = {(model, ck) for model, ckeys in VERDICTS for ck in ckeys}
    for mc in sorted(official_cells):
        assert cell_margin.get(mc, -1.0) > 0, f"official winner {mc} is not above its column floor"
    assert len(VERDICTS) == 2, f"expected two official split-level verdicts, got {len(VERDICTS)}"
    assert len(official_cells) == 5, \
        f"expected 5 gold-ringed cells (1 donor + 4 OP3 lineages), got {len(official_cells)}"
    assert len(win) == 2, \
        f"cross_cluster_headline must list exactly two beats-both-floor splits, got {len(win)}"

    # HONESTY CHECK (printed): after un-clamping, does any NON-official-winner cell carry a larger
    # positive per-cell margin than some official winner? If so it is a legitimate per-column win
    # (beats the binding floor on that lineage without officially winning the aggregated split); the
    # gold rings + the per-column colorbar framing keep that honest. Surface it, do not hide it.
    pos_cells = {mc: v for mc, v in cell_margin.items() if v > 0}
    win_min = min(cell_margin[mc] for mc in official_cells)     # weakest official-winner cell
    nonwin_over = {mc: v for mc, v in pos_cells.items()
                   if mc not in official_cells and v > win_min}
    print("  honesty: %d positive cells (%d official-winner, %d sub-verdict blue)"
          % (len(pos_cells), len(official_cells), len(pos_cells) - len(official_cells)))
    if nonwin_over:
        print("  honesty: non-winner cell(s) above the weakest official winner (%.4f):" % win_min)
        for (mdl, ck), v in sorted(nonwin_over.items(), key=lambda kv: -kv[1]):
            print(f"           {mdl} @ {ck}: +{v:.4f}  (no ring; legitimate per-column floor-beat)")

    # soft diverging norm: medium orange (below floor) | white at 0 (the floor) | medium blue (above
    # it). Blue saturates at the LARGEST positive per-cell margin (which is an official winner, so the
    # deepest blue reads as a verdict); orange saturates at a robust low percentile of the NEGATIVE
    # margins (≈0.56) so the bulk of losses spread across the orange range and the single far outlier
    # (ctrl-pred) does not flatten everyone into one shade.
    pos_margins = [v for v in cell_margin.values() if v > 0]
    neg_margins = [v for v in cell_margin.values() if v < 0]
    vmax_margin = max(pos_margins)
    neg_clip = abs(float(np.percentile(neg_margins, NEG_PCTILE)))
    mnorm = TwoSlopeNorm(vmin=-neg_clip, vcenter=0.0, vmax=vmax_margin)

    # ---- figure geometry (inches) ----
    xspan = (BOFF + (len(BLOCK_B) - 1) + 0.5 + 0.12) - GUT_LEFT
    yspan = (nM - .5 + YHEAD) - YFOOT
    cell = 0.286                                    # inch per data unit (square cells)
    land_w = cell * xspan
    land_h = cell * yspan

    m_left = 0.12
    m_right = 0.16
    m_top = 0.58                                    # title + subtitle band above the landscape
    legend_h = 1.00                                 # colorbar/marker keys + subtitle (upper) + role-key line (lower); sized for the full 18-row (14-model) grid
    gap_a_leg = 0.16
    gap_leg_b = 0.68                                # clear air so the legend never touches 2b's title
    donor_h = 1.95
    donor_xlab = 0.44
    m_bot = 0.34

    fig_w = m_left + land_w + m_right
    fig_h = m_top + land_h + gap_a_leg + legend_h + gap_leg_b + donor_h + donor_xlab + m_bot

    fig = plt.figure(figsize=(fig_w, fig_h))

    # landscape axes (2a)
    xA = m_left / fig_w
    yA = 1.0 - (m_top + land_h) / fig_h
    wA = land_w / fig_w
    hA = land_h / fig_h
    ax2a = fig.add_axes([xA, yA, wA, hA])
    ax2a.set_anchor("NW")
    ax2a.set_clip_on(False)
    draw_landscape(ax2a, cells, models, fam_of, cell_margin, mnorm)

    # donor-CDF axes (2b) — full text-width, docked at the foot
    xB = (m_left + 0.40) / fig_w
    wB = (land_w - 0.95) / fig_w
    hB = donor_h / fig_h
    yB = (m_bot + donor_xlab) / fig_h
    ax2b = fig.add_axes([xB, yB, wB, hB])
    draw_donor_panel(ax2b, gaps, wins, n, summ, pw)

    # ---- panel letters + panel titles (ONE shared hierarchy) -----------------------------------
    # Both panel letters share a left edge (PLX); both panel titles share a left edge (PTX) and ONE
    # size (FS_PANEL_TITLE), each sitting on its own letter's baseline. (a) was previously a large
    # centred title and (b) a smaller left title — now they match in size and alignment.
    cx_cells = xA + (0 - 0.5 - GUT_LEFT) / xspan * wA      # left edge of block-A cells (fig frac)
    PLX = xA - 0.002                                       # panel-letter left edge (shared)
    PTX = PLX + 0.024                                      # panel-title left edge (shared)
    bbB = ax2b.get_position()
    a_base_y = yA + hA + 0.040                             # (a) letter + title baseline
    a_sub_y = a_base_y - 0.165 / fig_h                     # (a) interpretive subtitle, just below
    b_base_y = bbB.y1 + 0.030                              # (b) letter + title baseline

    fig.text(PLX, a_base_y, "a", fontsize=FS_PANEL_LETTER, fontweight="bold",
             ha="left", va="bottom", color=INK)
    fig.text(PTX, a_base_y,
             r"Method $\times$ task performance landscape (response-direction Pearson-$\Delta$)",
             ha="left", va="bottom", fontsize=FS_PANEL_TITLE, fontweight="bold", color=NAVY_DARK)
    fig.text(PTX, a_sub_y,
             "Blue beats the binding floor on that column; gold rings = the two cell-context / "
             "donor split-level verdicts; the unseen-perturbation block has none.",
             ha="left", va="top", fontsize=FS_SUBTITLE, color=GREY_MID)

    fig.text(PLX, b_base_y, "b", fontsize=FS_PANEL_LETTER, fontweight="bold",
             ha="left", va="bottom", color=INK)
    fig.text(PTX, b_base_y, "CellOT crosses the donor barrier",
             fontsize=FS_PANEL_TITLE, fontweight="bold", ha="left", va="bottom", color=NAVY_DARK)

    # left shared axis title for the landscape
    fig.text(0.008, yA + hA / 2, "method (grouped by role)", rotation=90, ha="left",
             va="center", fontsize=FS_AXIS_LABEL, color=GREY_MID)

    # ================= legend strip between 2a and 2b =================
    # Stacked rows within the strip:
    #   band_mid  = colorbar + cell-marker keys (upper), colorbar subtitle sits just under the bar
    #   band_role = ONE compact role-key line, on its own baseline clearly BELOW the subtitle
    leg_y0 = (m_bot + donor_xlab + donor_h + gap_leg_b) / fig_h
    band_mid = leg_y0 + (legend_h * 0.72) / fig_h

    # colorbar (floor-crossing MARGIN) on the left, aligned under block-A cells
    cb_x0 = cx_cells
    cb_w = 0.182
    cb_h = 0.012
    cb_y0 = band_mid - cb_h / 2
    sm = plt.cm.ScalarMappable(cmap=DIV_CMAP, norm=mnorm)
    sm.set_array([mnorm.vmin, mnorm.vmax])
    cax = fig.add_axes([cb_x0, cb_y0, cb_w, cb_h])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([mnorm.vmin, 0.0, mnorm.vmax])
    cb.set_ticklabels([_fmt(mnorm.vmin), ".00", "+" + _fmt(mnorm.vmax)])
    cb.ax.tick_params(labelsize=7.0, length=0, pad=2)
    cb.outline.set_visible(True)                   # frame the bar so the white centre reads on the page
    cb.outline.set_edgecolor(CELL_EC); cb.outline.set_linewidth(0.6)
    _tl = cb.ax.get_xticklabels()
    if len(_tl) >= 3:
        _tl[0].set_ha("left"); _tl[1].set_ha("center"); _tl[-1].set_ha("right")
    fig.text(cb_x0, cb_y0 + cb_h + 0.009,
             "cell fill = per-cell margin (Pearson-Δ): printed cell value − binding floor, per column",
             fontsize=7.4, ha="left", va="bottom", color=GREY_MID)
    subtitle_y = cb_y0 - 0.018
    sub_artist = fig.text(cb_x0, subtitle_y,
                          "blue = above the floor;  white = at the floor;  orange = below it",
                          fontsize=7.0, ha="left", va="top", color=GREY_MID)
    # the role key gets its OWN baseline a clean step below the subtitle's rendered bottom
    fig.canvas.draw()
    _rend = fig.canvas.get_renderer()
    sub_bottom = sub_artist.get_window_extent(_rend).y0 / (fig_h * fig.dpi)
    band_role = sub_bottom - 0.018

    # keys to the right of the colorbar, on one baseline
    sww = 0.0135
    swh = sww * fig_w / fig_h
    lab_dx = sww + 0.008

    def _key(xt, draw, text, two_line=False):
        a = fig.add_axes([xt, band_mid - swh / 2, sww, swh])
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
        draw(a)
        fig.text(xt + lab_dx, band_mid, text, fontsize=7.0, ha="left",
                 va="center", color=GREY_MID, linespacing=1.0)

    def _d_win(a):
        a.add_patch(Rectangle((0, 0), 1, 1, fc=DIV_CMAP(mnorm(vmax_margin)), ec=CELL_EC, lw=0.8))
        a.add_patch(Rectangle((0.08, 0.08), 0.84, 0.84, fc="none", ec=WIN_RING, lw=1.8))
        a.scatter([0.5], [1.18], marker="*", s=55, c=WIN_RING, edgecolors=WIN_DARK,
                  linewidths=0.5, clip_on=False)

    def _d_adapt(a):
        a.add_patch(Rectangle((0, 0), 1, 1, fc="white", ec=CELL_EC, lw=0.8))
        a.plot([1 - 0.42, 1], [1, 1 - 0.42], color=ADAPT_C, lw=1.1, solid_capstyle="round")

    def _d_na(a):
        a.add_patch(Rectangle((0, 0), 1, 1, fc=NA_FC, ec=LEGEND_EC, lw=0.8))

    kx0 = cb_x0 + cb_w + 0.050
    kx = kx0
    _key(kx, _d_win, "beats both floor members\n(cell-mean & linear-PCA shift)")
    kx += lab_dx + 0.232
    _key(kx, _d_adapt, "adapted (re-fit)")
    kx += lab_dx + 0.120
    _key(kx, _d_na, "not applicable")

    # ---- ONE compact role key: four little swatches + names in the four role colours, on a single
    # line on its OWN baseline below the subtitle (mirrors Figure 1's ROLES box so the gutter accents
    # read). Same swatch size as the other legend swatches; one even pitch across the four.
    rx = cb_x0
    fig.text(rx, band_role, "row accent:", fontsize=7.0, ha="left", va="center",
             color=GREY_MID, fontstyle="italic")
    rx += 0.066
    for role, name in ROLE_KEY:
        a = fig.add_axes([rx, band_role - swh / 2, sww, swh])
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
        a.add_patch(Rectangle((0, 0), 1, 1, fc=ROLE_COLOR[role], ec="none"))
        fig.text(rx + lab_dx, band_role, name, fontsize=7.0, ha="left", va="center", color=GREY_MID)
        rx += lab_dx + 0.014 + 0.0098 * len(name)

    # ---- geometry assertions (content stays on-canvas; cursor stays above the axis floor) ----
    assert yA > 0, f"landscape axes underflow (yA={yA:.3f})"
    assert leg_y0 + legend_h / fig_h < yA, "legend strip overlaps the landscape"
    assert bbB.y1 < leg_y0, "donor panel overlaps the legend strip"
    assert kx + lab_dx + 0.110 < 1.0, f"legend keys overflow right edge (x={kx:.3f})"
    assert rx + lab_dx < 1.0, f"role key overflows right edge (x={rx:.3f})"
    assert band_role - swh / 2 > leg_y0, "role key drops below the legend strip"
    assert band_role + swh / 2 < sub_bottom - 0.006, "role key crowds the colorbar subtitle"

    out_dir = ROOT / "results" / "_paper"
    base = out_dir / "figure2_landscape_verdict"
    fig.savefig(base.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    # ---- flatten the raster outputs onto opaque white so the deposited PNG/TIFF are RGB (no alpha) ----
    from PIL import Image
    for suf in (".png", ".tiff"):
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
    print(f"wrote {base}.png/.pdf/.tiff   ({fig_w:.2f} x {fig_h:.2f} in; cell {cell} in; {nM} rows)")
    print(f"  2a verdict cells: CellOT @ T2 Soskic donor; FP-ridge @ T5 OP3 cell-context")
    print(f"  2b donor: {wins}/{n} over cell-mean floor; mean {summ['mean']:+.3f} "
          f"[{summ['lo']:+.3f},{summ['hi']:+.3f}]; paired Wilcoxon p={pw:.2e}")


if __name__ == "__main__":
    main()
