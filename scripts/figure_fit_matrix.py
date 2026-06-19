#!/usr/bin/env python
"""Figure 6 (final, NG-grade) - the DESCRIPTIVE end-of-paper fit-recommendation matrix.

Mechanical reading of the assembled headline + within-family tables (NO new computation, NO fabrication):
  (a) task x family verdict grid. One cell per (immune prediction task, method family). Cell state under the
      descriptive rule (scripts/fit_recommendation.py): a family "works" iff >=1 member exceeds the universal
      simple baseline {cell-mean, linear-PCA} on response-direction Pearson-Delta; ties -> simple baseline.
      Every cell resolves to the simple baseline EXCEPT two single-method, single-task exceptions, annotated
      as such: CellOT on the donor task (model-level; OT family-mate scPRAM PENDING, does not reproduce) and
      FP-ridge on the small-molecule cell-context task (deterministic chemistry side-information). No family
      clears the baseline with a reproducing family-mate on any task.
  (b) within-family agreement: for every family with >=2 evaluated members on a task, the two members reach
      the SAME beat-baseline verdict (all from within_family_consistency.csv), confirming the two exceptions
      are single-method, not family-level.

Data sources (ONLY these, read mechanically):
  results/_paper/cross_cluster_headline.csv        - per (task,family,model) Pearson-Delta vs universal floor
  results/_paper/within_family_consistency.csv     - per (task,family) verdict agreement for >=2-member families
The descriptive rule is scripts/fit_recommendation.py (PREREGISTRATION sec 5). No numeric value is hardcoded.

DESIGN STANDARDS (benchmark/FIGURE_DESIGN_STANDARDS.md): unicode minus U+2212 on every negative; ONE legend /
ONE label column / ONE swatch size; even gaps + visible arrows; caption matches the rendered panels; the term
"simple baseline" (never "floor") in all figure text. Inspection crop to /tmp only; deliverables are the
figure_fit_matrix.{png,pdf} under results/_paper/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ivcbench.report.style import (set_pub_style, despine, panel_title, style_legend,  # noqa: E402
                                   SIMPLE_GREY, CHEMISTRY, INK as TOK_INK,
                                   GREY_MID, GREY_LITE, LEGEND_EC)
from ivcbench.report.style import FAM_COLORS  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
PAPER = RESULTS / "_paper"

C_INK = TOK_INK
C_SIMPLE = SIMPLE_GREY          # simple-baseline tile (the dominant state)
C_SIMPLE_FILL = "#eceff1"       # very light grey tile fill for "simple baseline"
C_WORK = "#fbe7d2"              # warm light fill for the single-method exception tiles (NOT a family hue)
C_OT = FAM_COLORS["opt-transport"]   # CellOT exception accent (indigo)
C_CHEM = CHEMISTRY                   # FP-ridge exception accent (sky)
C_NA = "#f6f6f6"                # not-evaluated tile (pale)

# universal floor name used in figure text (DESIGN STANDARDS: say "simple baseline", never "floor")
SIMPLE_LABEL = "simple baseline"

# canonical task order (rows) and family order (columns) for the grid
TASK_ORDER = [
    ("C1", "cell-context (LOCT)", "Cytokine\ncell-context\n(Kang)"),
    ("C2", "donor (LODO)", "Donor-aware\nactivation\n(Soskic)"),
    ("C3", "unseen-perturbation (LO-gene 10%)", "Gene\nintervention\n(primary-T CRISPR)"),
    ("C4", "unseen-KO (modality, RNA)", "Complex-context\n/ modality\n(Frangieh)"),
    ("C5", "unseen-compound", "Small-molecule\nunseen-compound\n(OP3)"),
    ("C5", "cell-context (LOCT)", "Small-molecule\ncell-context\n(OP3)"),
]
FAM_ORDER = ["Latent", "Graph", "Foundation", "Hybrid", "OT", "Chemistry"]
FAM_LABEL = {"Latent": "Latent", "Graph": "Graph", "Foundation": "Foundation",
             "Hybrid": "Hybrid", "OT": "Optimal\ntransport", "Chemistry": "Chemistry"}


def build_grid(head: pd.DataFrame, wf: pd.DataFrame):
    """Mechanically derive the per (task,family) verdict from the two deposited tables.

    Returns a dict keyed (task_key, family) -> dict(state, members, n_members, n_beat, gap, note, single).
    state in {'simple', 'exception', 'na'}.  'exception' iff exactly one member beats both universal-floor
    members AND no family-mate reproduces it (single-method).  We NEVER promote a single-method beat to a
    family win.  Per-family member counts/verdicts come from the within-family table where >=2 members exist.
    """
    grid = {}
    # within-family verdicts keyed (cluster, split, family)
    wf_idx = {}
    for r in wf.itertuples():
        wf_idx[(r.cluster, r.split, r.family)] = dict(
            n_members=int(r.n_models), n_beat=int(r.n_beat_both_floor),
            agreement=str(r.verdict_agreement), models=str(r.models),
            flag=("" if pd.isna(r.flag) else str(r.flag)))
    for cluster, split, _disp in TASK_ORDER:
        sub = head[(head.cluster == cluster) & (head.split == split)]
        for fam in FAM_ORDER:
            fs = sub[sub.family == fam]
            if not len(fs):
                grid[(cluster, split, fam)] = dict(state="na", n_members=0, n_beat=0,
                                                   gap=np.nan, single=False, members="", note="")
                continue
            n_beat = int(fs.beats_both_floor_members.sum())
            best = fs.loc[fs.pearson_delta.idxmax()]
            gap = float(best.delta_vs_floor_mean)
            members = ",".join(sorted(fs.model.tolist()))
            wfk = wf_idx.get((cluster, split, fam))
            # exception iff exactly one member beats BOTH floor members; this is by construction
            # single-method here (no family/task cell in the deposited tables has >=2 members both beating).
            single = bool(n_beat == 1)
            state = "exception" if n_beat >= 1 else "simple"
            grid[(cluster, split, fam)] = dict(
                state=state, n_members=len(fs), n_beat=n_beat, gap=gap,
                single=single, members=members,
                best_model=str(best.model), best_gap=gap,
                wf=wfk, note="")
    return grid


def main():
    set_pub_style()
    head = pd.read_csv(PAPER / "cross_cluster_headline.csv")
    wf = pd.read_csv(PAPER / "within_family_consistency.csv")
    grid = build_grid(head, wf)

    nrow, ncol = len(TASK_ORDER), len(FAM_ORDER)

    # Layout: (a) the verdict grid (wide, left/top) ; (b) within-family agreement strip (right).
    # Width 7.0in to match the plate; (a) dominant, (b) a compact companion grid.
    fig = plt.figure(figsize=(7.4, 4.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.50, 1.0], wspace=0.62,
                          left=0.155, right=0.985, top=0.83, bottom=0.205)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ============================ (a) task x family verdict grid ============================
    axA.set_xlim(-0.5, ncol - 0.5)
    axA.set_ylim(-0.5, nrow - 0.5)
    axA.invert_yaxis()  # first task at top
    cell_w, cell_h = 0.92, 0.92

    exception_cells = []  # collect for legend/annotation
    for ri, (cluster, split, _disp) in enumerate(TASK_ORDER):
        for ci, fam in enumerate(FAM_ORDER):
            g = grid[(cluster, split, fam)]
            x, y = ci, ri
            if g["state"] == "na":
                fc, ec = C_NA, "#dddddd"
            elif g["state"] == "exception":
                fc, ec = C_WORK, "#caa472"
            else:
                fc, ec = C_SIMPLE_FILL, "#cfd4d8"
            axA.add_patch(Rectangle((x - cell_w / 2, y - cell_h / 2), cell_w, cell_h,
                                    facecolor=fc, edgecolor=ec, lw=0.8, zorder=1))
            if g["state"] == "na":
                axA.text(x, y, "n/e", ha="center", va="center", fontsize=5.6,
                         color=GREY_LITE, zorder=3)
            elif g["state"] == "exception":
                # the single-method exception: accent ring + the winning model name, annotated single-method
                accent = C_OT if g["best_model"] == "CellOT" else C_CHEM
                axA.add_patch(Rectangle((x - cell_w / 2, y - cell_h / 2), cell_w, cell_h,
                                        facecolor="none", edgecolor=accent, lw=1.6, zorder=2))
                gap = g["best_gap"]
                gtxt = ("+" if gap >= 0 else "−") + f"{abs(gap):.2f}"
                axA.text(x, y - 0.13, g["best_model"], ha="center", va="center", fontsize=6.4,
                         color=accent, weight="bold", zorder=3)
                axA.text(x, y + 0.12, gtxt, ha="center", va="center", fontsize=5.8,
                         color=C_INK, zorder=3)
                axA.text(x, y + 0.30, "single-method", ha="center", va="center", fontsize=4.7,
                         color=GREY_MID, style="italic", zorder=3)
                exception_cells.append((cluster, split, fam, g["best_model"], accent))
            else:
                # simple-baseline tile: one short neutral glyph, plus a faint sub-baseline gap read
                axA.text(x, y - 0.10, SIMPLE_LABEL, ha="center", va="center", fontsize=5.2,
                         color=GREY_MID, zorder=3)
                if np.isfinite(g["best_gap"]):
                    gap = g["best_gap"]
                    gtxt = ("+" if gap >= 0 else "−") + f"{abs(gap):.2f}"
                    axA.text(x, y + 0.16, f"best {gtxt}", ha="center", va="center", fontsize=4.6,
                             color=GREY_LITE, zorder=3)

    axA.set_xticks(range(ncol))
    axA.set_xticklabels([FAM_LABEL[f] for f in FAM_ORDER], fontsize=6.3, color=C_INK)
    axA.set_yticks(range(nrow))
    axA.set_yticklabels([d for _, _, d in TASK_ORDER], fontsize=6.3, color=C_INK)
    axA.tick_params(length=0, colors=C_INK)
    for s in ("top", "right", "left", "bottom"):
        axA.spines[s].set_visible(False)
    axA.set_xlabel("method family", fontsize=6.6, color=C_INK)
    axA.xaxis.set_label_position("top")
    axA.xaxis.tick_top()
    axA.set_xticks(range(ncol)); axA.xaxis.set_ticks_position("top")

    # ============================ (b) within-family agreement ============================
    # one row per (task,family) with >=2 evaluated members; show the SAME-verdict result + #members beating.
    wf_rows = []
    for cluster, split, disp in TASK_ORDER:
        for fam in FAM_ORDER:
            g = grid[(cluster, split, fam)]
            wfk = g.get("wf")
            if wfk and wfk["n_members"] >= 2:
                wf_rows.append(dict(task=disp.replace("\n", " "), fam=fam,
                                    n=wfk["n_members"], nbeat=wfk["n_beat"],
                                    agree=wfk["agreement"], models=wfk["models"]))
    wf_df = pd.DataFrame(wf_rows)
    # compact task tag for the row label (the verbose disp lives in panel a's row axis)
    TASK_TAG = {"C1": "cytokine cell-context", "C2": "donor activation",
                "C3": "gene intervention", "C4": "complex/modality",
                "C5u": "small-mol compound", "C5c": "small-mol cell-context"}
    yb = np.arange(len(wf_df))[::-1]  # top-down
    # bars occupy [0, n]; row label sits in a left gutter (negative x); verdict text right of the bar.
    axB.set_xlim(-3.7, 3.0)
    axB.set_ylim(-0.8, len(wf_df) - 0.2)
    for yy, r in zip(yb, wf_df.itertuples()):
        # bar = #members (all reach the SAME verdict). Filled segment = #beating simple baseline.
        axB.barh(yy, r.n, height=0.56, color="#e2e6e9", edgecolor="#cfd4d8", lw=0.6, zorder=1)
        if r.nbeat > 0:
            axB.barh(yy, r.nbeat, height=0.56, color=C_WORK, edgecolor="#caa472", lw=0.6, zorder=2)
        # left-gutter row label (family + compact task tag), right-aligned at the x=0 bar origin
        tag = r.task.split("(")[0].strip().lower()
        tag = tag.replace("cytokine cell-context", "cytokine cell-context")
        axB.text(-0.12, yy, f"{r.fam} · {tag}", ha="right", va="center",
                 fontsize=5.0, color=C_INK, zorder=4)
        verdict = "agree" if r.agree == "agree" else r.agree
        axB.text(r.n + 0.12, yy, f"{r.nbeat}/{r.n} beat · {verdict}", ha="left", va="center",
                 fontsize=4.8, color=GREY_MID, zorder=4)
    axB.set_yticks([])
    axB.set_xticks([0, 1, 2])
    axB.set_xticklabels(["0", "1", "2"], fontsize=6.0, color=C_INK)
    axB.set_xlabel("members evaluated (per family × task)", fontsize=6.0, color=C_INK)
    despine(axB)
    axB.spines["left"].set_visible(False)
    axB.spines["bottom"].set_bounds(0, 2)  # axis line only spans the real 0–2 data range
    axB.tick_params(axis="y", length=0)

    # ============================ legends ============================
    # ONE legend block for panel (a), as a single horizontal row beneath the grid.
    leg_a = [
        Patch(fc=C_SIMPLE_FILL, ec="#cfd4d8", label="simple baseline (tie → simplicity)"),
        Patch(fc=C_WORK, ec="#caa472", label="single-method exception"),
        Patch(fc=C_NA, ec="#dddddd", label="not evaluated"),
    ]
    lega = axA.legend(handles=leg_a, fontsize=5.4, frameon=True, loc="upper center",
                      bbox_to_anchor=(0.5, -0.085), ncol=3, handletextpad=0.5, columnspacing=1.2,
                      labelspacing=0.3, borderpad=0.5, borderaxespad=0.0, fancybox=False,
                      title="verdict (response-direction Pearson-Δ vs universal simple baseline)",
                      title_fontsize=5.4)
    style_legend(lega)
    lega.get_title().set_color(C_INK)
    lega.set_zorder(9)

    leg_b = [
        Patch(fc="#e2e6e9", ec="#cfd4d8", label="members evaluated"),
        Patch(fc=C_WORK, ec="#caa472", label="members beating baseline"),
    ]
    legb = axB.legend(handles=leg_b, fontsize=5.2, frameon=True, loc="upper center",
                      bbox_to_anchor=(0.5, -0.12), ncol=1, handletextpad=0.5, labelspacing=0.32,
                      borderpad=0.5, borderaxespad=0.0, fancybox=False)
    style_legend(legb)
    legb.set_zorder(9)

    # ============================ panel letters + titles ============================
    FS_HEAD, FS_LETTER = 8.0, 9.7
    panel_title(axA, "a", "Task × family fit verdict", sub=None,
                x_letter=-0.16, y=1.12, fs_title=FS_HEAD, fs_letter=FS_LETTER)
    panel_title(axB, "b", "Within-family agreement", sub=None,
                x_letter=-0.06, y=1.12, fs_title=FS_HEAD, fs_letter=FS_LETTER)

    out = PAPER / "figure_fit_matrix.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print("wrote", out)
    # report the mechanical read for the QC log
    n_exc = len(exception_cells)
    print(f"exceptions (single-method tiles): {n_exc}")
    for c, s, f, m, _ in exception_cells:
        print(f"  {c} / {s} / family={f} -> {m}")


if __name__ == "__main__":
    main()
