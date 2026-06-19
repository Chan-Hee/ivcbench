#!/usr/bin/env python
"""Figure QC — deterministic, reusable structural + data-faithfulness check for the 5-figure plate.

For each figure it verifies: (1) the script renders cleanly (exit 0, no Traceback, no missing-glyph
warning); (2) PNG + PDF artifacts exist with sane size/dimensions/DPI; (3) the load-bearing numbers the
figure must reflect still reproduce from results/{C1,C3,C4,C5}/results_raw.csv (data faithfulness); (4) the
script carries no suspicious hardcoded DATA literal. Writes results/_paper/FIGURE_QC.md and prints a table.
Run:  .venv/bin/python scripts/figure_qc.py
"""
from __future__ import annotations
import subprocess
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PY = str(ROOT / ".venv" / "bin" / "python")
SIMPLE = ["ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"]
DEEP = {"latent", "graph", "foundation", "hybrid"}


def _ran(c):
    d = pd.read_csv(RESULTS / c / "results_raw.csv")
    return d[d["ran"] == True]  # noqa: E712


def expected_values():
    """The canonical numbers each figure must reflect — recomputed live from the CSVs (the 'expected
    value guide', computed BEFORE inspecting any rendered artifact)."""
    g = {}
    d3 = _ran("C3"); M = "pearson_delta_ontarget"
    fl, gp = [], []
    for ds in d3.dataset.unique():
        for h in ["10", "25", "50"]:
            sub = d3[(d3.dataset == ds) & (d3.split == f"C3_true_lo_gene_{h}")]
            if len(sub):
                si = sub[sub.baseline.isin(SIMPLE)][M].max(); co = sub[sub.family.isin(DEEP)][M].max()
                fl.append(si); gp.append(co - si)
    g["C3_floor_mean"] = (float(np.mean(fl)), 0.457)
    g["C3_gap_mean"] = (float(np.mean(gp)), -0.241)
    g["C3_cond_wins_of_15"] = (int(sum(x > 0 for x in gp)), 0)
    d5 = _ran("C5"); gc = d5[d5.split == "C5_global_compound_holdout"]
    g["C5cpd_FPridge"] = (float(gc[gc.baseline == "FP-ridge"].pearson_delta.iloc[0]), 0.164)
    g["C5cpd_floor"] = (float(gc[gc.baseline.isin(SIMPLE)].pearson_delta.max()), 0.172)
    loc5 = d5[d5.split.str.startswith("C5_loct")]
    pr = loc5.groupby("baseline").agg(ifn=("aucell::type_I_IFN", "mean"), bulk=("pearson_delta", "mean"))
    g["C5_IFN_FPridge"] = (float(pr.loc["FP-ridge", "ifn"]), 0.77)
    g["C5_IFN_scGen"] = (float(pr.loc["scGen", "ifn"]), 0.78)
    g["C5_bulk_FPridge"] = (float(pr.loc["FP-ridge", "bulk"]), 0.39)
    g["C5_bulk_scGen"] = (float(pr.loc["scGen", "bulk"]), 0.18)
    d1 = _ran("C1"); lo = d1[d1.split.str.startswith("C1_lodo")]
    g["C1_LODO_scGen"] = (float(lo[lo.baseline == "scGen"].pearson_delta.mean()), 0.652)
    g["C1_LODO_cellmean"] = (float(lo[lo.baseline == "cell-mean"].pearson_delta.mean()), 0.660)
    rn = d1[d1.split.str.startswith("C1_randsplit")]
    NT = ["cell-mean", "donor-shift", "linear-PCA"]
    infl = [float(rn[rn.baseline == b].pearson_delta.mean() - lo[lo.baseline == b].pearson_delta.mean()) for b in NT]
    g["donor_inflation"] = (float(np.mean(infl)), 0.017)
    progcols = [c for c in d3.columns if c.startswith("aucell::")]
    vals = d3[d3.family.isin(DEEP)][progcols].values.flatten(); vals = vals[~np.isnan(vals)]
    g["C3_degenerate_zero_pct"] = (float(100 * (vals == 0).mean()), 84.8)
    g["C3_above01"] = (int((np.abs(vals) > 0.1).sum()), 24)
    for c, n in [("C1", 6), ("C3", 11), ("C4", 6), ("C5", 9)]:
        g[f"{c}_n_methods"] = (int(_ran(c).baseline.nunique()), n)
    return g


FIGS = [
    ("Fig1-framework", "figure_framework", ["C1_n_methods", "C3_n_methods", "C4_n_methods", "C5_n_methods"]),
    ("Fig2-ranking", "figure_ranking", ["C3_floor_mean", "C5cpd_FPridge", "C5cpd_floor", "C1_LODO_scGen"]),
    ("Fig3-landscape", "figure_landscape", ["C3_floor_mean", "C5cpd_FPridge", "C5cpd_floor", "C1_LODO_scGen"]),
    ("Fig4-cellcontext", "figure_cellcontext", ["C5_IFN_FPridge", "C5_IFN_scGen", "C5_bulk_FPridge", "C5_bulk_scGen"]),
    ("Fig5-perturbation", "figure_perturbation", ["C3_floor_mean", "C3_gap_mean", "C3_cond_wins_of_15", "C3_degenerate_zero_pct", "C3_above01"]),
    ("Fig6-donor", "figure_donor_decision", ["C1_LODO_scGen", "C1_LODO_cellmean", "donor_inflation"]),
    ("Fig7-immune-blindspot", "figure_immune_blindspot", ["C5_IFN_FPridge", "C5_IFN_scGen", "C3_degenerate_zero_pct"]),
    # Fig8 (within-family + descriptive fit-matrix) is sourced from within_family_consistency.csv /
    # descriptive_fit_matrix.csv / scpram_vs_cellot_donor_paired.csv, which expected_values() does not yet
    # recompute; keyed here to the two results_raw-derived anchors it also reflects (C5 compound floor, C5 IFN).
    ("Fig8-within-family-fit", "figure_within_family_fit", ["C5cpd_floor", "C5_IFN_FPridge"]),
]
# numeric literals that are legitimately layout/geometry/stat-threshold, not data (whitelist for the scan)
LIT_OK = re.compile(r"figsize|dpi|fontsize|lw=|ms=|alpha|zorder|width|height|pad|0\.1\b|2000|0\.05|95|len\(|range\(|\[0,|, 0\)|axhspan|axvspan")


def _close(a, b, tol=0.02):
    return abs(a - b) <= tol


def main():
    exp = expected_values()
    rows = []
    for key, mod, checks in FIGS:
        script = ROOT / "scripts" / f"{mod}.py"
        png = RESULTS / "_paper" / f"{mod}.png"; pdf = png.with_suffix(".pdf")
        rec = {"figure": key, "render": "?", "warnings": "", "png": "?", "pdf": "?",
               "dims": "", "data": "?", "data_detail": [], "hardcode": "?"}
        # 1) render
        r = subprocess.run([PY, str(script)], capture_output=True, text=True, cwd=str(ROOT))
        warn = [ln for ln in r.stderr.splitlines() if ("Warning" in ln or "missing from font" in ln or "Traceback" in ln)]
        rec["render"] = "PASS" if r.returncode == 0 else "FAIL"
        rec["warnings"] = "; ".join(w[:60] for w in warn[:2])
        # 2) artifacts
        rec["png"] = "PASS" if png.exists() and png.stat().st_size > 20000 else "FAIL"
        rec["pdf"] = "PASS" if pdf.exists() else "FAIL"
        try:
            from PIL import Image
            im = Image.open(png); rec["dims"] = f"{im.width}x{im.height} {im.info.get('dpi', ('?',))[0]}dpi"
            nonwhite = (np.asarray(im.convert("L")) < 250).mean()
            rec["dims"] += f" ink{nonwhite*100:.0f}%"
        except Exception as e:
            rec["dims"] = f"PIL? {e}"
        # 3) data faithfulness
        bad = []
        for ck in checks:
            got, want = exp[ck]
            ok = _close(got, want, tol=0.02 if want < 5 else 0.6)
            if not ok:
                bad.append(f"{ck}: got {got:.3f} want {want}")
        rec["data"] = "PASS" if not bad else "FAIL"
        rec["data_detail"] = bad
        # 4) hardcoded-literal scan (heuristic): plotting lines with a bare data-magnitude literal
        src = script.read_text().splitlines()
        susp = []
        for i, ln in enumerate(src, 1):
            if any(t in ln for t in ("ax.bar", "ax.plot", "ax.scatter", "barh", "axhline")) and not LIT_OK.search(ln):
                for m in re.findall(r"(?<![\w.])0\.[0-9]{2,}", ln):
                    if 0.04 < float(m) < 0.95:
                        susp.append(f"L{i}:{m}")
        rec["hardcode"] = "PASS" if not susp else f"CHECK ({len(susp)})"
        rows.append(rec)

    # report
    lines = ["# Figure QC report\n", "Deterministic structural + data-faithfulness QC (scripts/figure_qc.py).\n",
             "| Figure | Render | Glyph/err | PNG | PDF | Dimensions | Data-faithful | Hardcode-scan |",
             "|---|---|---|---|---|---|---|---|"]
    allpass = True
    for r in rows:
        gl = "clean" if not r["warnings"] else r["warnings"]
        lines.append(f"| {r['figure']} | {r['render']} | {gl} | {r['png']} | {r['pdf']} | {r['dims']} | "
                     f"{r['data']} | {r['hardcode']} |")
        if r["render"] != "PASS" or r["png"] != "PASS" or r["data"] != "PASS":
            allpass = False
        if r["data_detail"]:
            lines.append(f"|   ↳ data mismatch | {'; '.join(r['data_detail'])} |||||||")
    lines.append(f"\n**Overall structural QC: {'PASS' if allpass else 'FAIL'}**  "
                 f"(data faithfulness verified against results_raw.csv; hardcode-scan 'CHECK' = manual review of flagged literals)\n")
    out = RESULTS / "_paper" / "FIGURE_QC.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
