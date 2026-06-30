#!/usr/bin/env python3
"""Assemble the CROSS-CLUSTER HEADLINE + WITHIN-FAMILY CONSISTENCY tables.

SINGLE SOURCE OF TRUTH = the deposited prediction bundles. This script re-scores every
bundle under predictions/ (the same GPU-free path a reviewer runs via `make reproduce-eval`)
and aggregates those scores into the 35-cell headline. Nothing here reads a results_raw.csv
or any hand-maintained summary: the census is a pure function of the bundles, so it cannot
drift away from what a reviewer can reproduce. `scripts/check_consistency.py` (run by
`make test`) re-derives this file and fails the build if the committed copy disagrees.

Headline metric = response-direction (pearson_delta) DELTA vs the universal floor
{cell-mean, linear-PCA} (PREREG Sec 2). Biological unit per cluster (PREREG Sec 7):
  C1 lineage (LOCT) ; C2 donor (LODO) ; C3 dataset (LO-gene) ; C4 modality fold (RNA) ;
  C5 lineage (LOCT) + global compound (unseen-cpd). Floor reference = mean(cell-mean,
  linear-PCA) on pearson_delta, macro-averaged over the cluster's biological unit on the
  task-defining split. Delta_vs_floor = family - floor_mean; also report delta vs each member.

C2 carries two split-construction schemes among the deposited bundles (bespoke
`C2_soskic_LODO_*` for CellOT/CPA/STATE/scPRAM, framework `C2_lodo_*` for the floors / scGen);
the model->scheme assignment is disjoint, so unioning the C2 clusters recovers each model from
its own bundle. The donor index is normalised across the two schemes for the within-family rho.
"""
from __future__ import annotations
import glob
import os

import numpy as np
import pandas as pd

from ivcbench.eval.bundle import score_bundle

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
OUT = os.path.join(ROOT, "results", "_paper")

FAMILY = {
    "cell-mean": "Simple", "linear-PCA": "Simple", "ctrl-pred": "Simple", "donor-shift": "Simple",
    "scGen": "Latent", "CPA": "Latent", "chemCPA": "Chemistry", "FP-ridge": "Chemistry",
    "scGPT": "Foundation", "scFoundation": "Foundation",
    "GEARS": "Graph", "AttentionPert": "Graph",
    "STATE": "Hybrid", "PertAdapt": "Hybrid",
    "CellOT": "OT", "scPRAM": "OT", "CINEMA-OT": "OT",
    "linear-shift-KOemb": "Deterministic shift",  # category matches main Table 2 / Supplementary Table S2
}

# ---------------- re-score every deposited bundle (GPU-free), excluding the format-demo toys --------
def score_all():
    files = sorted(f for f in glob.glob(os.path.join(ROOT, "predictions", "**", "*.npz"), recursive=True)
                   if os.sep + "example" + os.sep not in f)
    df = pd.DataFrame(score_bundle(f) for f in files)
    df["dataset"] = df["dataset"].fillna("")
    return df

# ---------------- task-cell definitions (which bundles -> which census cell) ------------------------
# each cell: the bundle clusters it draws from, the split it is defined on, how to read the biological
# unit (for the macro-average and the within-family rho), and the PUBLISHED roster of conditioned
# models reported for that cell. The roster fixes the paper's 35-cell structure (which model x task is
# reported is a curatorial decision); the VALUES are sourced from the bundles. A model whose bundle
# exists but is not in the paper (e.g. PertAdapt on C2, whose donor-LODO bundle carries a non-canonical
# scheme name) is intentionally not reported here, matching the published census.
CELLS = [
    dict(cl="C1", task="cytokine/Kang", split="cell-context (LOCT)", unit="lineage", n_unit=8,
         clusters=["C1_LOCT"], match=lambda s: s.startswith("C1_loct"),
         unit_of=lambda r: r["split"].replace("C1_loct_", ""),
         roster=["CPA", "scGen"]),
    dict(cl="C2", task="donor/Soskic", split="donor (LODO)", unit="donor", n_unit=106,
         clusters=["C2", "C2_LODO"], match=lambda s: s.startswith(("C2_soskic_LODO", "C2_lodo")),
         unit_of=lambda r: r["split"].replace("C2_soskic_LODO_", "").replace("C2_lodo_", ""),
         roster=["CPA", "CellOT", "STATE", "scGen", "scPRAM"]),
    dict(cl="C3", task="gene/CRISPR", split="unseen-perturbation (LO-gene 10%)", unit="dataset", n_unit=5,
         clusters=["C3_LO_gene"], match=lambda s: s == "C3_true_lo_gene_10",
         unit_of=lambda r: r["dataset"],
         roster=["AttentionPert", "CINEMA-OT", "CPA", "GEARS", "PertAdapt", "STATE",
                 "scFoundation", "scGPT", "scGen"]),
    dict(cl="C4", task="complex/Frangieh", split="unseen-KO (modality, RNA)", unit="modality-fold", n_unit=2,
         clusters=["C4", "C4_Axis2"], match=lambda s: s.startswith("C4_modality_lo_ko"),
         unit_of=lambda r: r["split"],
         roster=["AttentionPert", "CPA", "CellOT", "GEARS", "STATE", "linear-shift-KOemb",
                 "scGen", "scPRAM"]),
    dict(cl="C5", task="small-mol/OP3", split="unseen-compound", unit="compound", n_unit=28,
         clusters=["C5", "C5_unseen_cpd"], match=lambda s: s == "C5_global_compound_holdout",
         unit_of=lambda r: r["split"],
         roster=["CINEMA-OT", "CPA", "FP-ridge", "STATE", "scGen", "chemCPA"]),
    dict(cl="C5", task="small-mol/OP3", split="cell-context (LOCT)", unit="lineage", n_unit=4,
         clusters=["C1_LOCT"], match=lambda s: s.startswith("C5_loct"),
         unit_of=lambda r: r["split"].replace("C5_loct_", ""),
         roster=["CINEMA-OT", "CPA", "FP-ridge", "STATE", "scGen"]),
]


def cell_long(df, cell):
    """Long table of per-(model, unit) pearson_delta for one census cell."""
    d = df[df["cluster"].isin(cell["clusters"]) & df["split"].map(cell["match"])].copy()
    d["unit"] = d.apply(cell["unit_of"], axis=1)
    # collapse any cluster/dataset duplication onto (model, unit) -> the model's own bundle value
    return d.groupby(["model", "unit"], as_index=False)["pearson_delta"].mean()


def build():
    """Re-score the bundles and assemble (headline, within-family) DataFrames in memory."""
    df = score_all()

    # ---------------- HEADLINE TABLE: family delta vs universal floor ----------------
    long_by_cell = {}
    rows = []
    for cell in CELLS:
        lng = cell_long(df, cell)
        long_by_cell[(cell["cl"], cell["split"])] = lng
        g = lng.groupby("model")["pearson_delta"].mean()  # macro over biological unit
        cm = g.get("cell-mean", np.nan)
        lp = g.get("linear-PCA", np.nan)
        floor_mean = np.nanmean([cm, lp])
        for model in cell["roster"]:  # published order; values from the bundles
            if model not in g.index:
                raise SystemExit(f"[assemble] {cell['cl']} {cell['split']}: no bundle for rostered model {model!r}")
            val = g[model]
            fam = FAMILY.get(model, "?")
            rows.append({
                "cluster": cell["cl"], "task": cell["task"], "split": cell["split"],
                "unit": cell["unit"], "n_unit": cell["n_unit"], "family": fam, "model": model,
                "pearson_delta": round(float(val), 4),
                "floor_cell_mean": round(float(cm), 4),
                "floor_linear_PCA": round(float(lp), 4),
                "floor_mean": round(float(floor_mean), 4),
                "delta_vs_floor_mean": round(float(val - floor_mean), 4),
                "delta_vs_cell_mean": round(float(val - cm), 4),
                "delta_vs_linear_PCA": round(float(val - lp), 4),
                "beats_both_floor_members": bool(val > cm and val > lp),
                "beats_floor_mean": bool(val > floor_mean),
            })
    head = pd.DataFrame(rows)

    # ---------------- WITHIN-FAMILY CONSISTENCY: families with >=2 models on a task ----------------
    con = []
    for (cl, task, split), grp in head.groupby(["cluster", "task", "split"]):
        for fam, fg in grp.groupby("family"):
            if len(fg) < 2:
                continue
            beats = fg["beats_both_floor_members"].tolist()
            con.append({
                "cluster": cl, "task": task, "split": split, "family": fam,
                "models": "+".join(fg["model"].tolist()),
                "deltas_vs_floor_mean": fg["delta_vs_floor_mean"].tolist(),
                "n_beat_both_floor": int(sum(beats)), "n_models": len(fg),
                "verdict_agreement": "agree" if (all(beats) or not any(beats)) else "split",
            })
    cons = pd.DataFrame(con)

    def family_rho(cl, split, models):
        """Spearman rho between two models' per-unit pearson_delta vectors (bundle-sourced)."""
        if len(models) != 2:
            return np.nan
        lng = long_by_cell.get((cl, split))
        if lng is None:
            return np.nan
        vecs = {m: lng[lng.model == m].set_index("unit")["pearson_delta"] for m in models}
        a, b = vecs[models[0]], vecs[models[1]]
        common = a.index.intersection(b.index)
        if len(common) < 3:
            return np.nan
        return float(a.loc[common].corr(b.loc[common], method="spearman"))

    if not cons.empty:
        cons["spearman_rho_pair"] = [
            family_rho(r.cluster, r.split, r.models.split("+")) for r in cons.itertuples()]
        cons["flag"] = np.where(
            cons["cluster"] == "C4",
            "C4: 2 modality folds only (rho undefined, <3 units); CellOT/scPRAM single-seed per split -> re-run for CI",
            "")
    return head, cons, len(df)


def main():
    head, cons, n = build()
    head.to_csv(os.path.join(OUT, "cross_cluster_headline.csv"), index=False)
    cons.to_csv(os.path.join(OUT, "within_family_consistency.csv"), index=False)
    print("=== HEADLINE (response-direction delta vs universal floor {cell-mean, linear-PCA}) ===")
    print(head[["cluster", "task", "split", "family", "model", "pearson_delta", "floor_mean",
                "delta_vs_floor_mean", "beats_both_floor_members"]].to_string(index=False))
    print(f"\n[bundle-sourced] {n} bundles re-scored -> {len(head)} headline cells")
    print("\n=== WITHIN-FAMILY CONSISTENCY ===")
    if not cons.empty:
        print(cons[["cluster", "task", "family", "models", "n_beat_both_floor", "n_models",
                    "verdict_agreement", "spearman_rho_pair"]].to_string(index=False))


if __name__ == "__main__":
    main()
