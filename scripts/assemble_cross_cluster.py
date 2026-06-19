#!/usr/bin/env python3
"""Assemble CROSS-CLUSTER HEADLINE + WITHIN-FAMILY CONSISTENCY tables.

Headline metric = response-direction (pearson_delta) DELTA vs the UNIVERSAL floor
{cell-mean, linear-PCA} (PREREG Sec 2). Biological unit per cluster (PREREG Sec 7):
  C1 lineage (LOCT) ; C2 donor (LODO) ; C3 dataset (LO-gene) ; C4 marker/modality (here
  modality folds, RNA) ; C5 lineage (LOCT) + global compound (unseen-cpd).

Floor reference = mean(cell-mean, linear-PCA) on pearson_delta, macro-averaged over the
cluster's biological unit on the task-defining split. Delta_vs_floor = family - floor_mean.
Also report delta vs each floor member. Real computed numbers only.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
OUT = os.path.join(ROOT, "results", "_paper")

FAMILY = {
    "cell-mean": "Simple", "linear-PCA": "Simple", "ctrl-pred": "Simple", "donor-shift": "Simple",
    "scGen": "Latent", "CPA": "Latent", "chemCPA": "Chemistry", "FP-ridge": "Chemistry",
    "scGPT": "Foundation", "scFoundation": "Foundation",
    "GEARS": "Graph", "AttentionPert": "Graph",
    "STATE": "Hybrid", "PertAdapt": "Hybrid",
    "CellOT": "OT", "scPRAM": "OT", "CINEMA-OT": "OT",
    "linear-shift-KOemb": "Deterministic shift",  # category matches Table 2 / S2a (deterministic shift, diagnostic comparator)
}

def macro(df, split_filter, unit_col, model_col, val_col="pearson_delta"):
    """Macro-average pearson_delta over biological unit, per model, on a split."""
    d = df[split_filter].copy()
    # per (model, unit) take the value (already per-unit for C2; per-dataset row for C3)
    g = d.groupby([model_col])[val_col].mean()
    return g

# ---------------- C1 (Kang IFN-beta), unit = lineage, split = LOCT ----------------
def c1_task():
    df = pd.read_csv(os.path.join(ROOT, "results", "C1", "results_raw.csv"))
    d = df[df["split"].str.startswith("C1_loct")]
    g = d.groupby("baseline")["pearson_delta"].mean()
    return ("C1", "cytokine/Kang", "cell-context (LOCT)", g, "lineage", 8)

# ---------------- C2 (Soskic CD4 activation), unit = donor, split = LODO ----------------
def c2_task():
    df = pd.read_csv(os.path.join(OUT, "results_raw_C2_rewrapped.csv"))
    d = df[(df["donor"] != "PENDING") & df["pearson_delta"].notna()]
    g = d.groupby("model")["pearson_delta"].mean()
    return ("C2", "donor/Soskic", "donor (LODO)", g, "donor", int(d["donor"].nunique()))

# ---------------- C3 (primary-T CRISPR), unit = dataset, split = true LO-gene (10%) ----------------
def c3_task():
    df = pd.read_csv(os.path.join(ROOT, "results", "C3", "results_raw.csv"))
    d = df[df["split"] == "C3_true_lo_gene_10"]
    g = d.groupby("baseline")["pearson_delta"].mean()  # macro over 5 datasets
    return ("C3", "gene/CRISPR", "unseen-perturbation (LO-gene 10%)", g, "dataset",
            int(d["dataset"].nunique()))

# ---------------- C4 (Frangieh), unit = modality fold (RNA), split = modality_lo_ko ----------------
def c4_task():
    base = pd.read_csv(os.path.join(ROOT, "results", "C4", "results_raw.csv"))
    base = base[base["modality"] == "RNA"]  # floor + scGen RNA
    fills = pd.read_csv(os.path.join(OUT, "results_raw_C4_fills_rewrapped.csv"))
    b = base.rename(columns={"baseline": "model"})[["model", "split", "pearson_delta"]]
    f = fills.rename(columns={"baseline": "model"})[["model", "split", "pearson_delta"]]
    allc4 = pd.concat([b, f], ignore_index=True)
    allc4 = allc4[allc4["split"].str.startswith("C4_modality_lo_ko")]
    g = allc4.groupby("model")["pearson_delta"].mean()  # macro over 2 folds
    return ("C4", "complex/Frangieh", "unseen-KO (modality, RNA)", g, "modality-fold", 2)

# ---------------- C5 (OP3 compounds): unseen-compound (global) + cell-context (LOCT) ----------------
def c5_unseen_task():
    df = pd.read_csv(os.path.join(ROOT, "results", "C5", "results_raw.csv"))
    d = df[df["split"] == "C5_global_compound_holdout"]
    g = d.groupby("baseline")["pearson_delta"].mean()
    # add chemCPA from native summary (unseen-compound). Prefer the deposited results/_paper copy so the
    # github_release deposit self-regenerates without the benchmark outputs/ tree; fall back to it.
    _cc = os.path.join(OUT, "chemcpa_op3_unseen_compound_summary.csv")
    if not os.path.exists(_cc):
        _cc = os.path.join(ROOT, "outputs", "additional_models", "chemcpa_op3_unseen_compound_summary.csv")
    cc = pd.read_csv(_cc)
    g = pd.concat([g, pd.Series({"chemCPA": float(cc["chemCPA_score"].iloc[0])})])
    return ("C5", "small-mol/OP3", "unseen-compound", g, "compound", 28)

def c5_loct_task():
    df = pd.read_csv(os.path.join(ROOT, "results", "C5", "results_raw.csv"))
    d = df[df["split"].str.startswith("C5_loct")]
    g = d.groupby("baseline")["pearson_delta"].mean()  # macro over 4 lineages
    return ("C5", "small-mol/OP3", "cell-context (LOCT)", g, "lineage", 4)

TASKS = [c1_task(), c2_task(), c3_task(), c4_task(), c5_unseen_task(), c5_loct_task()]

# ---------------- HEADLINE TABLE: family delta vs universal floor ----------------
rows = []
for cl, taskname, split, g, unit, nunit in TASKS:
    cm = g.get("cell-mean", np.nan)
    lp = g.get("linear-PCA", np.nan)
    floor_mean = np.nanmean([cm, lp])
    floor_max = np.nanmax([cm, lp])  # the harder of the two floor members
    for model, val in g.items():
        fam = FAMILY.get(model, "?")
        if fam == "Simple":
            continue
        rows.append({
            "cluster": cl, "task": taskname, "split": split, "unit": unit, "n_unit": nunit,
            "family": fam, "model": model,
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
head.to_csv(os.path.join(OUT, "cross_cluster_headline.csv"), index=False)

# ---------------- WITHIN-FAMILY CONSISTENCY: families with >=2 models on a task ----------------
con = []
for (cl, task, split), grp in head.groupby(["cluster", "task", "split"]):
    for fam, fg in grp.groupby("family"):
        if len(fg) < 2:
            continue
        beats = fg["beats_both_floor_members"].tolist()
        models = fg["model"].tolist()
        deltas = fg["delta_vs_floor_mean"].tolist()
        # agreement on beat-floor verdict
        agree = "agree" if (all(beats) or not any(beats)) else "split"
        con.append({
            "cluster": cl, "task": task, "split": split, "family": fam,
            "models": "+".join(models),
            "deltas_vs_floor_mean": deltas,
            "n_beat_both_floor": int(sum(beats)), "n_models": len(fg),
            "verdict_agreement": agree,
        })
cons = pd.DataFrame(con)

# Cross-model rho on the within-family axis (C3: across the 3 LO-gene folds, per dataset macro;
#   C2/C4/C5: across biological units). Compute where >=2 models share per-unit vectors.
def family_rho(cluster, models, splitkey):
    """Spearman rho between two models' per-unit pearson_delta vectors (real, if available)."""
    if cluster == "C3":
        df = pd.read_csv(os.path.join(ROOT, "results", "C3", "results_raw.csv"))
        d = df[df["split"] == "C3_true_lo_gene_10"]
        vecs = {m: d[d.baseline == m].set_index("dataset")["pearson_delta"] for m in models}
    elif cluster == "C1":
        df = pd.read_csv(os.path.join(ROOT, "results", "C1", "results_raw.csv"))
        d = df[df["split"].str.startswith("C1_loct")]
        vecs = {m: d[d.baseline == m].set_index("split")["pearson_delta"] for m in models}
    elif cluster == "C2":
        df = pd.read_csv(os.path.join(OUT, "results_raw_C2_rewrapped.csv"))
        d = df[df["donor"] != "PENDING"]
        vecs = {m: d[d.model == m].set_index("donor")["pearson_delta"] for m in models}
    elif cluster == "C5":
        df = pd.read_csv(os.path.join(ROOT, "results", "C5", "results_raw.csv"))
        if splitkey == "cell-context (LOCT)":
            d = df[df["split"].str.startswith("C5_loct")]
        else:
            d = df[df["split"] == "C5_global_compound_holdout"]
        vecs = {m: d[d.baseline == m].set_index("split")["pearson_delta"] for m in models}
    elif cluster == "C4":
        base = pd.read_csv(os.path.join(ROOT, "results", "C4", "results_raw.csv"))
        base = base[base.modality == "RNA"].rename(columns={"baseline": "model"})
        fills = pd.read_csv(os.path.join(OUT, "results_raw_C4_fills_rewrapped.csv")).rename(
            columns={"baseline": "model"})
        allc4 = pd.concat([base[["model", "split", "pearson_delta"]],
                           fills[["model", "split", "pearson_delta"]]], ignore_index=True)
        allc4 = allc4[allc4.split.str.startswith("C4_modality_lo_ko")]
        vecs = {m: allc4[allc4.model == m].set_index("split")["pearson_delta"] for m in models}
    else:
        return np.nan
    if len(models) != 2:
        return np.nan
    a, b = vecs[models[0]], vecs[models[1]]
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return np.nan
    return float(a.loc[common].corr(b.loc[common], method="spearman"))

cons["spearman_rho_pair"] = [
    family_rho(r.cluster, r.models.split("+"), r.split) if len(r.models.split("+")) == 2 else np.nan
    for r in cons.itertuples()]
# Flag: C4 has only 2 modality folds (<3 units) so cross-model rho is undefined; CellOT/scPRAM
# on C4 are 1-fold-per-split (only the two LO-KO fractions) -> flag for re-run per task note.
cons["flag"] = np.where(
    (cons["cluster"] == "C4"),
    "C4: 2 modality folds only (rho undefined, <3 units); CellOT/scPRAM single-seed per split -> re-run for CI",
    "")
cons.to_csv(os.path.join(OUT, "within_family_consistency.csv"), index=False)

print("=== HEADLINE (response-direction delta vs universal floor {cell-mean, linear-PCA}) ===")
print(head[["cluster","task","split","family","model","pearson_delta","floor_mean",
            "delta_vs_floor_mean","beats_both_floor_members"]].to_string(index=False))
print("\n=== WITHIN-FAMILY CONSISTENCY ===")
print(cons[["cluster","task","family","models","n_beat_both_floor","n_models",
            "verdict_agreement","spearman_rho_pair"]].to_string(index=False))
