#!/usr/bin/env python
"""Unseen-cytokine extrapolation (pseudobulk-DE) — a direct supplementary test of the benchmark's
no-unseen-perturbation-extrapolation law, extended from genes/compounds to CYTOKINES.

Data: data/human_cytokine_dict/hcd_mini.csv — the Cytokine Dictionary summary table: per
(gene x celltype x cytokine) the pseudobulk differential-expression log_fc of the cytokine response
vs PBS control. The table is SPARSE (significant-DE rows only); the DE vector for a (celltype,
cytokine) is built on that celltype's gene UNIVERSE (union of all its DE genes), 0-filling genes
that were not significantly perturbed (= no detected response).

Task (analogous to C3 true-leave-one-gene-out, here leave-one-CYTOKINE-out within each celltype):
  For each held-out cytokine c (a test stratum) in celltype t, predict its DE vector d_hat(t,c) over
  the celltype gene universe, and score response-direction Pearson correlation between d_hat(t,c) and
  the observed held DE d_obs(t,c). The held cytokine's DE IN celltype t is never read to make its own
  prediction (leak-safe).

Predictors
  Floors (perturbation-agnostic; no cytokine identity used for the held context):
    * zero               — no-response baseline (predict 0 everywhere). Pearson is undefined for a
                           constant vector -> scored as 0 by convention (no response-direction signal).
    * cytokine-mean      — mean DE across TRAINING cytokines in celltype t (the universal floor; the
                           "average cytokine does this" prior, the direct analog of the C3 cell-mean).
  Conditioned attempts (use the held cytokine's IDENTITY, never its held-celltype effect):
    * DE-profile-nearest — pick the single nearest TRAINING cytokine by DE-profile similarity measured
                           in OTHER celltypes (the cytokine's identity is observable from contexts that
                           are not the held one), then transfer that neighbour's DE in celltype t.
                           Leak-safe: the held (t,c) effect is never used to choose or build the pred.
    * feature-nearest    — pick the nearest TRAINING cytokine by a cytokine FEATURE vector derived from
                           annotations (receptor-family / structural class parsed from the cytokine
                           name: IFN-I / IFN-II / IFN-III, IL-1, IL-2Rgamma-common, IL-6/gp130, IL-10,
                           IL-17, TNF-superfamily, growth-factor-RTK, etc.), then transfer that
                           neighbour's DE in celltype t. Family ties broken by name-string similarity.

All scored on the SAME response-direction Pearson, SAME LOCO splits, SAME 0-filled gene universe.
Outputs per-(celltype, held cytokine) scores + per-celltype and pooled summaries to results/newdata/.

ABSOLUTE: every number is computed from the real table; nothing is fabricated. Cytokines with a
degenerate (all-zero / <2 nonzero) held DE vector are reported separately as unscoreable, not dropped
silently.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/human_cytokine_dict/hcd_mini.csv"
OUT = ROOT / "results/newdata"
OUT.mkdir(parents=True, exist_ok=True)

MIN_CYTOKINES = 12   # a celltype must carry >= this many cytokines to support a LOCO test
MIN_NONZERO = 5      # a held DE vector needs >= this many nonzero genes to score a direction


# --------------------------------------------------------------------------------------------------
# cytokine feature annotation — receptor-family / structural class parsed from the cytokine name.
# Derived purely from the cytokine identity (public immunology nomenclature), never from the data.
# Each cytokine -> a small set of family tokens; feature-similarity = Jaccard on these token sets.
# --------------------------------------------------------------------------------------------------
def cytokine_features(name: str) -> set[str]:
    n = name.strip()
    feats: set[str] = set()
    low = n.lower()

    # --- interferons (by receptor) ---
    if low.startswith("ifn-alpha") or low.startswith("ifn-beta") or low in (
            "ifn-omega", "ifn-epsilon"):
        feats |= {"IFN_typeI", "IFNAR", "antiviral", "JAK_STAT"}
    if low.startswith("ifn-gamma"):
        feats |= {"IFN_typeII", "IFNGR", "JAK_STAT"}
    if low.startswith("ifn-lambda"):
        feats |= {"IFN_typeIII", "IFNLR", "antiviral", "JAK_STAT"}

    # --- common gamma-chain (IL2RG) family ---
    if n in ("IL-2", "IL-4", "IL-7", "IL-9", "IL-15", "IL-21"):
        feats |= {"gamma_c", "IL2RG", "JAK_STAT", "lymphoid"}
    # --- IL-1 superfamily ---
    if n in ("IL-1-alpha", "IL-1-beta", "IL-1Ra", "IL-33", "IL-36-alpha", "IL-36Ra",
             "IL-37", "TSLP", "IL-18"):
        feats |= {"IL1_superfamily", "MyD88", "inflammatory"}
    if n == "TSLP":
        feats |= {"TSLPR", "epithelial"}
    # --- IL-6 / gp130 family ---
    if n in ("IL-6", "IL-11", "IL-27", "IL-31", "LIF", "OSM", "CT-1", "CNTF"):
        feats |= {"gp130", "JAK_STAT", "IL6_family"}
    # --- IL-10 family ---
    if n in ("IL-10", "IL-19", "IL-20", "IL-22", "IL-24", "IL-26"):
        feats |= {"IL10_family", "JAK_STAT", "STAT3"}
    # --- IL-12 family (heterodimeric) ---
    if n in ("IL-12", "IL-23", "IL-27", "IL-35"):
        feats |= {"IL12_family", "heterodimer", "JAK_STAT", "Tcell_polarizing"}
    # --- IL-17 family ---
    if n in ("IL-17A", "IL-17B", "IL-17C", "IL-17D", "IL-17E", "IL-17F"):
        feats |= {"IL17_family", "IL17R", "Act1", "inflammatory"}
    # IL-17E is IL-25
    if n == "IL-17E":
        feats |= {"IL25", "type2"}
    # --- common beta-chain (CSF2RB) myeloid growth factors ---
    if n in ("IL-3", "IL-5", "GM-CSF"):
        feats |= {"beta_c", "CSF2RB", "myeloid_growth", "JAK_STAT"}
    if n in ("G-CSF", "M-CSF", "FLT3L", "SCF"):
        feats |= {"myeloid_growth"}
    if n in ("M-CSF", "SCF", "FLT3L"):
        feats |= {"RTK"}
    # --- IL-13 (shares IL-4Ralpha; type-2) ---
    if n in ("IL-4", "IL-13"):
        feats |= {"type2", "STAT6", "IL4Ra"}
    if n == "IL-13":
        feats |= {"type2", "STAT6"}

    # --- TNF superfamily (ligands) ---
    tnf_sf = {"TNF-alpha", "LT-alpha1-beta2", "LT-alpha2-beta1", "FasL", "TRAIL", "RANKL",
              "CD40L", "CD27L", "CD30L", "OX40L", "4-1BBL", "GITRL", "LIGHT", "TWEAK", "TL1A",
              "APRIL", "BAFF", "EDA"}
    if n in tnf_sf:
        feats |= {"TNF_superfamily", "TNFRSF", "NFkB"}
    if n in ("APRIL", "BAFF"):
        feats |= {"Bcell_survival"}
    if n in ("CD40L", "CD27L", "CD30L", "OX40L", "4-1BBL", "GITRL"):
        feats |= {"costimulatory"}

    # --- receptor-tyrosine-kinase growth factors / hormones ---
    if n in ("EGF", "FGF-beta", "HGF", "IGF-1", "VEGF", "GDNF", "PSPN", "PDGF", "NGF"):
        feats |= {"RTK", "growth_factor"}
    if n in ("EPO", "TPO", "G-CSF", "GM-CSF", "IL-3", "PRL", "Leptin", "GH"):
        feats |= {"typeI_cytokine_R", "JAK_STAT"}
    if n in ("EPO", "TPO"):
        feats |= {"hematopoietic"}
    if n in ("PRL", "Leptin", "GH"):
        feats |= {"hormone"}
    if n in ("C5a",):
        feats |= {"complement", "GPCR", "inflammatory"}
    if n in ("Noggin", "Decorin"):
        feats |= {"TGFbeta_modulator"}
    if n in ("ADSF",):
        feats |= {"adipokine"}

    # --- coarse functional tags by interferon / interleukin / TNF prefix as a backstop ---
    if low.startswith("ifn"):
        feats |= {"interferon"}
    if low.startswith("il-"):
        feats |= {"interleukin"}

    if not feats:
        feats = {"other_" + re.sub(r"[^a-z0-9]", "", low)}
    return feats


def feat_jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------------------------------
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; returns nan if either side is constant (undefined direction)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def build_celltype_matrix(sub: pd.DataFrame):
    """Return (cytokines list, gene universe list, DE matrix [n_cyto x n_gene], 0-filled)."""
    universe = sorted(sub.gene.unique())
    gpos = {g: i for i, g in enumerate(universe)}
    cytos = sorted(sub.cytokine.unique())
    cpos = {c: i for i, c in enumerate(cytos)}
    M = np.zeros((len(cytos), len(universe)), dtype=np.float64)
    gi = sub.gene.map(gpos).to_numpy()
    ci = sub.cytokine.map(cpos).to_numpy()
    M[ci, gi] = sub.log_fc.to_numpy()
    return cytos, universe, M


def main():
    print("[load]", CSV, flush=True)
    df = pd.read_csv(CSV, usecols=["gene", "log_fc", "celltype", "cytokine"])
    print(f"[data] {len(df):,} DE rows | {df.celltype.nunique()} celltypes | "
          f"{df.cytokine.nunique()} cytokines | {df.gene.nunique()} genes", flush=True)

    # ----- precompute each cytokine's cross-celltype DE profile on a SHARED global gene set, used
    #       ONLY to choose the DE-profile nearest neighbour from contexts other than the held one. -----
    # global gene universe (all genes seen anywhere) for a common cross-celltype profile space
    global_genes = sorted(df.gene.unique())
    ggpos = {g: i for i, g in enumerate(global_genes)}
    # per (celltype, cytokine) sparse DE on the global gene axis, stored as dict for fast assembly
    # profile of a cytokine in a celltype = its 0-filled DE vector over global genes (mean-collapsed
    # across celltypes other than the held one, when selecting a neighbour).
    # We store per-celltype DE as sparse (gene_idx, val) to keep memory bounded.
    key = list(zip(df.celltype, df.cytokine))
    df_gidx = df.gene.map(ggpos).to_numpy()
    by_pair: dict[tuple, tuple] = {}
    # group once
    grp = df.groupby(["celltype", "cytokine"], sort=False).indices
    for (ct, cy), rowpos in grp.items():
        by_pair[(ct, cy)] = (df_gidx[rowpos], df["log_fc"].to_numpy()[rowpos])

    celltypes = sorted(df.celltype.unique())
    cyto_celltypes: dict[str, list] = {}
    for cy in df.cytokine.unique():
        cyto_celltypes[cy] = sorted({ct for (ct, c2) in by_pair if c2 == cy})

    def cross_profile(cy: str, exclude_ct: str) -> np.ndarray:
        """Mean 0-filled DE vector of cytokine cy over global genes, averaged across celltypes that
        are NOT the held celltype (leak-safe identity signal)."""
        acc = np.zeros(len(global_genes), dtype=np.float64)
        n = 0
        for ct in cyto_celltypes[cy]:
            if ct == exclude_ct:
                continue
            gi, vv = by_pair[(ct, cy)]
            v = np.zeros(len(global_genes), dtype=np.float64)
            v[gi] = vv
            acc += v
            n += 1
        return acc / n if n else acc

    rows = []           # per (celltype, held cytokine, method) scored rows
    unscoreable = []    # held cytokines whose own DE vector is degenerate

    for ct in celltypes:
        sub = df[df.celltype == ct]
        if sub.cytokine.nunique() < MIN_CYTOKINES:
            continue
        cytos, universe, M = build_celltype_matrix(sub)
        gpos = {g: i for i, g in enumerate(universe)}
        n_cyto = len(cytos)

        # precompute the cross-celltype profile (global axis) for each cytokine, excluding ct
        cprof = {cy: cross_profile(cy, ct) for cy in cytos}
        # restrict feature set once
        feats = {cy: cytokine_features(cy) for cy in cytos}

        for hi, hc in enumerate(cytos):
            d_obs = M[hi]                                  # observed held DE on celltype universe
            nz = int(np.count_nonzero(d_obs))
            if nz < MIN_NONZERO or d_obs.std() < 1e-12:
                unscoreable.append(dict(celltype=ct, cytokine=hc, n_nonzero=nz))
                continue
            train_idx = [j for j in range(n_cyto) if j != hi]

            # ---- floors ----
            cyto_mean = M[train_idx].mean(0)               # universal floor
            zero_pred = np.zeros_like(d_obs)               # no-response baseline

            # ---- conditioned: DE-profile nearest (leak-safe, chosen on OTHER celltypes) ----
            hp = cprof[hc]
            best_de, best_r = None, -np.inf
            for j in train_idx:
                r = pearson(hp, cprof[cytos[j]])
                if np.isfinite(r) and r > best_r:
                    best_r, best_de = r, j
            de_near_pred = M[best_de] if best_de is not None else cyto_mean
            de_near_neighbor = cytos[best_de] if best_de is not None else "<fallback:cyto-mean>"

            # ---- conditioned: feature nearest (annotation-derived) ----
            hf = feats[hc]
            best_f, best_j = None, -1.0
            for j in train_idx:
                jc = feat_jaccard(hf, feats[cytos[j]])
                if jc > best_j:
                    best_j, best_f = jc, j
            feat_near_pred = M[best_f] if (best_f is not None and best_j > 0) else cyto_mean
            feat_near_neighbor = (cytos[best_f] if (best_f is not None and best_j > 0)
                                  else "<fallback:cyto-mean>")
            feat_fallback = not (best_f is not None and best_j > 0)

            for method, pred, extra in [
                ("zero",               zero_pred,      {}),
                ("cytokine-mean",      cyto_mean,      {}),
                ("DE-profile-nearest", de_near_pred,   {"neighbor": de_near_neighbor,
                                                         "neighbor_sim": float(best_r)}),
                ("feature-nearest",    feat_near_pred, {"neighbor": feat_near_neighbor,
                                                        "neighbor_sim": float(best_j),
                                                        "fallback": bool(feat_fallback)}),
            ]:
                r = pearson(pred, d_obs)
                if method == "zero":
                    # constant 0 -> undefined direction; by convention contributes no signal (0).
                    r = 0.0
                rows.append(dict(celltype=ct, cytokine=hc, method=method,
                                 pearson=r, n_nonzero=nz, n_train=len(train_idx),
                                 n_universe=len(universe), **extra))
        print(f"  [{ct}] scored {sub.cytokine.nunique()} cytokines "
              f"({len([u for u in unscoreable if u['celltype']==ct])} unscoreable)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cytokine_loco_per_held.csv", index=False)
    pd.DataFrame(unscoreable).to_csv(OUT / "cytokine_loco_unscoreable.csv", index=False)
    print(f"\n[write] {len(res)} scored rows -> {OUT/'cytokine_loco_per_held.csv'}")
    print(f"[write] {len(unscoreable)} unscoreable held cytokines -> "
          f"{OUT/'cytokine_loco_unscoreable.csv'}")

    # ===================== SUMMARIES =====================
    METHODS = ["zero", "cytokine-mean", "DE-profile-nearest", "feature-nearest"]

    # per-celltype mean Pearson per method
    per_ct = (res.pivot_table(index="celltype", columns="method", values="pearson", aggfunc="mean")
                 .reindex(columns=METHODS))
    per_ct["n_held"] = res[res.method == "cytokine-mean"].groupby("celltype").size()
    per_ct.to_csv(OUT / "cytokine_loco_per_celltype.csv")

    # pooled (mean over ALL held cytokines across celltypes)
    pooled = res.groupby("method")["pearson"].agg(["mean", "median", "std", "count"]).reindex(METHODS)

    # paired floor-vs-conditioned gap (per held cytokine): conditioned − cytokine-mean
    wide = res.pivot_table(index=["celltype", "cytokine"], columns="method", values="pearson")
    wide = wide.reindex(columns=METHODS)
    floor = wide["cytokine-mean"]
    gap_de = wide["DE-profile-nearest"] - floor
    gap_ft = wide["feature-nearest"] - floor
    n_held_total = len(wide)
    beat_de = int((wide["DE-profile-nearest"] > floor).sum())
    beat_ft = int((wide["feature-nearest"] > floor).sum())
    beat_either = int(((wide["DE-profile-nearest"] > floor) |
                       (wide["feature-nearest"] > floor)).sum())

    # bootstrap CI of the mean paired gap (resampling unit = held cytokine)
    def boot_mean_ci(vals, n=10000, seed=0):
        v = np.asarray([x for x in vals if np.isfinite(x)], float)
        if len(v) == 0:
            return np.nan, np.nan, np.nan
        rng = np.random.default_rng(seed)
        bs = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
        return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    de_gap_m, de_gap_lo, de_gap_hi = boot_mean_ci(gap_de.values)
    ft_gap_m, ft_gap_lo, ft_gap_hi = boot_mean_ci(gap_ft.values)

    summary = dict(
        n_celltypes_tested=int(res.celltype.nunique()),
        n_held_cytokine_instances=int(n_held_total),
        n_unscoreable=int(len(unscoreable)),
        pooled_mean_pearson={m: float(pooled.loc[m, "mean"]) for m in METHODS},
        pooled_median_pearson={m: float(pooled.loc[m, "median"]) for m in METHODS},
        floor_method="cytokine-mean",
        conditioned_vs_floor_gap=dict(
            DE_profile_nearest=dict(mean=de_gap_m, ci_lo=de_gap_lo, ci_hi=de_gap_hi,
                                    n_beat_floor=beat_de, frac_beat=beat_de / n_held_total),
            feature_nearest=dict(mean=ft_gap_m, ci_lo=ft_gap_lo, ci_hi=ft_gap_hi,
                                 n_beat_floor=beat_ft, frac_beat=beat_ft / n_held_total),
            either_conditioned_beats_floor=dict(n=beat_either,
                                                frac=beat_either / n_held_total),
        ),
        zero_baseline_mean_pearson=float(pooled.loc["zero", "mean"]),
        per_celltype_csv=str(OUT / "cytokine_loco_per_celltype.csv"),
        verdict=("FLOOR WINS: no conditioned predictor beats the cytokine-mean floor on average"
                 if (de_gap_m <= 0 and ft_gap_m <= 0)
                 else "a conditioned predictor beats the floor on average"),
    )
    with open(OUT / "cytokine_loco_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ POOLED (mean Pearson over all held cytokines) ================")
    for m in METHODS:
        print(f"  {m:20s}  mean={pooled.loc[m,'mean']:+.4f}  median={pooled.loc[m,'median']:+.4f}  "
              f"n={int(pooled.loc[m,'count'])}")
    print("\n  floor = cytokine-mean")
    print(f"  DE-profile-nearest − floor:  mean gap {de_gap_m:+.4f} "
          f"[{de_gap_lo:+.4f}, {de_gap_hi:+.4f}]  | beats floor {beat_de}/{n_held_total} "
          f"({100*beat_de/n_held_total:.0f}%)")
    print(f"  feature-nearest    − floor:  mean gap {ft_gap_m:+.4f} "
          f"[{ft_gap_lo:+.4f}, {ft_gap_hi:+.4f}]  | beats floor {beat_ft}/{n_held_total} "
          f"({100*beat_ft/n_held_total:.0f}%)")
    print(f"  EITHER conditioned beats floor: {beat_either}/{n_held_total} "
          f"({100*beat_either/n_held_total:.0f}%)")
    print(f"\n  VERDICT: {summary['verdict']}")
    print(f"[write] summary -> {OUT/'cytokine_loco_summary.json'}")


if __name__ == "__main__":
    main()
