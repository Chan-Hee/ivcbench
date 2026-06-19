#!/usr/bin/env python
"""Program-recovery vs program-dimensionality (tests the "only low-rank programs survive" law).

For EVERY scored immune program (C3 5-program T-cell panel on 5 CRISPR datasets; C5 3-program OP3 panel;
C1 type-I IFN on Kang) we relate:
  * recovery  (y): the deposited AUCell-delta program-recovery correlation (T2_per_program_AUCell_map.csv
                   best_corr; the consolidated number the paper reports). C3 programs are pooled across
                   the 5 datasets exactly as program_null.csv pools the conditioned rows.
  * dimensionality / low-rank-ness (x): computed HERE from the ACTUAL deposited single cells, restricted
                   to each program's gene set AS SCORED (intersected with the dataset HVG panel). Three
                   proxies, every value recomputed, none fabricated:
       FVE_PC1  = fraction of variance explained by PC1 of the program gene block (higher = more low-rank)
       mean_|r| = mean absolute off-diagonal gene-gene Pearson correlation (co-regulation)
       part_ratio = participation ratio (sum(lambda)^2 / sum(lambda^2)) = effective # dimensions
                    (LOWER = more low-rank); we also report eff_dim_frac = part_ratio / n_genes.
  For C3 a program spans 5 datasets -> we compute the proxy per dataset and average (cell-count weighted),
  matching how recovery is pooled.

The law predicts: recovery RISES with low-rank-ness (higher FVE_PC1 / higher mean_|r| / lower part_ratio).
type-I IFN is highlighted. We report the real Spearman + Pearson correlation and whether it supports the law.

Pure recompute: results CSVs for y; src/ivcbench loaders for the cells behind x. CPU only.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivcbench.clusters.c3 import C3_PROGRAMS
from ivcbench.clusters.c5 import C5_PROGRAMS

OUTDIR = ROOT / "results" / "_paper"
OUTDIR.mkdir(parents=True, exist_ok=True)

# C1 IFN gene set (exactly the set c1_ifn_recovery.py scored)
C1_IFN = ["ISG15","IFI6","MX1","MX2","OAS1","OAS2","IFIT1","IFIT3","ISG20",
          "STAT1","IRF7","IFI44","IFI44L","RSAD2","USP18"]

C3_DATASETS = {
    "shifrut":            ("ivcbench.data.loaders.shifrut", "load", {}),
    "schmidt":            ("ivcbench.data.loaders.schmidt", "load", {}),
    "mccutcheon_CRISPRi": ("ivcbench.data.loaders.mccutcheon", "load", {"modality": "CRISPRi"}),
    "mccutcheon_CRISPRa": ("ivcbench.data.loaders.mccutcheon", "load", {"modality": "CRISPRa"}),
    "chen":               ("ivcbench.data.loaders.chen", "load", {}),
}


def _load(mod, fn, kw):
    import importlib
    m = importlib.import_module(mod)
    return getattr(m, fn)(**kw)


def dimensionality(X_cells: np.ndarray, gene_idx: np.ndarray) -> dict:
    """Low-rank-ness proxies of a program gene block on a cell matrix.
    X_cells: (n_cells, n_genes_panel) log-norm expression. gene_idx: program genes present in panel."""
    gi = np.asarray(sorted(set(int(i) for i in gene_idx)), dtype=int)
    n = len(gi)
    if n < 3:
        return dict(n_genes=n, fve_pc1=np.nan, mean_abs_r=np.nan, part_ratio=np.nan, eff_dim_frac=np.nan)
    B = X_cells[:, gi].astype(np.float64)          # cells x program genes
    # drop zero-variance genes (degenerate; cannot be in a correlation/PCA basis)
    sd = B.std(0)
    keep = sd > 1e-9
    B = B[:, keep]
    n = B.shape[1]
    if n < 3:
        return dict(n_genes=n, fve_pc1=np.nan, mean_abs_r=np.nan, part_ratio=np.nan, eff_dim_frac=np.nan)
    Bc = B - B.mean(0)
    # covariance eigenspectrum (PCA on z-scored genes -> correlation-matrix spectrum)
    Z = Bc / B.std(0)
    C = np.corrcoef(Z, rowvar=False)               # n x n gene-gene correlation
    eig = np.linalg.eigvalsh(C)
    eig = np.clip(eig, 0, None)
    tot = eig.sum()
    fve_pc1 = float(eig.max() / tot) if tot > 0 else np.nan
    part_ratio = float((eig.sum() ** 2) / (eig ** 2).sum()) if (eig ** 2).sum() > 0 else np.nan
    # mean absolute off-diagonal correlation
    iu = np.triu_indices(n, k=1)
    mean_abs_r = float(np.abs(C[iu]).mean())
    return dict(n_genes=int(n), fve_pc1=fve_pc1, mean_abs_r=mean_abs_r,
                part_ratio=part_ratio, eff_dim_frac=float(part_ratio / n))


def weighted_avg(vals, wts):
    vals = np.asarray(vals, float); wts = np.asarray(wts, float)
    m = np.isfinite(vals)
    if not m.any():
        return np.nan
    return float(np.average(vals[m], weights=wts[m]))


def main():
    rows = []

    # ---------- C5 (OP3) : 3 programs, AUCell-delta recovery is well-defined here ----------
    print("loading OP3 (C5) ...", flush=True)
    cs = _load("ivcbench.data.loaders.op3", "load", {})
    # recovery from the deposited T2 table
    t2 = pd.read_csv(ROOT / "results/_paper/immune_novelty/T2_per_program_AUCell_map.csv")
    t2c5 = t2[t2.cluster == "C5"].set_index("program")
    # dimensionality on TREATED (non-control) cells — recovery is about the perturbation program shift
    treated = ~cs.obs["is_control"].astype(bool).to_numpy()
    Xt = cs.X[treated]
    for name, genes in C5_PROGRAMS.items():
        gi = cs.gene_index(genes)
        dim = dimensionality(Xt, gi)
        rec = float(t2c5.loc[name, "best_corr"]) if name in t2c5.index else np.nan
        rows.append(dict(cluster="C5", dataset="op3_GSE279945", program=name,
                         recovery=rec, n_genes_scored=len(gi), n_cells=int(Xt.shape[0]),
                         is_IFN=(name == "type_I_IFN"), **dim))
        print(f"  C5 {name:20s} rec={rec:+.3f} fve_pc1={dim['fve_pc1']:.3f} "
              f"mean|r|={dim['mean_abs_r']:.3f} partR={dim['part_ratio']:.2f} n={dim['n_genes']}", flush=True)
    del cs, Xt

    # ---------- C3 : 5 programs averaged over the 5 CRISPR datasets ----------
    t2c3 = t2[t2.cluster == "C3"].set_index("program")
    # per-program, per-dataset dimensionality
    c3dim = {p: {"fve_pc1": [], "mean_abs_r": [], "part_ratio": [], "eff_dim_frac": [],
                 "n_genes": [], "w": []} for p in C3_PROGRAMS}
    for ds, (mod, fn, kw) in C3_DATASETS.items():
        print(f"loading C3/{ds} ...", flush=True)
        cs = _load(mod, fn, kw)
        treated = ~cs.obs["is_control"].astype(bool).to_numpy()
        Xt = cs.X[treated] if treated.sum() > 50 else cs.X
        for name, genes in C3_PROGRAMS.items():
            gi = cs.gene_index(genes)
            dim = dimensionality(Xt, gi)
            for k in ("fve_pc1", "mean_abs_r", "part_ratio", "eff_dim_frac", "n_genes"):
                c3dim[name][k].append(dim[k])
            c3dim[name]["w"].append(int(Xt.shape[0]))
            print(f"  C3/{ds} {name:18s} fve_pc1={dim['fve_pc1']:.3f} mean|r|={dim['mean_abs_r']:.3f} "
                  f"partR={dim['part_ratio']:.2f} n={dim['n_genes']}", flush=True)
        del cs, Xt
    for name in C3_PROGRAMS:
        D = c3dim[name]; w = D["w"]
        rec = float(t2c3.loc[name, "best_corr"]) if name in t2c3.index else np.nan
        rows.append(dict(cluster="C3", dataset="5xCRISPR(mean)", program=name,
                         recovery=rec, n_genes_scored=float(np.nanmean(D["n_genes"])),
                         n_cells=int(np.sum(w)), is_IFN=False,
                         n_genes=float(np.nanmean(D["n_genes"])),
                         fve_pc1=weighted_avg(D["fve_pc1"], w),
                         mean_abs_r=weighted_avg(D["mean_abs_r"], w),
                         part_ratio=weighted_avg(D["part_ratio"], w),
                         eff_dim_frac=weighted_avg(D["eff_dim_frac"], w)))

    # ---------- C1 : type-I IFN on Kang (recovery NOT estimable by AUCell-delta-across-strata) ----------
    # We still compute the program DIMENSIONALITY (the x-axis), and record recovery as the C5 IFN anchor
    # is the estimable IFN number; here we leave C1 recovery NaN (honest) but emit the dimensionality so
    # the IFN low-rank-ness is corroborated on a second dataset.
    print("loading Kang (C1) ...", flush=True)
    cs = _load("ivcbench.data.loaders.kang", "load", {})
    treated = ~cs.obs["is_control"].astype(bool).to_numpy()
    Xt = cs.X[treated]
    gi = cs.gene_index(C1_IFN)
    dim = dimensionality(Xt, gi)
    rows.append(dict(cluster="C1", dataset="kang_GSE96583", program="type_I_IFN",
                     recovery=np.nan, n_genes_scored=len(gi), n_cells=int(Xt.shape[0]),
                     is_IFN=True, **dim))
    print(f"  C1 type_I_IFN(IFN-beta cells) fve_pc1={dim['fve_pc1']:.3f} mean|r|={dim['mean_abs_r']:.3f} "
          f"partR={dim['part_ratio']:.2f} n={dim['n_genes']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "program_recovery_vs_dimensionality.csv", index=False)
    print("\nWROTE", OUTDIR / "program_recovery_vs_dimensionality.csv")
    print(df[["cluster","program","recovery","n_genes","fve_pc1","mean_abs_r","part_ratio","eff_dim_frac"]]
          .round(4).to_string(index=False))

    # ---------- correlation: recovery vs each dimensionality proxy (rows with both defined) ----------
    fit = df[df.recovery.notna()].copy()
    print(f"\nROWS WITH RECOVERY (used in correlation): {len(fit)}  "
          f"(C5={ (fit.cluster=='C5').sum() }, C3={ (fit.cluster=='C3').sum() })")
    corr_out = {}
    for proxy, sign in [("fve_pc1", +1), ("mean_abs_r", +1), ("part_ratio", -1), ("eff_dim_frac", -1)]:
        sub = fit[["recovery", proxy]].dropna()
        if len(sub) >= 3 and sub[proxy].std() > 0:
            pr, pp = stats.pearsonr(sub.recovery, sub[proxy])
            sr, sp = stats.spearmanr(sub.recovery, sub[proxy])
            corr_out[proxy] = dict(n=len(sub), pearson_r=round(float(pr),4), pearson_p=round(float(pp),4),
                                   spearman_r=round(float(sr),4), spearman_p=round(float(sp),4),
                                   law_direction=("higher->recover" if sign>0 else "lower->recover"))
            print(f"  recovery vs {proxy:12s}: Pearson r={pr:+.3f} (p={pp:.3f})  "
                  f"Spearman rho={sr:+.3f} (p={sp:.3f})  n={len(sub)}  "
                  f"[law expects {'+' if sign>0 else '-'} slope]")
    (OUTDIR / "program_recovery_vs_dimensionality_corr.json").write_text(json.dumps(corr_out, indent=2))
    print("WROTE", OUTDIR / "program_recovery_vs_dimensionality_corr.json")
    return df, corr_out


if __name__ == "__main__":
    main()
