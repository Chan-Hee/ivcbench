#!/usr/bin/env python
"""Req 14: C4 modality axis across multiple held-KO PARTITION seeds (CPU).

The deposited C4 uses one held-KO partition (seed=0). The simple floors are deterministic GIVEN a
partition, so their inferential weight comes from re-drawing the held-KO set. We re-run the 4 simple
floors + the conditioned linear-shift-KOemb family on held-KO partition seeds {0..4}, both modalities,
both holdout fractions, with the IDENTICAL leak-safe split+audit+downstream-only Pearson-Δ. We report the
protein-collapse gap (best-conditioned − best-simple) per seed and its across-seed CI, so the modality
verdict (the least-supported headline) becomes a multi-partition measurement rather than n=1. Leak-safe.
(scGen, the 2nd conditioned family, needs the scperturbench_eval env per seed and is escalated; the
linear-shift-KOemb family establishes the conditioned-vs-floor gap CPU-side here.)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from ivcbench.data.loaders.frangieh import load
from ivcbench.clusters import c4
from ivcbench.baselines.simple import SIMPLE_BASELINES
from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split
from ivcbench.metrics.response import pearson_delta
from ivcbench.metrics.distribution import e_distance
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/C4/partition_seeds.csv"
SEEDS = [0,1,2,3,4]

def gidx(cs, names):
    pos={g:i for i,g in enumerate(cs.var_names)}
    return np.array([pos[n] for n in names if n in pos], dtype=int)

def linear_koemb(cs, sp, spec):
    tr=sp.train_idx; obs=cs.obs.iloc[tr]
    is_ctrl=obs["is_control"].to_numpy(); pert=obs["perturbation"].to_numpy().astype(str)
    gpos={gg:i for i,gg in enumerate(cs.var_names)}; Xtr=cs.X[tr]; ctrl_X=Xtr[is_ctrl]
    k=int(min(50,ctrl_X.shape[0]-1,ctrl_X.shape[1]))
    gene_emb=PCA(n_components=max(2,k),random_state=0).fit(ctrl_X).components_.T
    ctrl_inf=cs.X[sp.inference_input_idx].mean(0) if len(sp.inference_input_idx) else ctrl_X.mean(0)
    ctrl_mean_tr=ctrl_X.mean(0)
    D,E=[],[]
    for gg in [g for g in np.unique(pert[~is_ctrl]) if g in gpos]:
        m=(pert==gg)&(~is_ctrl)
        if m.sum()==0: continue
        D.append(Xtr[m].mean(0)-ctrl_mean_tr); E.append(gene_emb[gpos[gg]])
    reg=Ridge(alpha=1.0).fit(np.vstack(E),np.vstack(D))
    test_perts=cs.obs.iloc[sp.test_idx]["perturbation"].to_numpy().astype(str)
    preds=[ctrl_inf+reg.predict(gene_emb[gpos[p]][None,:])[0] if p in gpos else ctrl_inf for p in test_perts]
    return np.vstack(preds), ctrl_inf

def main():
    rows=[]
    for modality,mtag in [("rna","RNA"),("protein","protein")]:
        cs=load(modality=modality); g=cs.uns["genes_perturbed"]
        for frac,lbl in [(0.25,"25"),(0.50,"50")]:
            for seed in SEEDS:
                held=c4.held_ko_fraction(g,frac,seed=seed)
                spec=c4.modality_lo_ko(held,lbl); sp=build_split(cs,spec); audit=audit_split(cs,sp)
                test_X=cs.X[sp.test_idx]; excl=gidx(cs,list(spec.held_values))
                # simple floors
                for B in SIMPLE_BASELINES:
                    b=B(); b.fit(cs,sp,side_info=cs.side_info); p=b.predict(cs,sp,side_info=cs.side_info)
                    r=pearson_delta(p.pred_cells,test_X,p.control_mean,sp.test_strata,excl)
                    rows.append(dict(modality=mtag,frac=lbl,seed=seed,model=b.name,family="simple",
                                     pearson_delta_ds=round(float(r["macro"]),4),leak_free=bool(audit["leak_free"])))
                # conditioned linear-shift-KOemb
                pred,ctrl_inf=linear_koemb(cs,sp,spec)
                r=pearson_delta(pred,test_X,ctrl_inf,sp.test_strata,excl)
                rows.append(dict(modality=mtag,frac=lbl,seed=seed,model="linear-shift-KOemb",family="latent",
                                 pearson_delta_ds=round(float(r["macro"]),4),leak_free=bool(audit["leak_free"])))
            print(f"{mtag} {lbl}% seeds {SEEDS} done",flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
    print(f"\nWROTE {OUT} ({len(df)} rows; leak_free={bool(df.leak_free.all())})")
    # gap = best-conditioned - best-simple, per (modality, frac, seed)
    print("\n=== conditioned(linear-shift-KOemb) − best-simple gap, per modality (across 5 partition seeds x 2 fracs = 10) ===")
    for mtag in ["RNA","protein"]:
        gaps=[]
        for frac in ["25","50"]:
            for seed in SEEDS:
                sub=df[(df.modality==mtag)&(df.frac==frac)&(df.seed==seed)]
                cond=sub[sub.model=="linear-shift-KOemb"].pearson_delta_ds.iloc[0]
                bs=sub[sub.family=="simple"].pearson_delta_ds.max()
                gaps.append(cond-bs)
        gaps=np.array(gaps)
        ci=stats.t.interval(0.95,len(gaps)-1,loc=gaps.mean(),scale=stats.sem(gaps))
        print(f"  {mtag:8s}: mean gap {gaps.mean():+.4f} (sd {gaps.std(ddof=1):.4f}), 95% t-CI [{ci[0]:+.4f},{ci[1]:+.4f}], "
              f"{int((gaps<0).sum())}/{len(gaps)} conditioned-LOSES, range [{gaps.min():+.3f},{gaps.max():+.3f}]")
    # explicit protein-vs-RNA collapse contrast: paired per (frac,seed)
    rna_g, pro_g = [], []
    for frac in ["25","50"]:
        for seed in SEEDS:
            for mtag,acc in [("RNA",rna_g),("protein",pro_g)]:
                sub=df[(df.modality==mtag)&(df.frac==frac)&(df.seed==seed)]
                acc.append(sub[sub.model=="linear-shift-KOemb"].pearson_delta_ds.iloc[0]-sub[sub.family=="simple"].pearson_delta_ds.max())
    rna_g, pro_g = np.array(rna_g), np.array(pro_g)
    diff = pro_g - rna_g  # how much MORE the protein modality collapses
    w=stats.wilcoxon(diff)
    print(f"\n  protein-collapse SHARPER than RNA: paired (protein_gap − RNA_gap) mean {diff.mean():+.4f} "
          f"over n={len(diff)} (frac×seed); Wilcoxon p={w.pvalue:.4g}; {int((diff<0).sum())}/{len(diff)} protein worse")

if __name__=="__main__":
    main()
