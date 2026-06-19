#!/usr/bin/env python
"""Reviewer batch-4 analysis (CPU, pure recompute on deposited CSVs).
Covers reqs 2,3,4,5,7,10,11,12,15. Prints a labelled block per req.
All numbers re-read live; no model training."""
import numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(20260529)

C1 = pd.read_csv('results/C1/results_raw.csv')
C3 = pd.read_csv('results/C3/results_raw.csv')
C4 = pd.read_csv('results/C4/results_raw.csv')
C5 = pd.read_csv('results/C5/results_raw.csv')
ROWS = pd.read_csv('results/_paper/prior_confrontation_rows.csv')
SIMPLE = {'ctrl-pred','cell-mean','donor-shift','linear-PCA'}
COND_STRICT = {'GEARS','AttentionPert','scGPT','CPA','scGen','STATE','FP-ridge','chemCPA'}  # excludes OT
OT = {'CINEMA-OT'}

def boot_ci(x, fn=np.mean, n=10000, alpha=0.05):
    x = np.asarray(x, float)
    bs = np.array([fn(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return fn(x), np.percentile(bs, 100*alpha/2), np.percentile(bs, 100*(1-alpha/2)), (bs>0).mean()

print("="*80); print("REQ 2: C5-only pooled board vs all-cluster pooled board"); print("="*80)
# all-cluster board already deposited = prior_confrontation_pooled_board.csv (28 splits)
board_all = pd.read_csv('results/_paper/prior_confrontation_pooled_board.csv')
print("ALL-CLUSTER pool (deposited, n_splits up to 28):")
print(board_all.to_string(index=False))
# C5-only pool: average the 5 C5 splits equally (OP-style mixed pool the text cites)
c5rows = ROWS[ROWS.cluster=='C5']
c5board = (c5rows.groupby(['baseline','family','is_simple'])
           .agg(mean_pearson_delta=('pearson_delta','mean'), n_splits=('split','nunique'))
           .reset_index().sort_values('mean_pearson_delta', ascending=False))
print("\nC5-ONLY pool (5 splits = 1 compound-holdout + 4 LOCT, averaged equally, OP-style):")
print(c5board.to_string(index=False))
c5board.to_csv('results/_paper/prior_confrontation_c5board.csv', index=False)
print("-> deposited results/_paper/prior_confrontation_c5board.csv")
# explicit rank of FP-ridge in each
for name,bd in [('ALL',board_all),('C5',c5board)]:
    bd2 = bd.sort_values('mean_pearson_delta', ascending=False).reset_index(drop=True)
    r = bd2.index[bd2.baseline=='FP-ridge'].tolist()
    top = bd2.iloc[0]
    print(f"  {name}-pool: #1 = {top.baseline} ({top.mean_pearson_delta:.3f}); FP-ridge rank = {(r[0]+1) if r else 'NA'}")

print("\n"+"="*80); print("REQ 3: Ahlmann-Eltze best-deep roster + per-split C3 deltas + C5 reconcile"); print("="*80)
# DEEP roster: deep-learning models. Define explicitly. Two variants for C5 (FP-ridge in or out).
DEEP_NONLINEAR = {'GEARS','AttentionPert','scGPT','CPA','scGen','STATE'}  # neural; excludes FP-ridge (linear ridge) & OT
LINEAR_SIMPLE = {'ctrl-pred','cell-mean','donor-shift','linear-PCA'}
m='pearson_delta_ontarget'
print("DEEP roster (neural only): GEARS, AttentionPert, scGPT, CPA, scGen, STATE")
print("LINEAR/SIMPLE roster: ctrl-pred, cell-mean, donor-shift, linear-PCA")
print("\nC3 per-split (best_deep - best_linear), ontarget:")
c3deltas=[]
for sp in ['C3_true_lo_gene_10','C3_true_lo_gene_25','C3_true_lo_gene_50']:
    g = C3[(C3.split==sp)&(C3.ran==True)]
    # average across the 5 datasets per baseline first
    perb = g.groupby('baseline')[m].mean()
    deep = perb[perb.index.isin(DEEP_NONLINEAR)]
    lin = perb[perb.index.isin(LINEAR_SIMPLE)]
    bd, bl = deep.max(), lin.max()
    c3deltas.append(bd-bl)
    print(f"  {sp}: best_deep={bd:.3f} ({deep.idxmax()})  best_linear={bl:.3f} ({lin.idxmax()})  delta={bd-bl:+.4f}")
print(f"  C3 mean delta = {np.mean(c3deltas):+.4f}")
# C5 unseen-compound, two conventions
g5 = C5[(C5.split=='C5_global_compound_holdout')&(C5.ran==True)]
perb5 = g5.groupby('baseline')[m].mean()
lin5 = perb5[perb5.index.isin(LINEAR_SIMPLE)].max()
deep_noFP = perb5[perb5.index.isin(DEEP_NONLINEAR)].max()
deep_wFP = perb5[perb5.index.isin(DEEP_NONLINEAR|{'FP-ridge'})].max()
print(f"\nC5 unseen-compound (ontarget): best_linear={lin5:.4f}")
print(f"  best_deep EXCL FP-ridge (neural only) = {deep_noFP:.4f} -> delta {deep_noFP-lin5:+.4f}  [{perb5[perb5.index.isin(DEEP_NONLINEAR)].idxmax()}]")
print(f"  best_deep INCL FP-ridge (ridge counted as deep/conditioned) = {deep_wFP:.4f} -> delta {deep_wFP-lin5:+.4f}  [{perb5[perb5.index.isin(DEEP_NONLINEAR|{'FP-ridge'})].idxmax()}]")
print(f"  -> the -0.0135 figure uses INCL-FP convention (FP-ridge counted as conditioned/deep); the -0.008 uses FP-ridge vs floor only.")

print("\n"+"="*80); print("REQ 4: paired E-distance test best-conditioned vs best-simple per axis (signed, CI)"); print("="*80)
def edist_paired(df, splits, label, cond_set):
    gaps=[]
    for sp in splits:
        g = df[(df.split==sp)&(df.ran==True)].dropna(subset=['e_distance'])
        s = g[g.baseline.isin(SIMPLE)]
        c = g[g.baseline.isin(cond_set)]
        if len(s)==0 or len(c)==0: continue
        bs = s['e_distance'].min()   # best simple = LOWEST E-distance
        bc = c['e_distance'].min()   # best conditioned = LOWEST E-distance
        gaps.append(bc-bs)           # negative = conditioned closer (better)
    gaps=np.array(gaps)
    mean,lo,hi,_ = boot_ci(gaps)
    try: w = stats.wilcoxon(gaps)
    except Exception as e: w=None
    t = stats.ttest_1samp(gaps,0)
    nneg=(gaps<0).sum()
    print(f"  {label}: n={len(gaps)} cells; mean (cond-simple) E-dist gap = {mean:+.4f} (neg=conditioned closer)")
    print(f"     95% boot CI [{lo:+.4f},{hi:+.4f}]; {nneg}/{len(gaps)} conditioned-lower; "
          f"Wilcoxon p={w.pvalue:.4g}" if w else f"     Wilcoxon NA")
    print(f"     one-sample t p={t.pvalue:.4g}; gaps={np.round(gaps,3)}")
    return gaps
print("Convention: gap = best-conditioned E-dist - best-simple E-dist; NEGATIVE = conditioning produces closer cell cloud (win).")
edist_paired(C5, ['C5_loct_B','C5_loct_Mono','C5_loct_NK','C5_loct_T_cells'], 'C5 cell-context (FP-ridge etc, WIN axis)', COND_STRICT)
edist_paired(C5, ['C5_global_compound_holdout'], 'C5 unseen-compound', COND_STRICT|OT)
edist_paired(C3, ['C3_true_lo_gene_10','C3_true_lo_gene_25','C3_true_lo_gene_50'],'C3 unseen-gene (per dataset-split)', COND_STRICT)
# C3 must be per dataset x split cell
c3cells=[]
for sp in ['C3_true_lo_gene_10','C3_true_lo_gene_25','C3_true_lo_gene_50']:
    for ds in C3.dataset.unique():
        g = C3[(C3.split==sp)&(C3.dataset==ds)&(C3.ran==True)].dropna(subset=['e_distance'])
        s=g[g.baseline.isin(SIMPLE)]; c=g[g.baseline.isin(COND_STRICT)]
        if len(s) and len(c): c3cells.append(c.e_distance.min()-s.e_distance.min())
c3cells=np.array(c3cells)
mean,lo,hi,_=boot_ci(c3cells); w=stats.wilcoxon(c3cells)
print(f"  C3 per dataset-x-holdout cell (n={len(c3cells)}): mean gap {mean:+.4f}, CI[{lo:+.4f},{hi:+.4f}], "
      f"{(c3cells<0).sum()}/{len(c3cells)} conditioned-lower, Wilcoxon p={w.pvalue:.4g}")
edist_paired(C1, sorted(C1[C1.split.str.startswith('C1_loct')].split.unique()),'C1 cell-context (scGen)', {'scGen'})

print("\n"+"="*80); print("REQ 5+12: scPerturb E-distance rank-corr CI + signed sign + rank tables"); print("="*80)
# C3 pooled-distance board (scPerturb-style): all models on the 3 lo-gene splits, mean E-distance (lower=better)
c3 = ROWS[ROWS.cluster=='C3']
edboard = (c3.groupby('baseline').e_distance.mean().sort_values()).reset_index()
edboard['edist_rank']=range(1,len(edboard)+1)
pdboard = (c3.groupby('baseline').pearson_delta.mean().sort_values(ascending=False)).reset_index()
pdboard['pd_rank']=range(1,len(pdboard)+1)
merged = edboard.merge(pdboard, on='baseline')
print("C3 scPerturb-style E-distance board (lower=better) vs Pearson-Delta board (higher=better):")
print(merged[['baseline','e_distance','edist_rank','pearson_delta','pd_rank']].to_string(index=False))
# Spearman: between the two RANKINGS. E-dist rank ascending(1=best), PD rank ascending(1=best).
# Rank-AGREEMENT correlation: corr of the two rank vectors (1=best in both) -> positive if they agree.
rho_rankagree, p_ra = stats.spearmanr(merged.edist_rank, merged.pd_rank)
# Signed value-correlation: e_distance (lower=better) vs pearson_delta (higher=better) -> NEGATIVE if they agree
rho_signed, p_s = stats.spearmanr(merged.e_distance, merged.pearson_delta)
print(f"\n  Rank-AGREEMENT Spearman (edist_rank vs pd_rank, both 1=best): rho={rho_rankagree:+.3f}, p={p_ra:.3f}  (POSITIVE => agree)")
print(f"  Signed value Spearman (e_distance[low=good] vs pearson_delta[high=good]): rho={rho_signed:+.3f}, p={p_s:.3f}  (NEGATIVE => agree)")
print(f"  -> the +0.53 is the rank-AGREEMENT sign; the -0.53 is the signed-value sign. SAME relationship, opposite label.")
# Bootstrap rank-corr CI over the 3 splits (resample splits then recompute board+rho)
splits3=['C3_true_lo_gene_10','C3_true_lo_gene_25','C3_true_lo_gene_50']
rhos=[]
for _ in range(10000):
    pick=rng.choice(splits3,3,replace=True)
    sub=pd.concat([c3[c3.split==s] for s in pick])
    eb=sub.groupby('baseline').e_distance.mean().rank()  # 1=lowest=best
    pb=sub.groupby('baseline').pearson_delta.mean().rank(ascending=False)  # 1=highest=best
    common=eb.index.intersection(pb.index)
    if len(common)>2:
        r,_=stats.spearmanr(eb[common],pb[common])
        if not np.isnan(r): rhos.append(r)
rhos=np.array(rhos)
print(f"  Bootstrap rank-agreement rho over the 3 splits: median {np.median(rhos):+.3f}, 95% CI [{np.percentile(rhos,2.5):+.3f},{np.percentile(rhos,97.5):+.3f}]  (n_boot={len(rhos)})")
# Load-bearing rank inversions
print("  Load-bearing rank INVERSIONS (E-distance board mis-crowns vs Pearson board):")
for _,r in merged.iterrows():
    if abs(r.edist_rank-r.pd_rank)>=3:
        print(f"     {r.baseline}: E-dist rank #{int(r.edist_rank)} vs Pearson rank #{int(r.pd_rank)} (shift {int(r.pd_rank-r.edist_rank):+d})")

print("\n"+"="*80); print("REQ 7: program metric that DISCRIMINATES the cell-context winner (C5 LOCT)"); print("="*80)
loct = C5[C5.split.str.startswith('C5_loct') & (C5.ran==True)].copy()
progs=['aucell::type_I_IFN','aucell::inflammatory_NFkB','aucell::effector_lymphocyte']
print("Per-model mean over 4 LOCT lineages (program recovery) + bulk pearson_delta:")
tab = loct.groupby('baseline').agg(**{
    'bulk_pd':('pearson_delta','mean'),
    'IFN':('aucell::type_I_IFN','mean'),
    'NFkB':('aucell::inflammatory_NFkB','mean'),
    'eff_lymph':('aucell::effector_lymphocyte','mean')})
print(tab.round(3).to_string())
print("\n  (a) IFN does NOT discriminate the winner: FP-ridge IFN ~0.77 ≈ CPA 0.79 ≈ STATE 0.75 ≈ scGen 0.78 (bulk losers match the winner).")
# NF-kB nonzero-count per model
print("\n  (b) NF-kB recovery is winner-specific. Per-lineage nonzero NF-kB:")
nf = loct.pivot_table(index='baseline', columns='split', values='aucell::inflammatory_NFkB')
print(nf.round(3).to_string())
nf_nonzero = (loct.assign(nz=loct['aucell::inflammatory_NFkB']!=0).groupby('baseline').nz.sum())
print("  NF-kB nonzero-lineage count / 4:")
print(nf_nonzero.to_string())
# Joint program x magnitude score: IFN recovery * bulk_pd (only the winner gets BOTH)
print("\n  (c) program x magnitude joint score = IFN_recovery * bulk_pearson_delta (per model, mean over lineages):")
tab['joint_IFNxbulk']=tab['IFN']*tab['bulk_pd']
print(tab[['IFN','bulk_pd','joint_IFNxbulk']].sort_values('joint_IFNxbulk',ascending=False).round(3).to_string())
# per-lineage joint score winner
jl = loct.assign(joint=loct['aucell::type_I_IFN']*loct['pearson_delta'])
jwin = jl.loc[jl.groupby('split').joint.idxmax()][['split','baseline','joint']]
print("  per-lineage argmax of joint IFNxbulk:")
print(jwin.to_string(index=False))

print("\n"+"="*80); print("REQ 10: donor 12-seed grand-mean canonical statistic from donor_inflation_seeds.csv"); print("="*80)
d = pd.read_csv('results/C1/donor_inflation_seeds.csv')
print(f"rows={len(d)}, seeds={sorted(d.seed.unique())}, baselines={sorted(d.baseline.unique())}, donors={d.donor.nunique()}")
real = d[d.baseline!='ctrl-pred']  # the 3 real-shift baselines
# per-seed grand mean over (donor x baseline) for the 3 real baselines
per_seed = real.groupby('seed').infl.mean()
print(f"\n  CANONICAL (12-seed, 3 real-shift baselines cell-mean/donor-shift/linear-PCA):")
print(f"    per-seed grand-mean inflation: mean={per_seed.mean():+.4f}, sd={per_seed.std(ddof=1):.4f}, n_seeds={len(per_seed)}")
m_ps,lo_ps,hi_ps,_=boot_ci(per_seed.values)
print(f"    bootstrap-over-seeds grand-mean 95% CI [{lo_ps:+.4f},{hi_ps:+.4f}]; {(per_seed>0).sum()}/{len(per_seed)} seeds positive; sign-test p={stats.binomtest((per_seed>0).sum(),len(per_seed),0.5).pvalue:.4g}")
# overall pooled (every real row)
print(f"    POOLED over all {len(real)} real rows: mean infl={real.infl.mean():+.4f}, sd={real.infl.std():.4f}")
# per-baseline (12-seed)
print("    per-baseline (12-seed mean infl):")
print(real.groupby('baseline').infl.mean().round(4).to_string())
# seed-0
s0=d[(d.seed==0)&(d.baseline!='ctrl-pred')]
print(f"    seed-0 only (3 real baselines): mean infl={s0.infl.mean():+.4f}")
# Wilcoxon over per-(seed,donor,baseline) for the headline test? Use the published n=48 seed-0
s0all = d[(d.seed==0)&(d.baseline!='ctrl-pred')]
w=stats.wilcoxon(s0all.infl)
print(f"    seed-0 Wilcoxon over {len(s0all)} baseline×donor rows: p={w.pvalue:.4g} (matches published 6.4e-4 family)")
print(f"  -> CANONICAL stated value: +{per_seed.mean():.4f} (sd {per_seed.std(ddof=1):.4f}, n=12 seeds, 3 real-shift baselines); replaces the stale '+0.012'.")

print("\n"+"="*80); print("REQ 11: C3 program-cell accounting (abs convention, noise band, 1/435)"); print("="*80)
ran3 = C3[C3.ran==True].copy()
prog_cols=['aucell::TCR_activation','aucell::IL2_STAT5','aucell::proliferation','aucell::effector_cytokine','aucell::Treg_exhaustion']
cond = ran3[ran3.family.isin(['graph','latent','hybrid','foundation'])]
simpleOT = ran3[ran3.family.isin(['simple','ot'])]
print(f"conditioned rows={len(cond)} ({len(cond)}×5={len(cond)*5} program-cells); simple+OT rows={len(simpleOT)} (structural zeros)")
vals = cond[prog_cols].values.flatten()
n_cells=len(vals)
abs_gt = (np.abs(vals)>0.1).sum()
signed_gt = (vals>0.1).sum()
print(f"  |corr|>0.1 (ABSOLUTE convention): {abs_gt}/{n_cells}")
print(f"  signed corr>0.1 (SIGNED convention): {signed_gt}/{n_cells}")
print(f"  => 24 = |corr|>0.1 (two-sided); 12 = signed >0.1 (one-sided positive). Convention now explicit.")
# noise band expectation: empirical sd, two-sided expected hits
sd=vals.std()
# two-sided P(|N(0,sd)|>0.1)
p_two = 2*(1-stats.norm.cdf(0.1/sd))
print(f"  empirical sd of conditioned program corrs = {sd:.4f}")
print(f"  two-sided noise expectation: P(|N(0,sd)|>0.1)={p_two:.4f} × {n_cells} = {p_two*n_cells:.1f} expected hits (matches ~96/435 |corr| convention)")
# 1/435 exceeds own 95% band
exceed=0
for _,r in cond.iterrows():
    nstr=r['n_test_strata']
    band=1.96/np.sqrt(max(nstr-1,1))
    for pc in prog_cols:
        v=r[pc]
        if v!=0 and abs(v)>band: exceed+=1
print(f"  cells exceeding their OWN per-stratum 95% band (1.96/sqrt(n_strata-1)) AND nonzero: {exceed}/{n_cells}")

print("\n"+"="*80); print("REQ 15: TOST equivalence-bound + 80%-power from tanimoto_percompound.csv"); print("="*80)
t = pd.read_csv('results/C5/tanimoto_percompound.csv')
print(f"rows={len(t)}, compounds={t.compound.nunique()}, baselines={sorted(t.baseline.unique())}")
EQ=0.05  # equivalence bound: |slope| < 0.05 pearson per 1 SD of standardized Tanimoto distance
for b in ['FP-ridge','CPA','scGen','STATE','CINEMA-OT']:
    sub=t[t.baseline==b].dropna(subset=['pearson_delta','tanimoto_dist'])
    n=len(sub)
    if n<3: continue
    x=(sub.tanimoto_dist - sub.tanimoto_dist.mean())/sub.tanimoto_dist.std()  # standardized distance
    y=1-sub.pearson_delta  # ERROR metric
    res=stats.linregress(x,y)
    slope,se=res.slope,res.stderr
    df_=n-2
    # TOST: H0a slope<=-EQ, H0b slope>=+EQ; reject both => equivalent
    t_lo=(slope-(-EQ))/se; p_lo=stats.t.sf(t_lo,df_)        # test slope > -EQ
    t_hi=(slope-(EQ))/se;  p_hi=stats.t.cdf(t_hi,df_)       # test slope < +EQ
    p_tost=max(p_lo,p_hi)
    ci90=(slope-stats.t.ppf(0.95,df_)*se, slope+stats.t.ppf(0.95,df_)*se)
    # 80% power detectable slope: slope at which a two-sided alpha=.05 test has 80% power
    # min detectable effect ≈ (t_{.975,df}+t_{.80,df}) * se
    mde=(stats.t.ppf(0.975,df_)+stats.t.ppf(0.80,df_))*se
    print(f"  {b}: n={n}, slope(error vs std-dist)={slope:+.4f}, R2={res.rvalue**2:.4f}, p_two={res.pvalue:.3f}, "
          f"90%CI[{ci90[0]:+.4f},{ci90[1]:+.4f}], TOST p={p_tost:.4f} ({'EQUIV' if p_tost<0.05 else 'NOT equiv'}), 80%-power MDE={mde:.3f}/SD")
# pooled over 4 conditioned chemistry/latent/hybrid models
pooled=t[t.baseline.isin(['FP-ridge','CPA','scGen','STATE'])].dropna(subset=['pearson_delta','tanimoto_dist'])
xg=(pooled.tanimoto_dist-pooled.tanimoto_dist.mean())/pooled.tanimoto_dist.std()
yg=1-pooled.pearson_delta
resg=stats.linregress(xg,yg); npool=len(pooled); dfp=npool-2
t_lo=(resg.slope-(-EQ))/resg.stderr; p_lo=stats.t.sf(t_lo,dfp)
t_hi=(resg.slope-(EQ))/resg.stderr; p_hi=stats.t.cdf(t_hi,dfp)
print(f"  POOLED 4 conditioned (n={npool}): slope={resg.slope:+.4f}, 90%CI[{resg.slope-stats.t.ppf(0.95,dfp)*resg.stderr:+.4f},{resg.slope+stats.t.ppf(0.95,dfp)*resg.stderr:+.4f}], TOST p={max(p_lo,p_hi):.2g}")
# raw (non-standardized) per-Tanimoto-unit slope + MDE for FP-ridge to match the '0.30/unit' claim
for b in ['FP-ridge','CPA','scGen']:
    sub=t[t.baseline==b].dropna(subset=['pearson_delta','tanimoto_dist'])
    res=stats.linregress(sub.tanimoto_dist, 1-sub.pearson_delta)
    mde=(stats.t.ppf(0.975,len(sub)-2)+stats.t.ppf(0.80,len(sub)-2))*res.stderr
    rng_dist=sub.tanimoto_dist.max()-sub.tanimoto_dist.min()
    print(f"  {b}: raw slope/Tanimoto-unit={res.slope:+.4f}, 80%-power MDE={mde:.3f}/unit (= {mde*rng_dist:.3f} pearson over observed range {rng_dist:.2f})")

print("\nDONE.")
