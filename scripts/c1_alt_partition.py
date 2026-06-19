#!/usr/bin/env python
"""Req 8: alternative Kang lineage partition for the cell-context win (robustness to partition choice).

The published C1 LOCT uses Kang's 8-class `cell` labels. A reviewer can ask whether the scGen cell-context
advantage is an artifact of that particular partition. We re-run the IDENTICAL leave-one-lineage-out logic
on an ALTERNATIVE 6-class partition that merges the two monocyte subsets (CD14+ + FCGR3A+ -> Monocyte)
and the two T subsets (CD4T + CD8T -> Tcell), leaving {Tcell, Monocyte, B, NK, DC, Mk}. If the scGen
advantage on the myeloid/lymphoid lineages survives the re-grouping, the cell-context win is not a
partition artifact. scGen via the ScGenC1 adapter (CPU), simple floors, leak-safe. NOT a second dataset
(Soskic processed files are per-file z-scored and unusable for a cross-celltype shift -- shift corr
naive-vs-memory = -0.99, a scaling artifact), but a second PARTITION of the one clean transcriptional
cytokine atlas.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from ivcbench.data.loaders import kang as kang_mod
from ivcbench.splits.spec import SplitSpec
from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split
from ivcbench.baselines.simple import SIMPLE_BASELINES
from ivcbench.baselines.heavy import ScGenC1
from ivcbench.metrics.response import pearson_delta
from ivcbench.metrics.distribution import e_distance
from scipy import stats

MERGE = {"Mono_CD14":"Monocyte","Mono_FCGR3A":"Monocyte","CD4T":"Tcell","CD8T":"Tcell"}
OUT = ROOT/"results/C1/alt_partition.csv"

def spec(held):
    return SplitSpec(name=f"C1_altloct_{held}", cluster="C1", key_col="cell_type_alt",
                     held_values=[held], control_inference_only=True, strata_cols=["donor_id"],
                     registry_task="C1_LOCT", note="alternative 6-class Kang partition LOCT")

def main():
    cs = kang_mod.load()
    coarse = cs.obs["cell_type_coarse"].astype(str)
    cs.obs["cell_type_alt"] = coarse.map(lambda x: MERGE.get(x, x))
    parts = sorted(cs.obs["cell_type_alt"].unique())
    print("alt partition:", parts, "| counts:", cs.obs.cell_type_alt.value_counts().to_dict(), flush=True)
    rows=[]
    for held in parts:
        sp = build_split(cs, spec(held)); audit = audit_split(cs, sp)
        test_X = cs.X[sp.test_idx]
        sg = ScGenC1(); sg.fit(cs, sp, side_info=cs.side_info); pr = sg.predict(cs, sp, side_info=cs.side_info)
        sg_pd = float(pearson_delta(pr.pred_cells, test_X, pr.control_mean, sp.test_strata, None)["macro"])
        sg_ed = float(e_distance(pr.pred_cells, test_X, sp.test_strata)["macro"])
        rows.append(dict(held=held, model="scGen", family="latent", pearson_delta=round(sg_pd,4),
                         e_distance=round(sg_ed,4), n_test=int(len(sp.test_idx)),
                         n_strata=int(len(np.unique(sp.test_strata))), leak_free=bool(audit["leak_free"])))
        for B in SIMPLE_BASELINES:
            b=B(); b.fit(cs,sp,side_info=cs.side_info); p=b.predict(cs,sp,side_info=cs.side_info)
            rows.append(dict(held=held, model=b.name, family="simple",
                             pearson_delta=round(float(pearson_delta(p.pred_cells,test_X,p.control_mean,sp.test_strata,None)["macro"]),4),
                             e_distance=round(float(e_distance(p.pred_cells,test_X,sp.test_strata)["macro"]),4),
                             n_test=int(len(sp.test_idx)), n_strata=int(len(np.unique(sp.test_strata))),
                             leak_free=bool(audit["leak_free"])))
        cur=[r for r in rows if r["held"]==held]
        print(f"[{held}] "+" ".join(f"{r['model']}={r['pearson_delta']:.3f}" for r in cur), flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
    print(f"\nWROTE {OUT} ({len(df)} rows; leak_free={bool(df.leak_free.all())})")
    print("\n=== alt-partition scGen vs best-simple gap per held lineage ===")
    gaps=[]
    for held in parts:
        sub=df[df.held==held]; sg=sub[sub.model=="scGen"].pearson_delta.iloc[0]
        bs=sub[sub.family=="simple"].pearson_delta.max(); bsn=sub[sub.family=="simple"].sort_values("pearson_delta").model.iloc[-1]
        gaps.append(sg-bs); print(f"  {held:10s}: scGen {sg:.4f} vs best-simple {bs:.4f} ({bsn}) gap {sg-bs:+.4f}")
    gaps=np.array(gaps)
    print(f"\n{int((gaps>0).sum())}/{len(gaps)} positive; mean gap {gaps.mean():+.4f}; "
          f"sign-test p={stats.binomtest((gaps>0).sum(),len(gaps),0.5).pvalue:.3f}")

if __name__=="__main__":
    main()
