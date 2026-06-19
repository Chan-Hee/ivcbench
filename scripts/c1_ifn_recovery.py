#!/usr/bin/env python
"""Req 9: C1 type-I IFN program-recovery direct re-score (the deposited C1 AUCell col is 0/144).

Score the C1 LOCT predictions (simple baselines, CPU) through the type-I IFN AUCell gene set, using
the SAME aucell_delta_corr machinery the C5 pipeline uses, so the C1 cell-context claim can rest on
program recovery rather than only Pearson-Δ/E-distance. scGen requires the scperturbench_eval env and
is handled by a sibling script (c1_ifn_recovery_scgen.py); this script does the floor/simple half +
deposits a combined CSV slot. Leak-safe via framework build_split + audit. CPU only.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.data.loaders import kang as kang_mod
from ivcbench.clusters import c1
from ivcbench.baselines.simple import SIMPLE_BASELINES
from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split
from ivcbench.metrics.program import aucell_delta_corr
from ivcbench.metrics.response import pearson_delta

IFN_GENES = ["ISG15","IFI6","MX1","MX2","OAS1","OAS2","IFIT1","IFIT3","ISG20",
             "STAT1","IRF7","IFI44","IFI44L","RSAD2","USP18"]
OUT = ROOT / "results/C1/ifn_recovery.csv"

def main():
    cs = kang_mod.load()
    gs_idx = np.asarray(cs.gene_index(IFN_GENES), dtype=int)
    print(f"IFN gene set: {len(gs_idx)}/{len(IFN_GENES)} genes present in HVG panel", flush=True)
    lineages = sorted(cs.obs["cell_type_coarse"].astype(str).unique())
    print("lineages:", lineages, flush=True)
    ctrl_mask = cs.obs["is_control"].astype(bool).to_numpy()
    ctrl_all = cs.X[ctrl_mask]
    rows=[]
    for lin in lineages:
        spec = c1.coarse_loct(lin)
        sp = build_split(cs, spec)
        audit = audit_split(cs, sp)
        test_X = cs.X[sp.test_idx]
        # control cells = held lineage's own controls (inference input)
        if hasattr(sp,'inference_input_idx') and len(sp.inference_input_idx):
            ctrl_cells = cs.X[sp.inference_input_idx]
        else:
            ctrl_cells = ctrl_all
        for B in SIMPLE_BASELINES:
            b=B(); b.fit(cs,sp,side_info=cs.side_info)
            pred=b.predict(cs,sp,side_info=cs.side_info)
            # bulk pearson_delta (sanity vs deposited)
            resp=pearson_delta(pred.pred_cells, test_X, pred.control_mean, sp.test_strata, None)
            ac=aucell_delta_corr(pred.pred_cells, test_X, ctrl_cells, gs_idx, sp.test_strata)
            rows.append(dict(lineage=lin, baseline=b.name, model_family='simple',
                             pearson_delta=round(float(resp['macro']),4),
                             ifn_recovery=round(float(ac['corr']),4),
                             ifn_obs_mean_delta=round(float(ac.get('obs_mean_delta',np.nan)),4),
                             n_strata=int(len(np.unique(sp.test_strata))),
                             leak_free=bool(audit['leak_free'])))
        print(f"{lin:14s} done; "+" ".join(f"{r['baseline']}={r['ifn_recovery']:+.3f}" for r in rows if r['lineage']==lin), flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nWROTE {OUT} ({len(df)} rows; all leak_free={bool(df.leak_free.all())})")
    print("\n=== simple-baseline type-I IFN AUCell-Δ recovery (mean over 8 lineages) ===")
    print(df.groupby('baseline').agg(ifn=('ifn_recovery','mean'),
          ifn_nonzero=('ifn_recovery',lambda x:(x!=0).sum()), bulk=('pearson_delta','mean')).round(4).to_string())
    print("\nobserved type-I IFN-Δ magnitude per lineage (size of the true ctrl->stim IFN shift):")
    print(df[df.baseline=='cell-mean'][['lineage','ifn_obs_mean_delta']].to_string(index=False))

if __name__=='__main__':
    main()
