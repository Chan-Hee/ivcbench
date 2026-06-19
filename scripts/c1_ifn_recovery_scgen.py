#!/usr/bin/env python
"""Req 9 (scGen half): C1 type-I IFN program recovery for scGen, run in scperturbench_eval (CPU).

The per-DONOR-stratum AUCell-Δ correlation is structurally degenerate on C1 (single seen perturbation
IFN-β, donor strata, every model emits one tiled profile -> constant per-stratum pred -> corr 0). So we
instead score the PER-LINEAGE IFN-Δ magnitude recovery: across the 8 held lineages, does the model's
predicted type-I IFN AUCell shift (pred-ctrl) track the observed shift (stim-ctrl)? The simple floor
ANTI-tracks it (cell-mean r=-0.91, it predicts a near-constant shift). This tests whether the conditioned
latent (scGen) recovers the lineage-specific IFN-program magnitude the floor cannot. Leak-safe: held
lineage's stim cells never enter training (framework build_split + audit); scGen runner decodes the held
lineage's OWN control + the global latent δ. CPU via the scGen-C1 subprocess adapter.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.data.loaders import kang as kang_mod
from ivcbench.clusters import c1
from ivcbench.baselines.heavy import ScGenC1
from ivcbench.baselines.simple import SIMPLE_BASELINES
from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split
from ivcbench.metrics.program import aucell
from ivcbench.metrics.response import pearson_delta
from scipy import stats

IFN_GENES = ["ISG15","IFI6","MX1","MX2","OAS1","OAS2","IFIT1","IFIT3","ISG20",
             "STAT1","IRF7","IFI44","IFI44L","RSAD2","USP18"]
OUT = ROOT / "results/C1/ifn_recovery_perlineage.csv"

def main():
    cs = kang_mod.load()
    gs = np.asarray(cs.gene_index(IFN_GENES), dtype=int)
    lineages = sorted(cs.obs["cell_type_coarse"].astype(str).unique())
    rows = []
    for lin in lineages:
        sp = build_split(cs, c1.coarse_loct(lin))
        audit = audit_split(cs, sp)
        assert audit["leak_free"], f"LEAK {lin}"
        test_X = cs.X[sp.test_idx]
        ctrl = cs.X[sp.inference_input_idx] if len(sp.inference_input_idx) else cs.X[cs.obs["is_control"].astype(bool).to_numpy()]
        ctrl_auc = float(aucell(ctrl, gs).mean())
        obs_ifn = float(aucell(test_X, gs).mean()) - ctrl_auc
        # scGen
        sg = ScGenC1(); sg.fit(cs, sp, side_info=cs.side_info)
        pred = sg.predict(cs, sp, side_info=cs.side_info)
        sg_ifn = float(aucell(pred.pred_cells, gs).mean()) - ctrl_auc
        sg_bulk = float(pearson_delta(pred.pred_cells, test_X, pred.control_mean, sp.test_strata, None)["macro"])
        rows.append(dict(lineage=lin, model="scGen", obs_ifn_delta=round(obs_ifn,4),
                         pred_ifn_delta=round(sg_ifn,4), bulk_pearson_delta=round(sg_bulk,4),
                         leak_free=bool(audit["leak_free"])))
        # simple floors for the same lineage
        for B in SIMPLE_BASELINES:
            b = B(); b.fit(cs, sp, side_info=cs.side_info); p = b.predict(cs, sp, side_info=cs.side_info)
            ifn = float(aucell(p.pred_cells, gs).mean()) - ctrl_auc
            bulk = float(pearson_delta(p.pred_cells, test_X, p.control_mean, sp.test_strata, None)["macro"])
            rows.append(dict(lineage=lin, model=b.name, obs_ifn_delta=round(obs_ifn,4),
                             pred_ifn_delta=round(ifn,4), bulk_pearson_delta=round(bulk,4),
                             leak_free=bool(audit["leak_free"])))
        print(f"{lin:14s} obs_ifn={obs_ifn:+.4f} scGen_pred_ifn={sg_ifn:+.4f} scGen_bulk={sg_bulk:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nWROTE {OUT} ({len(df)} rows, leak_free={bool(df.leak_free.all())})")
    print("\n=== per-lineage IFN-Δ magnitude recovery: corr(pred_ifn_delta, obs_ifn_delta) over 8 lineages ===")
    for m in df.model.unique():
        sub = df[df.model==m]
        if sub.pred_ifn_delta.std() < 1e-9:
            print(f"  {m:12s}: pred IFN-Δ has ~0 variance across lineages"); continue
        r,p = stats.pearsonr(sub.pred_ifn_delta, sub.obs_ifn_delta)
        rho,_ = stats.spearmanr(sub.pred_ifn_delta, sub.obs_ifn_delta)
        err = float(np.abs(sub.pred_ifn_delta.values - sub.obs_ifn_delta.values).mean())
        print(f"  {m:12s}: Pearson r={r:+.3f} (p={p:.3f}), Spearman rho={rho:+.3f}, mean|err|={err:.4f}")

if __name__=="__main__":
    main()
