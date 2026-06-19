#!/usr/bin/env python
"""Reqs 1 & 8: SECOND transcriptional cell-context win instance (NOT compounds).

Soskic CD4 activation atlas (data/C2/soskic, processed HVG h5ad on disk): resting (0h) vs TCR/CD28-
stimulated (16h). The SEEN perturbation is stimulation; the held cell-CONTEXT axis is CD4 state
(Naive vs Memory) — the naive->memory state-transfer axis the manuscript's C1 taxonomy defines
(Cano-Gamez analog). Leave-one-CD4-state-out: train the conditioned model + simple floors on (resting,
stim) for ONE state, predict the HELD state's stimulated response from its OWN resting cells. This is a
seen-perturbation / unseen-context TRANSCRIPTIONAL split, so a cell-context win here is a second clean
instance of the dissociation's blue half beyond C5 FP-ridge and C1 Kang scGen.

Pipeline: build a CellSet on the 381 shared genes (jointly re-standardized so the two files are
comparable), then run the framework build_split+audit and the SAME ScGenC1 latent adapter + 4 simple
floors used on Kang. CPU. Leak-safe: held state's stim cells never enter training.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import anndata as ad, scipy.sparse as spx
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ivcbench.data.schema import CellSet, CONTROL_TOKEN, validate_cellset
from ivcbench.splits.spec import SplitSpec
from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split
from ivcbench.baselines.simple import SIMPLE_BASELINES
from ivcbench.baselines.heavy import ScGenC1
from ivcbench.metrics.response import pearson_delta
from ivcbench.metrics.distribution import e_distance
from ivcbench.metrics.program import aucell
from scipy import stats

SOSKIC = ROOT / "data/C2/soskic"
REST = "restingCells_CD4only_HVGs_processed.h5ad"
STIM = "stimulatedCells_highlyActiveCD4_16h_HVGs_processed.h5ad"
IFN_GENES = ["ISG15","IFI6","MX1","MX2","OAS1","OAS2","IFIT1","IFIT3","ISG20",
             "STAT1","IRF7","IFI44","IFI44L","RSAD2","USP18"]
CAP = 8000  # per (state x condition) cell cap, for CPU tractability (balanced subsample)
OUT = ROOT / "results/C2/soskic_loct.csv"

def _dense(a):
    return a.toarray() if spx.issparse(a) else np.asarray(a)

def load_soskic():
    r = ad.read_h5ad(SOSKIC / REST)
    s = ad.read_h5ad(SOSKIC / STIM)
    shared = sorted(set(map(str, r.var_names)) & set(map(str, s.var_names)))
    print(f"shared genes = {len(shared)}", flush=True)
    rng = np.random.default_rng(0)
    frames, Xs, meta = [], [], []
    for adata, cond, ctrl in [(r, "control", True), (s, "stimulation", False)]:
        adata = adata[:, shared].copy()
        ct = adata.obs["Cell_type"].astype(str).to_numpy()
        for state in ["CD4_Naive", "CD4_Memory"]:
            idx = np.where(ct == state)[0]
            if len(idx) > CAP:
                idx = rng.choice(idx, CAP, replace=False)
            X = _dense(adata.X[idx]).astype(np.float32)
            Xs.append(X)
            meta.append(pd.DataFrame(dict(
                cell_type_coarse=state, cell_type_fine=state, perturbation=cond,
                condition=cond, donor_id=adata.obs["Donor"].astype(str).to_numpy()[idx],
                timepoint=("0h" if ctrl else "16h"), batch=adata.obs["Plate"].astype(str).to_numpy()[idx],
                is_control=ctrl)))
            print(f"  {state} {cond}: {len(idx)} cells", flush=True)
    X = np.vstack(Xs)
    obs = pd.concat(meta, ignore_index=True)
    # joint re-standardize across the merged matrix so resting/stim files share a scale
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    obs["perturbation"] = np.where(obs["is_control"], CONTROL_TOKEN, obs["perturbation"])
    cs = CellSet(X=X, obs=obs.reset_index(drop=True), var_names=shared,
                 side_info={}, uns=dict(dataset="soskic_CD4_16h", modality="rna"))
    validate_cellset(cs)
    return cs

def loct_spec(held_state):
    return SplitSpec(name=f"C2_soskic_loct_{held_state}", cluster="C2", key_col="cell_type_coarse",
                     held_values=[held_state], control_inference_only=True, strata_cols=["donor_id"],
                     registry_task="C1_LOCT",
                     note="held CD4 state's stim cells hidden; only its resting cells are inference input")

def main():
    cs = load_soskic()
    gs = np.asarray(cs.gene_index(IFN_GENES), dtype=int)
    print(f"IFN genes present in 381-panel: {len(gs)}", flush=True)
    rows = []
    for held in ["CD4_Naive", "CD4_Memory"]:
        sp = build_split(cs, loct_spec(held))
        audit = audit_split(cs, sp)
        test_X = cs.X[sp.test_idx]
        ctrl_cells = cs.X[sp.inference_input_idx] if len(sp.inference_input_idx) else cs.X[cs.obs.is_control.to_numpy()]
        ctrl_auc = float(aucell(test_X if len(gs)==0 else ctrl_cells, gs).mean()) if len(gs) else np.nan
        obs_ifn = (float(aucell(test_X, gs).mean()) - ctrl_auc) if len(gs) else np.nan
        # scGen
        sg = ScGenC1(); sg.fit(cs, sp, side_info=cs.side_info)
        pr = sg.predict(cs, sp, side_info=cs.side_info)
        sg_pd = float(pearson_delta(pr.pred_cells, test_X, pr.control_mean, sp.test_strata, None)["macro"])
        sg_ed = float(e_distance(pr.pred_cells, test_X, sp.test_strata)["macro"])
        sg_ifn = (float(aucell(pr.pred_cells, gs).mean()) - ctrl_auc) if len(gs) else np.nan
        rows.append(dict(held_state=held, model="scGen", family="latent",
                         pearson_delta=round(sg_pd,4), e_distance=round(sg_ed,4) if sg_ed==sg_ed else np.nan,
                         pred_ifn_delta=round(sg_ifn,4) if sg_ifn==sg_ifn else np.nan,
                         obs_ifn_delta=round(obs_ifn,4) if obs_ifn==obs_ifn else np.nan,
                         n_strata=int(len(np.unique(sp.test_strata))), leak_free=bool(audit["leak_free"])))
        for B in SIMPLE_BASELINES:
            b = B(); b.fit(cs, sp, side_info=cs.side_info); p = b.predict(cs, sp, side_info=cs.side_info)
            bpd = float(pearson_delta(p.pred_cells, test_X, p.control_mean, sp.test_strata, None)["macro"])
            bed = float(e_distance(p.pred_cells, test_X, sp.test_strata)["macro"])
            bifn = (float(aucell(p.pred_cells, gs).mean()) - ctrl_auc) if len(gs) else np.nan
            rows.append(dict(held_state=held, model=b.name, family="simple",
                             pearson_delta=round(bpd,4), e_distance=round(bed,4),
                             pred_ifn_delta=round(bifn,4) if bifn==bifn else np.nan,
                             obs_ifn_delta=round(obs_ifn,4) if obs_ifn==obs_ifn else np.nan,
                             n_strata=int(len(np.unique(sp.test_strata))), leak_free=bool(audit["leak_free"])))
        cur = [r for r in rows if r["held_state"]==held]
        print(f"[{held}] " + " ".join(f"{r['model']}={r['pearson_delta']:.3f}" for r in cur), flush=True)
    df = pd.DataFrame(rows); OUT.parent.mkdir(parents=True, exist_ok=True); df.to_csv(OUT, index=False)
    print(f"\nWROTE {OUT} ({len(df)} rows; leak_free={bool(df.leak_free.all())})")
    # cell-context verdict
    print("\n=== Soskic CD4 leave-one-state-out: scGen vs best-simple Pearson-Δ ===")
    for held in ["CD4_Naive","CD4_Memory"]:
        sub = df[df.held_state==held]
        sg = sub[sub.model=="scGen"].pearson_delta.iloc[0]
        bs = sub[sub.family=="simple"].pearson_delta.max()
        bsn = sub[sub.family=="simple"].sort_values("pearson_delta").model.iloc[-1]
        print(f"  held {held}: scGen {sg:.4f} vs best-simple {bs:.4f} ({bsn})  gap {sg-bs:+.4f}")

if __name__=="__main__":
    main()
