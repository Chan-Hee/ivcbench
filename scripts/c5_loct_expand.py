#!/usr/bin/env python
"""Expand the C5 cell-context (LOCT) axis from 4 coarse lineages to 6 fine immune lineages.

The published C5 LOCT splits use OP3's 4-class `cell_type` (T cells / B cells / Myeloid / NK), giving
n=4 held-out contexts — too few for a Wilcoxon to reach p<0.125. OP3 also carries a 6-class
`cell_type_orig` (T cells CD4+, T cells CD8+, T regulatory cells, NK, Myeloid, B), which the loader's
_CT_MAP already maps to CD4T/CD8T/Treg/NK/B/Mono. Running the SAME leave-one-cell-type-out logic on the
finer column gives n=6 held-out contexts, all leak-audited through the framework's run_job (same audit,
same metric). This is a properly-powered companion to the 4-lineage headline, not a new model.

Reuses: ivcbench loader machinery (with cell_type_orig as the cell-type column), SplitSpec LOCT,
run_job (leak audit + pearson_delta). FP-ridge is the headline chemistry model; simple baselines are the
floor. Pure CPU.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivcbench.baselines.chemistry import FPRidge  # noqa: E402
from ivcbench.baselines.simple import SIMPLE_BASELINES  # noqa: E402
from ivcbench.clusters.c5 import cross_celltype_loct, C5_PROGRAMS  # noqa: E402
from ivcbench.data.loaders import op3 as op3mod  # noqa: E402
from ivcbench.data.preprocess import PreprocessConfig, preprocess  # noqa: E402
from ivcbench.data.schema import CONTROL_TOKEN  # noqa: E402
from ivcbench.runner.run import run_job  # noqa: E402

OP3_PATH = os.environ.get("IVCBENCH_OP3_PATH",
                          str(ROOT / "data/C5/op3/GSE279945_sc_counts_processed.h5ad"))


def load_fine(subsample_per_group: int = 40, n_hvg: int = 2000, seed: int = 0):
    """Identical to op3.load but reads the 6-class cell_type_orig column for the cell-type axis."""
    import anndata
    ad = anndata.read_h5ad(OP3_PATH, backed="r")
    obs = ad.obs.copy()
    c_ct = op3mod._col(obs, "cell_type_orig")  # <-- the only change vs op3.load
    c_sm = op3mod._col(obs, "sm_name", "compound", "perturbation")
    c_sml = op3mod._col(obs, "SMILES", "smiles", "canonical_smiles")
    c_ctrl = op3mod._col(obs, "control", "is_control")
    c_donor = op3mod._col(obs, "donor_id", "donor")
    c_plate = op3mod._col(obs, "plate_name", "plate", "library_id")
    assert c_ct is not None and c_sm is not None

    is_ctrl = (obs[c_ctrl].astype(str).str.lower().isin(["true", "1"]) if c_ctrl is not None
               else obs[c_sm].astype(str).str.contains("Dimethyl", case=False, na=False))

    rng = np.random.default_rng(seed)
    key = obs[c_sm].astype(str) + "|" + obs[c_ct].astype(str) + "|" + (
        obs[c_donor].astype(str) if c_donor else "d0")
    pos = np.arange(len(obs))
    keep = []
    for _, g in pd.Series(pos).groupby(key.to_numpy()):
        v = g.to_numpy()
        keep.append(v if len(v) <= subsample_per_group
                    else rng.choice(v, subsample_per_group, replace=False))
    idx = np.sort(np.concatenate(keep))

    sub = ad[idx].to_memory()
    counts = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    sobs = sub.obs
    sm = sobs[c_sm].astype(str).to_numpy()
    ctrl = is_ctrl.to_numpy()[idx]
    pert = np.where(ctrl, CONTROL_TOKEN, sm)
    ct = sobs[c_ct].astype(str).map(lambda x: op3mod._CT_MAP.get(x, x)).to_numpy()
    donor = sobs[c_donor].astype(str).to_numpy() if c_donor else np.array(["d0"] * len(sobs))
    plate = sobs[c_plate].astype(str).to_numpy() if c_plate else donor

    obs_out = pd.DataFrame({
        "cell_type_coarse": ct, "cell_type_fine": ct, "perturbation": pert,
        "condition": "24h", "donor_id": donor, "timepoint": "24h",
        "batch": plate, "is_control": ctrl,
    })
    fingerprint = {}
    if c_sml is not None:
        sml_map = (sobs[[c_sm, c_sml]].astype(str).drop_duplicates(c_sm)
                   .set_index(c_sm)[c_sml].to_dict())
        for name, smi in sml_map.items():
            if name in set(pert) and name != CONTROL_TOKEN:
                fp = op3mod._morgan(smi)
                if fp is not None:
                    fingerprint[name] = fp
    cs = preprocess(counts, list(sub.var_names), obs_out, side_info={"fingerprint": fingerprint},
                    uns={"dataset": "op3_GSE279945_fine", "accession": "GSE279945",
                         "n_cells_total": int(ad.n_obs), "immune_program": {"immunomod_moa": []}},
                    cfg=PreprocessConfig(n_hvg=n_hvg))
    cs.uns["compounds"] = sorted(set(cs.obs["perturbation"]) - {CONTROL_TOKEN})
    return cs


def main():
    cs = load_fine()
    lineages = sorted(set(cs.obs["cell_type_coarse"]))
    print("fine lineages:", lineages, "| n cells:", cs.X.shape[0], flush=True)
    progs = {k: v for k, v in C5_PROGRAMS.items()}
    rows = []
    for lin in lineages:
        spec = cross_celltype_loct(held_lineage=lin)
        for B in list(SIMPLE_BASELINES) + [FPRidge]:
            try:
                r = run_job(cs, spec, B(), seed=0, immune_programs=progs)
            except Exception as e:  # leak errors must surface
                print(f"  !! {lin} {getattr(B(), 'name', B)} -> {type(e).__name__}: {e}", flush=True)
                raise
            rows.append({k: r.get(k) for k in
                         ["baseline", "family", "split", "action", "ran", "leak_free",
                          "n_train", "n_test", "n_test_strata", "pearson_delta",
                          "pearson_delta_lo", "pearson_delta_hi", "e_distance"]})
            print(f"  {lin:6s} {r.get('baseline'):12s} ran={r.get('ran')} leak_free={r.get('leak_free')} "
                  f"pearson_delta={r.get('pearson_delta')}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/C5/loct_fine6.csv", index=False)
    print("wrote results/C5/loct_fine6.csv", flush=True)


if __name__ == "__main__":
    main()
