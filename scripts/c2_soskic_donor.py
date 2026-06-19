#!/usr/bin/env python
"""Soskic CD4-activation DONOR-axis benchmark (leave-one-donor-out, n=106).

control = 0h resting, perturbed = 16h stimulated (main analysis). Hold out ONE donor: train scGen + the 4
simple baselines on the OTHER donors' (resting, stim) cells, predict the held donor's 16h response from its
OWN 0h cells. Lineage (CD4 Naive/Memory) is a within-donor stratum covariate. Leak-safe: the held donor's
stim cells never enter training. Same ScGenC1 adapter + SIMPLE_BASELINES + pearson_delta/e_distance used on
Kang, so the verdict is directly comparable to the Kang donor row.

n = 106 QC'd donors (the recruited cohort is 119; the processed portal data has 106 paired 0h/16h donors).

Args: --test (first K donors), --chunk I N (donor shard I of N), --cap C (per donor x condition cell
      cap), --out PATH. For GPU scGen, use the cloned CUDA-JAX env via
      IVCBENCH_SCPERTURBENCH_EVAL_PYTHON=<conda-env>/bin/python
      plus --scgen-accelerator gpu --scgen-devices 1.
"""
from __future__ import annotations
import sys, argparse, os
from pathlib import Path
import numpy as np, pandas as pd
import anndata as ad, scipy.sparse as spx
from scipy import stats
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

SOSKIC = ROOT / "data/C2/soskic"
REST = "restingCells_CD4only_HVGs_processed.h5ad"
STIM = "stimulatedCells_highlyActiveCD4_16h_HVGs_processed.h5ad"

# Same immune-program vocabulary already used elsewhere in the benchmark: T-cell activation and
# IL2/STAT5 from C3, type-I IFN from C1/C5, plus a compact type-II IFN module for activated CD4 T cells.
SOSKIC_PROGRAMS: dict[str, list[str]] = {
    "T_cell_activation": ["CD69", "IL2RA", "CD40LG", "TNFRSF9", "NR4A1", "NR4A2", "NR4A3",
                          "EGR1", "EGR2", "IRF4", "REL", "NFKBIA", "CD28", "TNFRSF4"],
    "IL2_STAT5": ["IL2RA", "IL2RB", "IL2RG", "STAT5A", "STAT5B", "CISH", "SOCS1", "SOCS3",
                  "BCL2", "MYC", "IL2"],
    "type_I_IFN": ["ISG15", "IFI6", "MX1", "MX2", "OAS1", "OAS2", "OAS3", "OASL", "IFIT1",
                   "IFIT2", "IFIT3", "IFITM1", "IFITM3", "ISG20", "IRF7", "STAT1", "STAT2",
                   "RSAD2", "USP18", "IFI44", "IFI44L", "BST2", "XAF1", "HERC5", "LY6E"],
    "type_II_IFN": ["IFNG", "STAT1", "IRF1", "CXCL9", "CXCL10", "CXCL11", "GBP1", "GBP2",
                    "GBP5", "TAP1", "PSMB8", "PSMB9", "HLA-DRA", "HLA-DRB1", "SOCS1"],
}


def _dense(a):
    return a.toarray() if spx.issparse(a) else np.asarray(a)


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.ones_like(p)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranks = np.arange(1, len(order) + 1)
    vals = p[order] * len(order) / ranks
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    q[order] = np.clip(vals, 0, 1)
    return q


def response_gene_idx(cs: CellSet, train_idx: np.ndarray, max_genes: int = 200, min_genes: int = 50) -> np.ndarray:
    """Training-only control-vs-stim response genes for the held-donor fold.

    The task requires response genes for evaluation but forbids using them for model selection. We therefore
    select them inside each fold from training donors only and pass the indices only to the metric call.
    """
    tr_obs = cs.obs.iloc[train_idx]
    is_ctrl = tr_obs["is_control"].to_numpy().astype(bool)
    ctrl = cs.X[train_idx[is_ctrl]]
    stim = cs.X[train_idx[~is_ctrl]]
    if len(ctrl) < 3 or len(stim) < 3:
        return np.arange(cs.X.shape[1])
    t = stats.ttest_ind(stim, ctrl, axis=0, equal_var=False, nan_policy="omit")
    p = np.nan_to_num(t.pvalue, nan=1.0, posinf=1.0, neginf=1.0)
    q = _bh_qvalues(p)
    effect = np.abs(stim.mean(0) - ctrl.mean(0))
    ranked = np.argsort(-effect)
    sig = ranked[q[ranked] < 0.05]
    if len(sig) < min_genes:
        sig = ranked[:min(min_genes, len(ranked))]
    return np.asarray(sig[:min(max_genes, len(sig))], dtype=int)


def e_distance_basis(cs: CellSet, train_idx: np.ndarray, max_cells: int = 5000) -> np.ndarray:
    """Leak-safe PCA basis sample for E-distance, drawn only from training cells."""
    if len(train_idx) <= max_cells:
        return cs.X[train_idx]
    rng = np.random.default_rng(0)
    return cs.X[rng.choice(train_idx, size=max_cells, replace=False)]


def program_delta_mae(pred_cells: np.ndarray, test_cells: np.ndarray, control_cells: np.ndarray,
                      pred_strata: np.ndarray, control_strata: np.ndarray, cs: CellSet) -> dict:
    """Mean absolute error of AUCell program shifts, averaged over program × lineage strata.

    Raw per-program columns are retained for source data; lower is better. `aucell_delta_score = -MAE`
    is also emitted so a positive scGen-minus-baseline comparison has the same sign convention as Pearson.
    """
    out: dict[str, float | int] = {}
    vals = []
    pred_strata = np.asarray([str(x).split("=", 1)[-1] for x in pred_strata])
    control_strata = np.asarray([str(x).split("=", 1)[-1] for x in control_strata])
    for name, genes in SOSKIC_PROGRAMS.items():
        gs = np.asarray(cs.gene_index(genes), dtype=int)
        out[f"aucell::{name}_n_genes"] = int(len(gs))
        if len(gs) == 0:
            out[f"aucell::{name}_mae"] = np.nan
            continue
        errs = []
        for s in np.unique(pred_strata):
            tm = pred_strata == s
            cm = control_strata == s
            if tm.sum() == 0 or cm.sum() == 0:
                continue
            ctrl = float(aucell(control_cells[cm], gs).mean())
            obs_d = float(aucell(test_cells[tm], gs).mean()) - ctrl
            pred_d = float(aucell(pred_cells[tm], gs).mean()) - ctrl
            errs.append(abs(pred_d - obs_d))
        mae = float(np.mean(errs)) if errs else np.nan
        out[f"aucell::{name}_mae"] = mae
        if mae == mae:
            vals.append(mae)
    out["aucell_delta_mae"] = float(np.mean(vals)) if vals else np.nan
    out["aucell_delta_score"] = -out["aucell_delta_mae"] if out["aucell_delta_mae"] == out["aucell_delta_mae"] else np.nan
    return out


def load_soskic_donor(cap_per_donor_cond=300):
    """CellSet on the shared HVGs, jointly re-standardized; per (donor x condition x celltype) cap so all
    106 donors are represented comparably."""
    r = ad.read_h5ad(SOSKIC / REST)
    s = ad.read_h5ad(SOSKIC / STIM)
    shared = sorted(set(map(str, r.var_names)) & set(map(str, s.var_names)))
    print(f"shared genes = {len(shared)}", flush=True)
    rng = np.random.default_rng(0)
    Xs, meta = [], []
    for adata, cond, ctrl in [(r, "stimulation", True), (s, "stimulation", False)]:
        adata = adata[:, shared]
        ct = adata.obs["Cell_type"].astype(str).to_numpy()
        don = adata.obs["Donor"].astype(str).to_numpy()
        plate = adata.obs["Plate"].astype(str).to_numpy()
        for state in ["CD4_Naive", "CD4_Memory"]:
            for d in np.unique(don):
                idx = np.where((ct == state) & (don == d))[0]
                if len(idx) == 0:
                    continue
                if len(idx) > cap_per_donor_cond:
                    idx = rng.choice(idx, cap_per_donor_cond, replace=False)
                Xs.append(_dense(adata.X[idx]).astype(np.float32))
                meta.append(pd.DataFrame(dict(
                    cell_type_coarse=state, cell_type_fine=state,
                    perturbation=(CONTROL_TOKEN if ctrl else cond), condition=cond,
                    donor_id=d, timepoint=("0h" if ctrl else "16h"), is_control=ctrl,
                    batch=plate[idx])))
    X = np.vstack(Xs)
    obs = pd.concat(meta, ignore_index=True)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    cs = CellSet(X=X, obs=obs.reset_index(drop=True), var_names=shared,
                 side_info={}, uns=dict(dataset="soskic_CD4_16h_donor", modality="rna"))
    validate_cellset(cs)
    print(f"CellSet: {cs.X.shape[0]} cells x {cs.X.shape[1]} genes; donors={obs.donor_id.nunique()}", flush=True)
    return cs


def lodo_spec(donor):
    return SplitSpec(name=f"C2_soskic_LODO_{donor}", cluster="C2", key_col="donor_id",
                     held_values=[donor], control_inference_only=True, strata_cols=["cell_type_coarse"],
                     registry_task="C1_LODO",
                     note="held donor's 16h-stim hidden; its 0h cells are the inference input")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--chunk", type=int, nargs=2, default=None, metavar=("I", "N"))
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--out", type=str, default=str(ROOT / "results/C2/soskic_donor_raw.csv"))
    ap.add_argument("--skip-existing", action="store_true",
                    help="Resume from --out by skipping donors with all 5 model rows already written.")
    ap.add_argument("--no-scgen", action="store_true", help="Run simple baselines only (debug / partial rescue).")
    ap.add_argument("--epochs", type=int, default=None, help="Override IVCBENCH_SCGEN_EPOCHS for scGen.")
    ap.add_argument("--scgen-accelerator", choices=["cpu", "gpu", "auto"], default=None,
                    help="Forwarded to the scGen runner through IVCBENCH_SCGEN_ACCELERATOR.")
    ap.add_argument("--scgen-devices", default=None,
                    help="Forwarded to the scGen runner through IVCBENCH_SCGEN_DEVICES (usually 1).")
    args = ap.parse_args()

    if args.epochs is not None:
        os.environ["IVCBENCH_SCGEN_EPOCHS"] = str(args.epochs)
    if args.scgen_accelerator is not None:
        os.environ["IVCBENCH_SCGEN_ACCELERATOR"] = args.scgen_accelerator
    if args.scgen_devices is not None:
        os.environ["IVCBENCH_SCGEN_DEVICES"] = str(args.scgen_devices)

    cs = load_soskic_donor(args.cap)
    donors = sorted(cs.obs.donor_id.unique())
    if args.test:
        donors = donors[: args.test]
    if args.chunk:
        i, n = args.chunk
        donors = donors[i::n]
    out_path = Path(args.out)
    expected_models = {"ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"}
    if not args.no_scgen:
        expected_models.add("scGen")
    rows = []
    done = set()
    if args.skip_existing and out_path.exists():
        old = pd.read_csv(out_path)
        rows = old.to_dict("records")
        for donor, sub in old.groupby("donor"):
            if expected_models.issubset(set(sub["model"].astype(str))):
                done.add(str(donor))
        donors = [d for d in donors if d not in done]
        print(f"resume: loaded {len(old)} rows from {out_path}; skipping {len(done)} complete donors", flush=True)
    est_gb = cs.X.nbytes / 1e9
    print(f"running {len(donors)} donor folds; gpu={args.gpu}; cap={args.cap}; "
          f"matrix≈{est_gb:.2f} GB; scGen epochs={os.environ.get('IVCBENCH_SCGEN_EPOCHS', '60')}; "
          f"accelerator={os.environ.get('IVCBENCH_SCGEN_ACCELERATOR', 'cpu')}", flush=True)

    import time
    for k, d in enumerate(donors):
        t0 = time.time()
        spec = lodo_spec(d)
        sp = build_split(cs, spec)
        audit = audit_split(cs, sp)
        test_X = cs.X[sp.test_idx]
        strat = sp.test_strata
        ctrl_X = cs.X[sp.inference_input_idx]
        ctrl_strat = cs.obs.iloc[sp.inference_input_idx]["cell_type_coarse"].astype(str).to_numpy()
        rg = response_gene_idx(cs, sp.train_idx)
        ed_basis = e_distance_basis(cs, sp.train_idx)
        # scGen (conditioned)
        if not args.no_scgen:
            sg = ScGenC1()
            if args.gpu is not None:
                sg.cuda_device = args.gpu
            sg.fit(cs, sp, side_info=cs.side_info)
            pr = sg.predict(cs, sp, side_info=cs.side_info)
            prog = program_delta_mae(pr.pred_cells, test_X, ctrl_X, strat, ctrl_strat, cs)
            rows.append(dict(axis="Donor", dataset="Soskic CD4 activation", timepoint="16h",
                             split=spec.name, donor=d, model="scGen", family="latent",
                             pearson_delta=round(float(pearson_delta(pr.pred_cells, test_X, pr.control_mean, strat, rg)["macro"]), 4),
                             e_distance=round(float(e_distance(pr.pred_cells, test_X, strat, fit_on=ed_basis)["macro"]), 4),
                             **{k: (round(float(v), 4) if isinstance(v, float) and v == v else v) for k, v in prog.items()},
                             n_test=int(len(sp.test_idx)), n_strata=int(len(np.unique(strat))),
                             n_response_genes=int(len(rg)), leak_free=bool(audit["leak_free"])))
        for B in SIMPLE_BASELINES:
            b = B(); b.fit(cs, sp, side_info=cs.side_info); p = b.predict(cs, sp, side_info=cs.side_info)
            prog = program_delta_mae(p.pred_cells, test_X, ctrl_X, strat, ctrl_strat, cs)
            rows.append(dict(axis="Donor", dataset="Soskic CD4 activation", timepoint="16h",
                             split=spec.name, donor=d, model=b.name, family="simple",
                             pearson_delta=round(float(pearson_delta(p.pred_cells, test_X, p.control_mean, strat, rg)["macro"]), 4),
                             e_distance=round(float(e_distance(p.pred_cells, test_X, strat, fit_on=ed_basis)["macro"]), 4),
                             **{k: (round(float(v), 4) if isinstance(v, float) and v == v else v) for k, v in prog.items()},
                             n_test=int(len(sp.test_idx)), n_strata=int(len(np.unique(strat))),
                             n_response_genes=int(len(rg)),
                             leak_free=bool(audit["leak_free"])))
        cur = {r["model"]: r["pearson_delta"] for r in rows if r["donor"] == d}
        print(f"[{k+1}/{len(donors)}] donor {d} ({time.time()-t0:.0f}s) " +
              " ".join(f"{m}={v:.3f}" for m, v in cur.items()), flush=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)  # checkpoint after every donor

    print(f"\nWROTE {out_path} ({len(rows)} rows; leak_free={bool(pd.DataFrame(rows).leak_free.all())})")


if __name__ == "__main__":
    Path(ROOT / "results/C2").mkdir(parents=True, exist_ok=True)
    main()
