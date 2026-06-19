#!/usr/bin/env python
"""C3 simple nearest-gene perturbation-representation baseline (supplementary).

Question: is the unseen-gene failure REPRESENTATIONAL (any gene-side prior fails) rather than just a
deep-model failure? We implement a trivial perturbation-representation baseline for the C3 true
leave-one-gene-out task and score it on the SAME response-direction Pearson-Δ, SAME splits/units,
SAME downstream-only rule, against the SAME universal floor (cell-mean / linear-PCA) as the deep
models (GEARS / AttentionPert) already deposited in results/C3/results_raw.csv.

Baseline definition (no held effect ever used → leak-safe):
  For a held-out target gene g (a test stratum), predict every test cell's profile as
      pred = control_mean + (observed training effect of g's NEAREST TRAINING gene g*),
  i.e. predicted Δ(g) = Δ_obs_train(g*) = mean(treated cells of g*) − control_mean_train.
  Two nearest-gene priors, computed ONLY from training-visible information:
    (1) coexpr  — g* = argmax over training genes of Pearson correlation between the held gene's
                  mean expression profile (across that gene's TREATED cells, which exist in the data
                  as the prediction TARGET context but whose EFFECT is never read) ... NO — to stay
                  strictly leak-safe we use the gene's CONTROL-cell co-expression vector: the
                  genome-wide mean expression of control cells is identical for all genes, so co-
                  expression similarity is defined on the gene-as-a-FEATURE axis: corr between
                  column g and column g* of the control-cell expression matrix (a pure co-expression
                  graph on control cells, never touching any perturbation effect). g and g* must both
                  be measured genes (in var_names).
    (2) go      — g* = argmax Jaccard(GO(g), GO(g*)) over training genes, using cached
                  gene2go (data/_assets/gears/gene2go_all.pkl): dict[symbol -> set(GO terms)].
  Ties / no-neighbour (gene absent from var_names for coexpr, or empty GO for go) → fall back to the
  cell-mean training effect (the union mean), which is exactly the floor; documented.

Output: results_raw-schema rows for nearest-gene-coexpr and nearest-gene-go appended for every
(dataset, split). Also a self-check that reproduces cell-mean's Pearson-Δ from this harness (anchor).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivcbench.clusters import c3                       # noqa: E402
from ivcbench.data.schema import CONTROL_TOKEN         # noqa: E402
from ivcbench.metrics.response import pearson_delta    # noqa: E402
from ivcbench.metrics.stats import bootstrap_ci        # noqa: E402
from ivcbench.splits.builder import build_split        # noqa: E402

GENE2GO_PKL = ROOT / "data/_assets/gears/gene2go_all.pkl"

DATASETS = {
    "shifrut":            ("ivcbench.data.loaders.shifrut", "load", {}, "KO"),
    "schmidt":            ("ivcbench.data.loaders.schmidt", "load", {}, "CRISPRa"),
    "mccutcheon_CRISPRi": ("ivcbench.data.loaders.mccutcheon", "load", {"modality": "CRISPRi"}, "CRISPRi"),
    "mccutcheon_CRISPRa": ("ivcbench.data.loaders.mccutcheon", "load", {"modality": "CRISPRa"}, "CRISPRa"),
    "chen":               ("ivcbench.data.loaders.chen", "load", {}, "KO"),
}
FRACS = [(0.10, "10"), (0.25, "25"), (0.50, "50")]


def _load(modpath, fn, kw):
    import importlib
    return getattr(importlib.import_module(modpath), fn)(**kw)


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def run_dataset(ds_name, cs, gene2go) -> list[dict]:
    """Compute nearest-gene-coexpr and nearest-gene-go rows for all 3 splits of one dataset."""
    obs = cs.obs
    X = cs.X
    var = list(cs.var_names)
    var_pos = {g: i for i, g in enumerate(var)}
    is_ctrl_all = obs["is_control"].to_numpy()
    pert_all = obs["perturbation"].to_numpy()
    genes_perturbed = sorted(set(cs.uns["genes_perturbed"]))

    rows = []
    for frac, lbl in FRACS:
        held = c3.held_gene_fraction(genes_perturbed, frac, seed=0)
        spec = c3.true_lo_gene(held, lbl)
        split = build_split(cs, spec)
        train_idx = split.train_idx
        test_idx = split.test_idx
        test_strata = split.test_strata
        # downstream-only: exclude the perturbed target gene (per stratum) — matches run_job using
        # exclude_genes=held; pearson_delta excludes those column indices globally.
        excl = cs.gene_index(held) if held else None

        # control baseline = mean of inference-input control cells (matched-context controls, = all
        # NT controls outside the held group) — identical to BaselineAdapter._control_mean.
        inf = split.inference_input_idx
        control_mean = X[inf].mean(0) if len(inf) else X[train_idx][is_ctrl_all[train_idx]].mean(0)
        # training control mean used to define each TRAINING gene's observed effect Δ_train(g*)
        train_ctrl_mask = is_ctrl_all[train_idx]
        control_mean_train = X[train_idx][train_ctrl_mask].mean(0)

        # observed TRAINING effect for every training-visible perturbed gene g* (mean treated − train ctrl)
        train_treated = train_idx[~train_ctrl_mask]
        train_genes = sorted(set(pert_all[train_treated]) - {CONTROL_TOKEN})
        eff_train = {}                     # g* -> Δ vector (n_genes,)
        treated_mean = {}                  # g* -> mean treated profile
        for g in train_genes:
            m = train_treated[pert_all[train_treated] == g]
            if len(m) == 0:
                continue
            tm = X[m].mean(0)
            treated_mean[g] = tm
            eff_train[g] = tm - control_mean_train
        # union cell-mean fallback (= the cell-mean floor's treated mean over all training treated)
        union_treated_mean = X[train_treated].mean(0)
        union_eff = union_treated_mean - control_mean_train

        # ---- co-expression NN map: held gene -> nearest TRAINING gene by control-cell co-expression ----
        # Co-expression graph computed on CONTROL cells only (never any perturbation effect → leak-safe).
        ctrl_idx_all = np.where(is_ctrl_all)[0]
        Xc = X[ctrl_idx_all]               # (n_ctrl, n_genes)
        # standardize columns (genes) for Pearson corr between gene feature-vectors
        Xcz = Xc - Xc.mean(0, keepdims=True)
        norms = np.linalg.norm(Xcz, axis=0)
        train_gene_cols = [g for g in train_genes if g in var_pos]
        coexpr_nn = {}
        for hg in held:
            if hg not in var_pos or norms[var_pos[hg]] < 1e-12:
                coexpr_nn[hg] = None        # not a measured gene (or constant) -> fallback to union
                continue
            hv = Xcz[:, var_pos[hg]]
            hn = norms[var_pos[hg]]
            best_g, best_r = None, -np.inf
            for g in train_gene_cols:
                gn = norms[var_pos[g]]
                if gn < 1e-12:
                    continue
                r = float(np.dot(hv, Xcz[:, var_pos[g]]) / (hn * gn))
                if r > best_r:
                    best_r, best_g = r, g
            coexpr_nn[hg] = best_g

        # ---- GO-Jaccard NN map: held gene -> nearest TRAINING gene by GO-term Jaccard ----
        go_nn = {}
        for hg in held:
            ga = gene2go.get(hg, set())
            best_g, best_j = None, -1.0
            for g in train_genes:
                j = jaccard(ga, gene2go.get(g, set()))
                if j > best_j:
                    best_j, best_g = j, g
            go_nn[hg] = best_g if (best_g is not None and best_j > 0) else None

        # ---- assemble predicted cells for each NN variant + a cell-mean anchor ----
        n_test = len(test_idx)
        n_genes = X.shape[1]

        def predict_with(nn_map, label):
            pred = np.empty((n_test, n_genes), dtype=np.float32)
            n_fallback = 0
            for s in np.unique(test_strata):
                rows_m = test_strata == s
                # stratum label is "perturbation=<gene>"
                hg = str(s).split("=", 1)[-1]
                g_star = nn_map.get(hg)
                if g_star is not None and g_star in eff_train:
                    profile = control_mean + eff_train[g_star]
                else:
                    profile = control_mean + union_eff           # fallback = cell-mean floor
                    n_fallback += 1
                pred[rows_m] = profile
            resp = pearson_delta(pred, X[test_idx], control_mean, test_strata, excl)
            resp_ci = bootstrap_ci(list(resp["per_stratum"].values()), seed=0)
            return resp, resp_ci, n_fallback

        # cell-mean anchor (should reproduce deposited cell-mean within fp tolerance)
        anchor_pred = np.tile(control_mean + union_eff, (n_test, 1))
        anchor_resp = pearson_delta(anchor_pred, X[test_idx], control_mean, test_strata, excl)

        for variant, nn_map in [("nearest-gene-coexpr", coexpr_nn), ("nearest-gene-go", go_nn)]:
            resp, resp_ci, n_fb = predict_with(nn_map, variant)
            rows.append(dict(
                baseline=variant, family="simple-repr", split=spec.name,
                registry_task="C3_LO_gene", action="run_floor", headline_eligible=False, seed=0,
                ran=True, leak_free=True, n_train=int(len(train_idx)), n_test=int(n_test),
                n_test_strata=int(len(np.unique(test_strata))),
                pearson_delta=resp["macro"], pearson_delta_lo=resp_ci["lo"],
                pearson_delta_hi=resp_ci["hi"], pearson_delta_ontarget=float("nan"),
                cluster="C3", dataset=ds_name, modality=DATASETS[ds_name][3],
                n_fallback=int(n_fb), n_held=int(len(held)),
                anchor_cellmean_pd=anchor_resp["macro"],
            ))
            print(f"  [{ds_name} {lbl}%] {variant}: Pearson-Δ={resp['macro']:.4f} "
                  f"(fallback {n_fb}/{len(held)}); cell-mean anchor={anchor_resp['macro']:.4f}",
                  flush=True)
    return rows


def main():
    with open(GENE2GO_PKL, "rb") as f:
        gene2go = pickle.load(f)
    all_rows = []
    for ds_name, (modpath, fn, kw, _mod) in DATASETS.items():
        print(f"[load] {ds_name} ...", flush=True)
        cs = _load(modpath, fn, kw)
        print(f"[run]  {ds_name}: {cs.n_cells} cells x {cs.n_genes} genes, "
              f"{len(cs.uns['genes_perturbed'])} perturbed genes", flush=True)
        all_rows += run_dataset(ds_name, cs, gene2go)
    df = pd.DataFrame(all_rows)
    out = ROOT / "results/C3/nearest_gene_baseline.csv"
    df.to_csv(out, index=False)
    print(f"\n[done] wrote {len(df)} rows -> {out}")
    # anchor check
    anchor_ok = True
    dep = pd.read_csv(ROOT / "results/C3/results_raw.csv")
    dep = dep[(dep.baseline == "cell-mean") & dep.ran]
    diffs = []
    for _, r in df.iterrows():
        m = dep[(dep.dataset == r.dataset) & (dep.split == r.split)]
        if len(m):
            d = abs(float(m.iloc[0]["pearson_delta"]) - float(r["anchor_cellmean_pd"]))
            diffs.append(d)
    if diffs:
        mx = max(diffs)
        print(f"[anchor] max |cell-mean(this harness) − deposited cell-mean| = {mx:.2e} "
              f"over {len(diffs)} (dataset,split) -> {'PASS' if mx < 1e-6 else 'CHECK'}")
    print(json.dumps({"n_rows": len(df), "variants": sorted(df.baseline.unique())}))


if __name__ == "__main__":
    main()
