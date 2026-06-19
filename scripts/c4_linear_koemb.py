#!/usr/bin/env python
"""C4 'linear-shift + KO-embedding' conditioned baseline (reviewer request 11), evaluated DIRECTLY
(bypassing the applicability registry, which only knows the 13 named baselines) but with the IDENTICAL
leak-safe split + audit + Pearson-Δ(downstream-only) + E-distance metric calls the framework uses.
Genuinely conditioned: each held KO gets a DIFFERENT predicted gene-space shift from its leak-safe
gene-side embedding (control-only PCA loadings), vs cell-mean's single global shift.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "src")
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from ivcbench.data.loaders.frangieh import load
from ivcbench.clusters import c4
from ivcbench.splits.builder import build_split
from ivcbench.splits.audit import audit_split
from ivcbench.metrics.response import pearson_delta
from ivcbench.metrics.distribution import e_distance
from ivcbench.metrics.stats import bootstrap_ci


def gene_index(cs, names):
    pos = {g: i for i, g in enumerate(cs.var_names)}
    return np.array([pos[n] for n in names if n in pos], dtype=int)


for modality, mtag in [("rna", "RNA"), ("protein", "protein-CITE")]:
    cs = load(modality=modality)
    g = cs.uns["genes_perturbed"]
    for frac, lbl in [(0.25, "25"), (0.50, "50")]:
        held = c4.held_ko_fraction(g, frac, seed=0)
        spec = c4.modality_lo_ko(held, lbl)
        sp = build_split(cs, spec)
        audit = audit_split(cs, sp)          # hard leak gate
        tr = sp.train_idx
        obs = cs.obs.iloc[tr]
        is_ctrl = obs["is_control"].to_numpy()
        pert = obs["perturbation"].to_numpy().astype(str)
        genes = list(cs.var_names); gpos = {gg: i for i, gg in enumerate(genes)}
        Xtr = cs.X[tr]
        ctrl_X = Xtr[is_ctrl]
        k = int(min(50, ctrl_X.shape[0] - 1, ctrl_X.shape[1]))
        gpca = PCA(n_components=max(2, k), random_state=0).fit(ctrl_X)
        gene_emb = gpca.components_.T
        ctrl_inf = cs.X[sp.inference_input_idx].mean(0) if len(sp.inference_input_idx) else ctrl_X.mean(0)
        ctrl_mean_tr = ctrl_X.mean(0)
        train_kos = [gg for gg in np.unique(pert[~is_ctrl]) if gg in gpos]
        D, E = [], []
        for gg in train_kos:
            m = (pert == gg) & (~is_ctrl)
            if m.sum() == 0:
                continue
            D.append(Xtr[m].mean(0) - ctrl_mean_tr)
            E.append(gene_emb[gpos[gg]])
        reg = Ridge(alpha=1.0).fit(np.vstack(E), np.vstack(D))
        # predict per test cell from its KO embedding
        test_perts = cs.obs.iloc[sp.test_idx]["perturbation"].to_numpy().astype(str)
        preds = []
        for p in test_perts:
            if p in gpos:
                preds.append(ctrl_inf + reg.predict(gene_emb[gpos[p]][None, :])[0])
            else:
                preds.append(ctrl_inf)
        pred = np.vstack(preds)
        test_X = cs.X[sp.test_idx]
        excl = gene_index(cs, list(spec.held_values))           # downstream-only
        resp = pearson_delta(pred, test_X, ctrl_inf, sp.test_strata, excl)
        resp_incl = pearson_delta(pred, test_X, ctrl_inf, sp.test_strata, None)
        dist = e_distance(pred, test_X, sp.test_strata, fit_on=Xtr)
        ci = bootstrap_ci(list(resp["per_stratum"].values()), seed=0)
        print(f"linear-shift-KOemb {mtag:12s} {lbl}%: leak_free={audit['leak_free']} "
              f"pearson_delta(ds-only)={resp['macro']:.4f} [{ci['lo']:.4f},{ci['hi']:.4f}] "
              f"ontarget={resp_incl['macro']:.4f} e_dist={dist['macro']:.4f} "
              f"(n_train={audit['n_train']}, n_test={audit['n_test']}, n_strata={audit['n_test_strata']})")
