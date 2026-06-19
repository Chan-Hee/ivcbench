"""Prediction-bundle I/O — the deposited model-output layer for GPU-free predictions->metrics reproduction.

One bundle per (cluster x model x split) stores the EXACT arrays a scorer fed to pearson_delta / e_distance,
so `scripts/reproduce_eval.py` recomputes both axes identically with no raw data, checkpoints, or GPU. A single
writer (here) is shared by the runner and every model script, so all emitters produce an identical format.
"""
from __future__ import annotations

import os

import numpy as np

from ..metrics.response import pearson_delta
from ..metrics.distribution import e_distance


def save_bundle(path, *, pred_cells=None, test_cells=None, cell_strata=None, control_mean,
                genes, exclude_gene_idx=None, fit_on=None, n_pca=50,
                pred_means=None, obs_means=None, strata=None, cluster="", model="", split="", **uns):
    """Write one prediction bundle to an explicit path. Provide EITHER per-cell arrays
    (pred_cells/test_cells/cell_strata -> both metrics reproduce) OR per-stratum means
    (pred_means/obs_means/strata -> Pearson-Δ only). If `fit_on` (the training cells used for the
    e_distance PCA basis) is given, its PCA-50 basis is stored so energy distance reproduces exactly."""
    b = dict(control_mean=np.asarray(control_mean, np.float32), genes=np.asarray(genes, object),
             cluster=str(cluster), model=str(model), split=str(split), **uns)
    if pred_cells is not None:
        b["pred_cells"] = np.asarray(pred_cells, np.float32)
        b["test_cells"] = np.asarray(test_cells, np.float32)
        b["cell_strata"] = np.asarray(cell_strata, object)
    if pred_means is not None:
        b["pred_means"] = np.asarray(pred_means, np.float32)
        b["obs_means"] = np.asarray(obs_means, np.float32)
        b["strata"] = np.asarray(strata if strata is not None else np.arange(len(pred_means)), object)
    if exclude_gene_idx is not None:
        b["exclude_gene_idx"] = np.asarray(exclude_gene_idx, int)
    if fit_on is not None:
        from sklearn.decomposition import PCA
        tr = np.asarray(fit_on, np.float64)
        k = int(min(n_pca, tr.shape[0] - 1, tr.shape[1]))
        pca = PCA(n_components=max(2, k), random_state=0).fit(tr)
        b["pca_components"] = pca.components_.astype(np.float32)
        b["pca_mean"] = pca.mean_.astype(np.float32)
    np.savez_compressed(path, **b)
    return path


def dump_bundle(out_dir, *, cluster, model, split, **kw):
    """Env/dir convenience wrapper: write <out_dir>/<cluster>__<model>__<split>.npz if out_dir is set, else
    no-op. NEVER raises — prediction-dumping must not abort a scored run. Pass IVCBENCH_PRED_DUMP as out_dir."""
    if not out_dir:
        return None
    try:
        os.makedirs(out_dir, exist_ok=True)
        fn = f"{cluster}__{model}__{split}.npz".replace("/", "_")
        return save_bundle(os.path.join(out_dir, fn), cluster=cluster, model=model, split=split, **kw)
    except Exception:  # noqa: BLE001 — never let prediction-dumping break the run
        return None


def score_bundle(path):
    """Reproduce a bundle's scores with the frozen metric code. Per-cell -> Pearson-Δ + energy distance
    (exact if the train PCA basis is stored); per-stratum means -> Pearson-Δ only."""
    d = np.load(path, allow_pickle=True)
    ctrl = np.asarray(d["control_mean"], np.float64)
    excl = np.asarray(d["exclude_gene_idx"], int) if "exclude_gene_idx" in d.files else None
    meta = {k: (d[k].item() if d[k].shape == () else d[k]) for k in ("cluster", "model", "split") if k in d.files}
    if "pred_cells" in d.files:
        pred = np.asarray(d["pred_cells"], np.float64); obs = np.asarray(d["test_cells"], np.float64)
        sid = np.asarray(d["cell_strata"]); n_strata = len(np.unique(sid))
        pr = pearson_delta(pred, obs, ctrl, sid, exclude_genes=excl)
        if "pca_components" in d.files:
            from scipy.spatial.distance import cdist
            comp = np.asarray(d["pca_components"], np.float64); pm = np.asarray(d["pca_mean"], np.float64)
            P = (pred - pm) @ comp.T; T = (obs - pm) @ comp.T
            en = lambda a, b: 2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean()
            per = [en(P[sid == s], T[sid == s]) for s in np.unique(sid) if (sid == s).sum() >= 2]
            ed = float(np.mean(per)) if per else float("nan")
        else:
            ed = e_distance(pred, obs, sid, fit_on=obs)["macro"]
    else:
        pred = np.asarray(d["pred_means"], np.float64); obs = np.asarray(d["obs_means"], np.float64)
        sid = np.arange(obs.shape[0]); n_strata = obs.shape[0]
        pr = pearson_delta(pred, obs, ctrl, sid, exclude_genes=excl)
        ed = float("nan")
    return {**{k: str(meta.get(k, "")) for k in ("cluster", "model", "split")},
            "n_test_strata": int(n_strata), "pearson_delta": pr["macro"], "e_distance": ed}
