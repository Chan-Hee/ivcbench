#!/usr/bin/env python
"""CellFlow runner (OP3 / C5) — `ivc-cellflow` env.

Invoked by an ivcbench SubprocessAdapter:
    <ivc-cellflow python> cellflow_c5_runner.py <in.npz> <out.npz>

CellFlow (Klein et al., bioRxiv 2025.04.11.648220; theislab/cellflow) learns a CONDITIONAL
FLOW from the unperturbed cell distribution to the perturbed one, with the perturbation entering
through a learned condition embedding of a perturbation *representation* rather than through a
label. One script serves BOTH OP3 entry points because the published interface is identical in
both; only which axis is held out changes:

  * T5u (unseen compound)     — the held compound never appears in training; the model sees only
    its Morgan fingerprint. This is exactly the unseen-drug setting of the CellFlow manuscript and
    of the `500_combosciplex` tutorial (held-out drug conditions, fingerprint reps in `uns`).
  * T5c (cell-context, LOCT)  — every compound is seen; the held lineage enters ONLY as the SOURCE
    distribution (its own DMSO cells at inference). This is the structure of the `100_pbmc`
    tutorial, where a cytokine seen in other donors is transferred onto a held donor's own
    control cells.

Leak-safe: the model is fit on `X_train` only (the framework's train fold). The held lineage's
treated cells / the held compound's cells are never seen; the held unit contributes only control
cells, via `X_ctrl_inf`, which is what the split declares as inference input.

Cells live in a PCA space fitted on the TRAIN fold only (`cellflow.preprocessing.centered_pca`,
the authors' own helper), generated cells are decoded back to the HVG gene panel with the same
basis (`reconstruct_pca`), and the per-compound mean of the decoded cells is returned — the
profile the harness scores with Pearson-delta.

Knobs (env): IVCBENCH_CELLFLOW_{ITERS,BATCH,NPCA,MAXCELLS,NCTRL,COND_DIM,HIDDEN}.

Licensing note: the GitHub repo ships an MIT LICENSE file, but the released distribution
(PyPI `cellflow-tools` 0.0.9) declares PolyForm-Noncommercial-1.0.0. Academic benchmarking is
covered either way; do not redistribute the package as if it were MIT.
"""
from __future__ import annotations

import functools
import os
import sys

import numpy as np

# JAX must not preallocate the whole card: this runner shares its GPU with other onboarding jobs.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


def build_inputs(in_path: str, centered_pca, project_pca):
    """Payload -> (train AnnData in PCA space, control-source AnnData, held compounds, genes).

    Split out of `main` so the data contract can be exercised without importing CellFlow (the
    preflight in outputs/onboarding/cellflow/ passes numpy equivalents of the two PCA helpers).
    """
    import anndata as ad
    import pandas as pd

    n_pca = int(os.environ.get("IVCBENCH_CELLFLOW_NPCA", "50"))
    max_cells = int(os.environ.get("IVCBENCH_CELLFLOW_MAXCELLS", "60000"))
    n_ctrl = int(os.environ.get("IVCBENCH_CELLFLOW_NCTRL", "2000"))

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if "fingerprint_keys" not in d.files:
        raise RuntimeError(
            "CellFlow: payload has no fingerprint_* — CellFlow conditions on a "
            "compound representation, so side_info['fingerprint'] is required."
        )
    fp = {
        str(k): np.asarray(v, dtype=np.float32).ravel()
        for k, v in zip(d["fingerprint_keys"], d["fingerprint_vals"])
    }
    fp_dim = len(next(iter(fp.values())))

    pred_cpds = [c for c in test_perts if c in fp]
    if not pred_cpds:
        raise RuntimeError(
            f"CellFlow: none of the {len(test_perts)} held labels has a fingerprint"
        )
    if not is_ctrl.any():
        raise RuntimeError("CellFlow: training fold has no control cells to flow from")

    # ---- stratified cell cap (OP3 control pools are large) -------------------------------------
    if X.shape[0] > max_cells:
        rng0 = np.random.default_rng(0)
        labels = np.where(is_ctrl, "control", pert_train).astype(str)
        keep = []
        for lab, cnt in zip(*np.unique(labels, return_counts=True)):
            idx = np.where(labels == lab)[0]
            take = max(1, int(round(max_cells * cnt / X.shape[0])))
            keep.append(idx if take >= cnt else rng0.choice(idx, take, replace=False))
        sel = np.sort(np.concatenate(keep))
        X, pert_train, is_ctrl = X[sel], pert_train[sel], is_ctrl[sel]
    if X_ctrl_inf.shape[0] > n_ctrl:
        X_ctrl_inf = X_ctrl_inf[
            np.random.default_rng(0).choice(X_ctrl_inf.shape[0], n_ctrl, replace=False)
        ]

    # ---- training AnnData ----------------------------------------------------------------------
    compound = np.where(is_ctrl, "control", pert_train).astype(str)
    # a training compound with no fingerprint cannot be encoded -> drop it (never a target here)
    ok = np.array([c == "control" or c in fp for c in compound])
    X, compound = X[ok], compound[ok]
    adata_tr = ad.AnnData(
        X=np.asarray(X, dtype=np.float32),
        obs=pd.DataFrame(
            {"compound": compound, "is_control": compound == "control"},
            index=[f"tr{i}" for i in range(X.shape[0])],
        ),
    )
    adata_tr.var_names = genes
    adata_tr.obs["is_control"] = adata_tr.obs["is_control"].astype("boolean")
    reps = {c: fp[c] for c in sorted(set(compound) - {"control"})}
    for c in pred_cpds:  # held compounds: representation only, no cells
        reps[c] = fp[c]
    reps["control"] = np.zeros(fp_dim, dtype=np.float32)
    adata_tr.uns["fingerprints"] = reps

    # ---- control AnnData used as the SOURCE distribution at inference ---------------------------
    adata_ct = ad.AnnData(
        X=np.asarray(X_ctrl_inf, dtype=np.float32),
        obs=pd.DataFrame(
            {"compound": ["control"] * X_ctrl_inf.shape[0], "is_control": True},
            index=[f"ct{i}" for i in range(X_ctrl_inf.shape[0])],
        ),
    )
    adata_ct.var_names = genes
    adata_ct.obs["is_control"] = adata_ct.obs["is_control"].astype("boolean")
    adata_ct.uns["fingerprints"] = reps

    # ---- PCA space fitted on the TRAIN fold only (leak-safe), authors' own helpers --------------
    n_pca = int(min(n_pca, adata_tr.n_obs - 1, adata_tr.n_vars - 1))
    centered_pca(adata_tr, n_comps=n_pca, method="scanpy", keep_centered_data=False)
    project_pca(query_adata=adata_ct, ref_adata=adata_tr)

    print(
        f"[CellFlow] train={adata_tr.n_obs} cells x {adata_tr.n_vars} genes | "
        f"train compounds={len(set(compound)) - 1} | predict={len(pred_cpds)} | "
        f"ctrl-source={adata_ct.n_obs} | PCA={n_pca}",
        flush=True,
    )
    return adata_tr, adata_ct, pred_cpds, genes


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import pandas as pd
    from cellflow.model import CellFlow
    from cellflow.preprocessing import centered_pca, project_pca, reconstruct_pca
    from cellflow.utils import match_linear

    iters = int(os.environ.get("IVCBENCH_CELLFLOW_ITERS", "30000"))
    batch = int(os.environ.get("IVCBENCH_CELLFLOW_BATCH", "1024"))
    cond_dim = int(os.environ.get("IVCBENCH_CELLFLOW_COND_DIM", "64"))
    hidden = int(os.environ.get("IVCBENCH_CELLFLOW_HIDDEN", "1024"))

    adata_tr, adata_ct, pred_cpds, genes = build_inputs(
        in_path, centered_pca, project_pca
    )
    print(
        f"[CellFlow] iters={iters} batch={batch} cond_dim={cond_dim} hidden={hidden}",
        flush=True,
    )

    # ---- CellFlow ------------------------------------------------------------------------------
    cf = CellFlow(adata_tr, solver="otfm")
    cf.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        perturbation_covariates={"drug": ("compound",)},
        perturbation_covariate_reps={"drug": "fingerprints"},
        max_combination_length=1,
        null_value=0.0,
    )
    model_kw = dict(
        condition_mode="deterministic",
        regularization=0.0,
        pooling="mean",
        layers_before_pool={
            "drug": {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0}
        },
        layers_after_pool={
            "layer_type": "mlp",
            "dims": [256, 256],
            "dropout_rate": 0.0,
        },
        condition_embedding_dim=cond_dim,
        cond_output_dropout=0.9,
        hidden_dims=[hidden, hidden, hidden],
        decoder_dims=[2 * hidden, 2 * hidden, 2 * hidden],
        conditioning="concatenation",
        match_fn=functools.partial(match_linear, epsilon=1.0, tau_a=1.0, tau_b=1.0),
        linear_projection_before_concatenation=True,
    )
    # the noise on the probability path is `probability_path=` in recent releases and `flow=` in
    # older ones; pick whichever this installation exposes rather than pinning a version
    import inspect

    sig = inspect.signature(cf.prepare_model).parameters
    noise = {"constant_noise": 0.5}
    model_kw["probability_path" if "probability_path" in sig else "flow"] = noise
    model_kw = {k: v for k, v in model_kw.items() if k in sig}
    cf.prepare_model(**model_kw)
    cf.train(num_iterations=iters, batch_size=batch)

    # covariate_data must carry every registered covariate AND the control key column
    # (DataManager._get_condition_data indexes covariate_data[... + [control_key]]).
    cov = pd.DataFrame(
        {
            "compound": pred_cpds,
            "condition": pred_cpds,
            "is_control": [True] * len(pred_cpds),
        }
    )
    cov["is_control"] = cov["is_control"].astype("boolean")
    preds = cf.predict(
        adata=adata_ct,
        covariate_data=cov,
        sample_rep="X_pca",
        condition_id_key="condition",
    )

    # ---- decode the generated cells back to the HVG gene panel with the TRAIN basis -------------
    pred_perts, pred_means = [], []
    for c in pred_cpds:
        if c not in preds:
            continue
        Z = np.squeeze(np.asarray(preds[c], dtype=np.float32))
        if Z.ndim == 1:
            Z = Z[None, :]
        a = ad.AnnData(X=np.zeros((Z.shape[0], len(genes)), dtype=np.float32))
        a.var_names = genes
        a.obsm["X_pca"] = Z
        reconstruct_pca(a, use_rep="X_pca", ref_adata=adata_tr)
        pred_perts.append(c)
        pred_means.append(np.asarray(a.layers["X_recon"], dtype=np.float32).mean(0))
    if not pred_perts:
        raise RuntimeError(
            "CellFlow: predict() returned no profile for any held compound"
        )

    np.savez(
        out_path,
        pred_perts=np.array(pred_perts, dtype=object),
        pred_means=np.vstack(pred_means).astype(np.float32),
    )
    print(
        f"[CellFlow] wrote {len(pred_perts)} predicted profiles -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
