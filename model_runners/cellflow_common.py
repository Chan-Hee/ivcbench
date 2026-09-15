#!/usr/bin/env python
"""Shared CellFlow model setup for the ivcbench CellFlow runners — `ivc-cellflow` env.

`cellflow_c5_runner.py` (the first CellFlow entry point onboarded here) builds its solver inline
inside `main()`, so there is no importable function to reuse. This module lifts that block
VERBATIM — the same `otfm` solver, the same `prepare_model` keyword set, the same
`probability_path`/`flow` version probe, the same PCA encode/decode helpers — so the held-group
(T1/T2) and held-gene (T3/T4) entry points train an identically configured model and only the
*condition* and the *held axis* differ between tasks. Any change here changes every CellFlow cell.

Nothing in this file is task-specific; the task lives in the caller's `prepare_data(...)` call.
"""
from __future__ import annotations

import functools
import os

import numpy as np

# JAX must not preallocate the whole card: these runners share their GPU with other jobs.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


def knobs() -> dict:
    """The env-tunable knobs, identical names/defaults to cellflow_c5_runner.py."""
    return dict(
        iters=int(os.environ.get("IVCBENCH_CELLFLOW_ITERS", "30000")),
        batch=int(os.environ.get("IVCBENCH_CELLFLOW_BATCH", "1024")),
        cond_dim=int(os.environ.get("IVCBENCH_CELLFLOW_COND_DIM", "64")),
        hidden=int(os.environ.get("IVCBENCH_CELLFLOW_HIDDEN", "1024")),
        n_pca=int(os.environ.get("IVCBENCH_CELLFLOW_NPCA", "50")),
        max_cells=int(os.environ.get("IVCBENCH_CELLFLOW_MAXCELLS", "60000")),
        n_ctrl=int(os.environ.get("IVCBENCH_CELLFLOW_NCTRL", "2000")),
    )


def prepare_model(cf, cond_dim: int, hidden: int, cond_group: str) -> None:
    """`cellflow_c5_runner.py`'s prepare_model block, with the condition group name parameterised.

    `layers_before_pool` is keyed by the perturbation-covariate group, which is "drug" on the
    compound cluster and "stim"/"gene" here; everything else is byte-identical to the C5 setup.
    """
    from cellflow.utils import match_linear

    model_kw = dict(
        condition_mode="deterministic",
        regularization=0.0,
        pooling="mean",
        layers_before_pool={
            cond_group: {"layer_type": "mlp", "dims": [256, 256], "dropout_rate": 0.0}
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


def decode_mean(Z, genes, adata_tr):
    """Decode generated PCA-space cells back to the gene panel with the TRAIN basis -> mean profile.

    Same three lines as cellflow_c5_runner.py's decode loop.
    """
    import anndata as ad
    from cellflow.preprocessing import reconstruct_pca

    Z = np.squeeze(np.asarray(Z, dtype=np.float32))
    if Z.ndim == 1:
        Z = Z[None, :]
    a = ad.AnnData(X=np.zeros((Z.shape[0], len(genes)), dtype=np.float32))
    a.var_names = genes
    a.obsm["X_pca"] = Z
    reconstruct_pca(a, use_rep="X_pca", ref_adata=adata_tr)
    return np.asarray(a.layers["X_recon"], dtype=np.float32).mean(0)


def stratified_cap(X, labels, max_cells: int, seed: int = 0):
    """cellflow_c5_runner.py's stratified cell cap, returned as an index vector."""
    n = X.shape[0]
    if n <= max_cells:
        return np.arange(n)
    rng0 = np.random.default_rng(seed)
    keep = []
    for lab, cnt in zip(*np.unique(labels, return_counts=True)):
        idx = np.where(labels == lab)[0]
        take = max(1, int(round(max_cells * cnt / n)))
        keep.append(idx if take >= cnt else rng0.choice(idx, take, replace=False))
    return np.sort(np.concatenate(keep))
