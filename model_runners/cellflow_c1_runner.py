#!/usr/bin/env python
"""CellFlow runner for the held-GROUP tasks — `ivc-cellflow` env.

Invoked by ivcbench.baselines.heavy.CellFlowC1:
    <ivc-cellflow python> cellflow_c1_runner.py <in.npz> <out.npz>

Covers T1 (Kang IFN-beta, held CELL TYPE) and T2 (Soskic CD4 activation, held DONOR). Both have the
same structure and it is CellFlow's own published PBMC experiment: ONE stimulus, seen in training,
and a held *group* whose stimulated cells are hidden. CellFlow's recipe for that shape is

  * the stimulus is a perturbation covariate -- here a single CATEGORICAL column (`condition`,
    values {control, <stim>}). No `perturbation_covariate_reps` is passed, so CellFlow fits its own
    `OneHotEncoder` over that column (`DataManager._get_primary_covar_encoder`); that is the released
    API's native handling of a categorical perturbation and needs no external representation.
  * the held group is a SPLIT covariate (`split_covariates=[<group column>]`), exactly as `donor` is
    in the 100_pbmc tutorial. A split covariate defines which control cells are the SOURCE of the
    flow; it is never embedded into the condition. That is why the held group may be a value CellFlow
    has never seen: at inference the held unit's own control cells simply become their own source
    population, while the condition vector is the (seen) stimulus one-hot.

    Registering the group as a `sample_covariate` instead would be WRONG here and CellFlow says so:
    sample covariates ARE embedded (`_get_sample_covariates_embedding`), and an unseen value raises
    "Representation for '<group>' not found". Split covariate is the construct for a held unit.

Leak-safe: fit on `X_train` only. The held unit's stimulated cells never enter training; the unit
contributes only its control cells, through `X_ctrl_inf`, which is what the split declares as
inference input (`control_inference_only=True` on both C1_LOCT and C2_LODO).

ONE PROFILE FOR THE HELD GROUP: the perturbation axis has a single seen value, so the runner emits
one predicted stimulated profile, keyed by the stimulus label. The adapter therefore declares
`pred_key_is_group = True`.

Cells live in a PCA space fitted on the TRAIN fold only (`centered_pca`), generated cells are decoded
back to the HVG panel with the same basis (`reconstruct_pca`), and the mean of the decoded cells is
returned -- the profile the harness scores with Pearson-delta.

Knobs (env): IVCBENCH_CELLFLOW_{ITERS,BATCH,NPCA,MAXCELLS,NCTRL,COND_DIM,HIDDEN} (shared with the
C5 runner; see model_runners/cellflow_common.py).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cellflow_common import (
    decode_mean,
    knobs,
    prepare_model,
    stratified_cap,
)  # noqa: E402

GROUP_COL = "group"  # obs column the runner writes the held-axis label into
COND_COL = "condition"  # obs column carrying {control, <stim>}
COND_GROUP = "stim"  # CellFlow perturbation-covariate group name


def _held_axis(d, held_label: str):
    """Which harness axis is the held one -> (train labels, inference labels, axis name).

    `SubprocessAdapter._build_payload` emits `celltype_train/celltype_inf` from `cell_type_coarse`
    and `gem_train/gem_inf` from `donor_id` whenever those columns exist, plus `held_label`. On T1
    the held axis is the lineage, on T2 the donor; the held axis is the one whose INFERENCE labels
    are all equal to `held_label` (the split hands the model only the held unit's control cells).
    """
    cands = [
        ("cell_type_coarse", "celltype_train", "celltype_inf"),
        ("donor_id", "gem_train", "gem_inf"),
    ]
    for axis, ktr, kinf in cands:
        if ktr not in d.files or kinf not in d.files:
            continue
        inf = np.asarray([str(x) for x in d[kinf]])
        if len(inf) and (inf == held_label).all():
            return np.asarray([str(x) for x in d[ktr]]), inf, axis
    return None, None, None


def build_inputs(in_path: str, centered_pca, project_pca):
    """Payload -> (train AnnData in PCA space, control-source AnnData, stim label, group col, genes).

    Split out of `main` so the data contract can be exercised without importing CellFlow.
    """
    import anndata as ad
    import pandas as pd

    k = knobs()
    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    pert_train = np.asarray([str(p) for p in d["pert_train"]])
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if len(test_perts) != 1:
        raise RuntimeError(
            "CellFlow-C1: this entry point expects ONE seen stimulus, "
            f"payload has {len(test_perts)}: {test_perts[:5]}"
        )
    stim = test_perts[0]
    if not is_ctrl.any() or not (~is_ctrl).any():
        raise RuntimeError(
            "CellFlow-C1: training fold needs both control and stimulated cells"
        )

    held_label = str(d["held_label"]) if "held_label" in d.files else ""
    g_train, _g_inf, axis = _held_axis(d, held_label)
    if g_train is None:
        # no usable group structure in the payload -> one global control pool (documented fallback)
        g_train, axis = np.array(["all"] * X.shape[0]), None
        held_label = held_label or "held"

    cond = np.where(is_ctrl, "control", pert_train).astype(str)

    # ---- split covariate hygiene ---------------------------------------------------------------
    # CellFlow requires every SOURCE split value to have a matching TARGET split value
    # (`DataManager._verify_split_covariates`). A training group that contributes only control cells
    # (or only stimulated cells) cannot form a source->target pair, so its cells are dropped from the
    # training fold rather than silently breaking setup. If fewer than two groups survive, the group
    # structure carries no information and the split covariate is dropped entirely.
    use_split = axis is not None
    if use_split:
        ok_groups = {
            g
            for g in np.unique(g_train)
            if (cond[g_train == g] == "control").any()
            and (cond[g_train == g] != "control").any()
        }
        dropped = sorted(set(np.unique(g_train)) - ok_groups)
        if len(ok_groups) < 2:
            use_split, dropped = False, []
        else:
            keep = np.array([g in ok_groups for g in g_train])
            X, cond, g_train, is_ctrl = (
                X[keep],
                cond[keep],
                g_train[keep],
                is_ctrl[keep],
            )
        if dropped:
            print(
                f"[CellFlow-C1] dropped {len(dropped)} training group(s) lacking a "
                f"control/stimulated pair: {dropped[:6]}",
                flush=True,
            )
    if not use_split:
        g_train = np.array(["all"] * X.shape[0])
        held_group_value = "all"
        print(
            "[CellFlow-C1] no usable split covariate -> single global control"
            " population",
            flush=True,
        )
    else:
        held_group_value = held_label

    # ---- stratified cell cap -------------------------------------------------------------------
    sel = stratified_cap(
        X, np.char.add(np.asarray(cond, str), np.asarray(g_train, str)), k["max_cells"]
    )
    X, cond, g_train = X[sel], cond[sel], g_train[sel]
    if X_ctrl_inf.shape[0] > k["n_ctrl"]:
        X_ctrl_inf = X_ctrl_inf[
            np.random.default_rng(0).choice(
                X_ctrl_inf.shape[0], k["n_ctrl"], replace=False
            )
        ]

    # ---- training AnnData ----------------------------------------------------------------------
    adata_tr = ad.AnnData(
        X=np.asarray(X, dtype=np.float32),
        obs=pd.DataFrame(
            {
                COND_COL: np.asarray(cond, str),
                GROUP_COL: np.asarray(g_train, str),
                "is_control": np.asarray(cond, str) == "control",
            },
            index=[f"tr{i}" for i in range(X.shape[0])],
        ),
    )
    adata_tr.var_names = genes
    adata_tr.obs["is_control"] = adata_tr.obs["is_control"].astype("boolean")

    # ---- the held unit's OWN control cells: the source distribution at inference ----------------
    adata_ct = ad.AnnData(
        X=np.asarray(X_ctrl_inf, dtype=np.float32),
        obs=pd.DataFrame(
            {
                COND_COL: ["control"] * X_ctrl_inf.shape[0],
                GROUP_COL: [held_group_value] * X_ctrl_inf.shape[0],
                "is_control": True,
            },
            index=[f"ct{i}" for i in range(X_ctrl_inf.shape[0])],
        ),
    )
    adata_ct.var_names = genes
    adata_ct.obs["is_control"] = adata_ct.obs["is_control"].astype("boolean")

    # ---- PCA space fitted on the TRAIN fold only (leak-safe), authors' own helpers --------------
    n_pca = int(min(k["n_pca"], adata_tr.n_obs - 1, adata_tr.n_vars - 1))
    centered_pca(adata_tr, n_comps=n_pca, method="scanpy", keep_centered_data=False)
    project_pca(query_adata=adata_ct, ref_adata=adata_tr)

    print(
        f"[CellFlow-C1] train={adata_tr.n_obs} cells x {adata_tr.n_vars} genes |"
        f" stimulus='{stim}' | held axis={axis} value='{held_group_value}' | train"
        f" groups={len(set(map(str, g_train)))} |"
        f" split_cov={'yes' if use_split else 'no'} | ctrl-source={adata_ct.n_obs} |"
        f" PCA={n_pca}",
        flush=True,
    )
    return (
        adata_tr,
        adata_ct,
        stim,
        (GROUP_COL if use_split else None),
        held_group_value,
        genes,
    )


def main(in_path: str, out_path: str) -> None:
    import pandas as pd
    from cellflow.model import CellFlow
    from cellflow.preprocessing import centered_pca, project_pca

    k = knobs()
    adata_tr, adata_ct, stim, split_col, held_value, genes = build_inputs(
        in_path, centered_pca, project_pca
    )
    print(
        f"[CellFlow-C1] iters={k['iters']} batch={k['batch']} cond_dim={k['cond_dim']} "
        f"hidden={k['hidden']}",
        flush=True,
    )

    cf = CellFlow(adata_tr, solver="otfm")
    cf.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        # single categorical stimulus, no external representation -> CellFlow's own one-hot encoder
        perturbation_covariates={COND_GROUP: (COND_COL,)},
        split_covariates=[split_col] if split_col else None,
        max_combination_length=1,
        null_value=0.0,
    )
    prepare_model(cf, k["cond_dim"], k["hidden"], COND_GROUP)
    cf.train(num_iterations=k["iters"], batch_size=k["batch"])

    # covariate_data must carry every registered covariate (perturbation + split) AND the control key
    # column, because DataManager._get_condition_data indexes covariate_data[... + [control_key]].
    cov = {COND_COL: [stim], "condition_id": [stim], "is_control": [True]}
    if split_col:
        cov[split_col] = [held_value]
    cov = pd.DataFrame(cov)
    cov["is_control"] = cov["is_control"].astype("boolean")
    preds = cf.predict(
        adata=adata_ct,
        covariate_data=cov,
        sample_rep="X_pca",
        condition_id_key="condition_id",
    )

    if stim not in preds:
        raise RuntimeError(
            f"CellFlow-C1: predict() returned no profile for '{stim}' "
            f"(keys: {sorted(preds)[:5]})"
        )
    profile = decode_mean(preds[stim], genes, adata_tr)

    np.savez(
        out_path,
        pred_perts=np.array([stim], dtype=object),
        pred_means=profile[None, :].astype(np.float32),
    )
    print(
        f"[CellFlow-C1] wrote 1 predicted profile for '{stim}' -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
