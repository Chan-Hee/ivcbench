#!/usr/bin/env python
"""Biolord runner for the COMPOUND cluster (C5 / OP3) — `ctx-biolord` conda env.

Invoked by ivcbench.baselines.heavy.BiolordC5:
    <env python> biolord_c5_runner.py <in.npz> <out.npz>

Why this is NATIVE, not adapted
-------------------------------
biolord (Piran et al., Nat Biotechnol 2024) decomposes a cell into a residual "unknown-attribute"
state plus DISENTANGLED annotated attributes, and generates a cell under changed attribute values.
For small-molecule perturbation the published model conditions on chemistry directly:

  * paper, Methods (sci-Plex 3): "We use RDKit chemically informed features embedding of the drugs,
    as well as the dosage as ordered attributes."  and  "To allow generalization to unseen drugs, we
    take advantage of existing prior knowledge and obtain chemically informed embedding of the drugs
    using RDKit features."  The cell line is passed as a categorical attribute.
  * paper, Methods (unseen-drug counterfactuals): "Specifically, we generate the expression
    prediction for control cells with labels of unseen compounds."
  * official code, nitzanlab/biolord_reproducibility, scripts/biolord/sciplex3/base_experiment_sciplex3.py:
        varying_arg["ordered_attributes_keys"]     = ["rdkit2d_dose"]
        varying_arg["categorical_attributes_keys"] = ["cell_type"]
    and utils/utils_perturbations_sciplex3.py::compute_prediction, which predicts by keeping the
    CONTROL cells' expression and index and swapping in the target drug's attribute row:
        dataset_comb[layer]   = dataset_control[layer]
        dataset_comb["ind_x"] = dataset_control["ind_x"]
        for key in dataset_control:
            if key not in [layer, "ind_x"]:
                dataset_comb[key] = repeat_n(dataset[key][idx, :], n_obs)
        pred, _ = model.module.get_expression(dataset_comb)

This runner is that published recipe, verbatim in structure, with the harness's compound-side
representation (side_info['fingerprint'] -> payload fingerprint_keys/fingerprint_vals: an RDKit
Morgan/ECFP4 1024-bit vector per compound, the same chemistry every other C5 model in this benchmark
consumes) in place of `rdkit2d_dose`.

The descriptor is an INPUT of the published API, not a change to the method: biolord's own
`Biolord.setup_anndata(adata, ordered_attributes_keys=...)` takes any obs/obsm key as an ordered
attribute, so which molecular descriptor fills that slot is the caller's to choose. Using the same
fingerprint as the rest of the C5 roster keeps the chemistry identical across the comparison.

DOSE, measured rather than assumed. GSE279945_sc_counts_processed.h5ad carries three values of
obs['dose_uM'] -- 1.0 (245,580 cells), 14.1 (28,452) and 0.1 (24,055) -- so "OP3 has a single dose"
is false of the file as a whole. It is true of the arm this model conditions on: every one of the
219,247 TREATED cells is at 1.0 uM, and no compound of the 144 appears at more than one dose. The
other two values belong entirely to obs['control'] == True, which in OP3 covers the DMSO vehicle
and the two positive-control compounds (Belinostat at 0.1 uM is one of them, not a treated
condition). The dose component of the published attribute is therefore constant over every
condition the model is asked to distinguish, and is dropped rather than encoded as a constant.

Both C5 splits therefore run through one code path:
  T5c  C5_loct_<lineage>       held LINEAGE, compounds SEEN.  The held lineage is never a training
                               category, so the cell-type CATEGORICAL attribute is omitted and the
                               lineage identity is carried by the held cells' own residual state
                               (identical argument to biolord_c1_runner.py). A per-compound profile
                               is generated for every held-lineage compound -- not one global
                               "treated" profile -- so every stratum gets its own prediction.
  T5u  C5_global_compound_holdout   held COMPOUNDS, never seen in training. Their fingerprints are
                               unseen VALUES of a continuous attribute the model was trained on, so
                               the prediction is defined without any external regression. Every
                               lineage is present in training, so `cell_type` IS registered as the
                               published categorical attribute.
The regime is auto-detected: the categorical cell-type attribute is registered only when every
inference-side lineage is present on the training side (true for T5u, false for T5c).

Leak-safe: only payload["X_train"] enters fitting (split="train"); the inference control cells are
split="ood" and are only ever decoded; held compounds' treated cells are never in the payload.
Knobs: $IVCBENCH_BIOLORD_{EPOCHS,LATENT,MAXCELLS,BATCH,ORDERED_LATENT,ATTR_WIDTH,DECODER_WIDTH}.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import pandas as pd
    import scanpy as sc  # noqa: F401  (biolord expects the scanpy stack present)
    import torch
    from biolord import Biolord

    epochs = int(os.environ.get("IVCBENCH_BIOLORD_EPOCHS", "200"))
    n_latent = int(os.environ.get("IVCBENCH_BIOLORD_LATENT", "32"))
    max_cells = int(
        os.environ.get("IVCBENCH_BIOLORD_MAXCELLS", "60000")
    )  # C5 folds are ~47-51k: no subsampling by default
    batch = int(os.environ.get("IVCBENCH_BIOLORD_BATCH", "256"))
    n_ord = int(
        os.environ.get("IVCBENCH_BIOLORD_ORDERED_LATENT", "256")
    )  # published sciplex3 range
    attr_w = int(os.environ.get("IVCBENCH_BIOLORD_ATTR_WIDTH", "512"))
    dec_w = int(os.environ.get("IVCBENCH_BIOLORD_DECODER_WIDTH", "1024"))
    max_src = int(
        os.environ.get("IVCBENCH_BIOLORD_MAXSRC", "2000")
    )  # control cells decoded per compound

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    pert_train = np.asarray([str(p) for p in d["pert_train"]])
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if not test_perts:
        raise RuntimeError("Biolord-C5: no held compound label in the payload")
    if "fingerprint_keys" not in d.files:
        raise RuntimeError(
            "Biolord-C5: payload has no fingerprint_* — biolord's compound attribute "
            "is the RDKit chemistry vector, so the compound side-rep is required"
        )
    fp = {
        str(k): np.asarray(v, dtype=np.float32).ravel()
        for k, v in zip(d["fingerprint_keys"], d["fingerprint_vals"])
    }
    fp_dim = len(next(iter(fp.values())))
    zero_fp = np.zeros(
        fp_dim, dtype=np.float32
    )  # DMSO / control == no chemistry (dose 0)

    pred_cpds = [c for c in test_perts if c in fp]
    if not pred_cpds:
        raise RuntimeError(
            f"Biolord-C5: none of the {len(test_perts)} held compounds has a "
            f"fingerprint (keys e.g. {sorted(fp)[:3]})"
        )

    # ---- regime detection: register the published `cell_type` categorical attribute only when every
    # inference-side lineage is also a TRAINING category (T5u). On T5c the held lineage is unseen, so
    # a categorical value for it does not exist and the lineage is carried by the residual state.
    ct_tr = (
        np.asarray([str(c) for c in d["celltype_train"]])
        if "celltype_train" in d.files
        else None
    )
    ct_inf = (
        np.asarray([str(c) for c in d["celltype_inf"]])
        if "celltype_inf" in d.files
        else None
    )
    use_ct = (
        ct_tr is not None
        and ct_inf is not None
        and len(ct_inf) == X_ctrl_inf.shape[0]
        and set(ct_inf) <= set(ct_tr)
    )
    regime = (
        "T5u/unseen-compound (cell_type categorical registered)"
        if use_ct
        else (
            "T5c/held-lineage (cell_type omitted; lineage carried by the residual"
            " state)"
        )
    )

    # ---- stratified cell cap: keep every compound represented ----
    lab_all = np.where(is_ctrl, "control", pert_train)
    keep_rows = np.arange(X.shape[0])
    if X.shape[0] > max_cells:
        rng0 = np.random.default_rng(0)
        parts = []
        for lab, cnt in zip(*np.unique(lab_all, return_counts=True)):
            idx = np.where(lab_all == lab)[0]
            take = max(1, int(round(max_cells * cnt / X.shape[0])))
            parts.append(idx if take >= cnt else rng0.choice(idx, take, replace=False))
        keep_rows = np.sort(np.concatenate(parts))
    # a training compound with no parsable SMILES has no attribute value -> drop those cells
    ok = np.asarray([lab_all[i] == "control" or lab_all[i] in fp for i in keep_rows])
    keep_rows = keep_rows[ok]
    Xtr, ltr = X[keep_rows], lab_all[keep_rows]
    ct_tr_k = ct_tr[keep_rows] if use_ct else None

    src_rows = np.arange(X_ctrl_inf.shape[0])
    if len(src_rows) > max_src:
        src_rows = np.sort(
            np.random.default_rng(0).choice(src_rows, max_src, replace=False)
        )
    Xsrc = X_ctrl_inf[src_rows]
    ct_inf_k = ct_inf[src_rows] if use_ct else None

    n_tr, n_hd = Xtr.shape[0], Xsrc.shape[0]
    obs = pd.DataFrame(
        {"split": np.array(["train"] * n_tr + ["ood"] * n_hd)},
        index=[f"tr{i}" for i in range(n_tr)] + [f"hd{i}" for i in range(n_hd)],
    )
    if use_ct:
        obs["cell_type"] = pd.Categorical(np.concatenate([ct_tr_k, ct_inf_k]))
    adata = ad.AnnData(X=np.vstack([Xtr, Xsrc]).astype(np.float32), obs=obs)
    adata.var_names = genes
    adata.obs["split"] = adata.obs["split"].astype("category")
    # the ORDERED (continuous) compound attribute — this is what makes an unseen compound predictable
    adata.obsm["compound_fp"] = np.vstack(
        [
            np.vstack([fp[c] if c != "control" else zero_fp for c in ltr]),
            np.tile(zero_fp, (n_hd, 1)),
        ]
    ).astype(np.float32)

    print(
        f"[Biolord-C5] {regime}\n"
        f"[Biolord-C5] train cells={n_tr} (ctrl={int((ltr=='control').sum())}, "
        f"train compounds={len(set(ltr)-{'control'})}) source-ctrl cells={n_hd} "
        f"targets={len(pred_cpds)}/{len(test_perts)} fp_dim={fp_dim} "
        f"latent={n_latent} epochs={epochs}",
        flush=True,
    )

    Biolord.setup_anndata(
        adata,
        ordered_attributes_keys=["compound_fp"],
        categorical_attributes_keys=(["cell_type"] if use_ct else None),
    )
    module_params = {  # published sciplex3 shape (biolord_reproducibility)
        "decoder_width": dec_w,
        "decoder_depth": 4,
        "attribute_nn_width": attr_w,
        "attribute_nn_depth": 2,
        "n_latent_attribute_ordered": n_ord,
        "n_latent_attribute_categorical": 4,
        "gene_likelihood": "normal",
        "reconstruction_penalty": 1e2,
        "unknown_attribute_penalty": 1e1,
        "unknown_attribute_noise_param": 1e-1,
        # NOT switched off here. The final 102 ruling keeps biolord T5u as N on the AUTHORS'
        # chemical path (biolord_reproducibility/scripts/biolord/sciplex3/base_experiment_sciplex3.py
        # :31-34, :41-55, :83-90), which uses RDKit + dose ordered attributes, a cell-type
        # categorical attribute AND the trained unknown latent. Turning that latent off would run a
        # different method: it is the only carrier of cell-specific state, and the normal decoder
        # does not take control expression separately. The held-unit problem this file's earlier
        # comment described is real on the CONTEXT tasks -- which is why T1/T2/T5c are excluded
        # rather than rescued by disabling a core path. On T5u the held entity is the COMPOUND, and
        # the control cells whose latent is used are training cells the split already shares.
        "attribute_dropout_rate": 0.1,
        "use_batch_norm": False,
        "use_layer_norm": False,
        "seed": 0,
    }
    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": 1e-4,
        "latent_wd": 1e-4,
        "decoder_lr": 1e-4,
        "decoder_wd": 1e-4,
        "attribute_nn_lr": 1e-2,
        "attribute_nn_wd": 4e-8,
        "step_size_lr": 45,
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
    }
    model = Biolord(
        adata=adata,
        n_latent=n_latent,
        model_name="ivcbench_c5",
        module_params=module_params,
        train_classifiers=False,
        split_key="split",
        train_split="train",
        valid_split="train",
        test_split="ood",
    )
    with tempfile.TemporaryDirectory(prefix="biolord_c5_") as td:
        model.train(
            max_epochs=epochs,
            batch_size=batch,
            plan_kwargs=trainer_params,
            early_stopping=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            num_workers=0,
            default_root_dir=td,
        )
    model.module.eval()

    # ---- prediction: the authors' compute_prediction recipe (control cells + swapped attribute) ----
    src = adata[adata.obs["split"] == "ood"].copy()
    ds = model.get_dataset(src)
    layer = "X" if "X" in ds else "layers"
    n_obs = int(ds[layer].size(0))
    dev = model.device

    def repeat_n(x, n):
        return x.to(dev).view(1, -1).repeat(n, 1)

    pred_perts, pred_means = [], []
    for c in pred_cpds:
        comb = {layer: ds[layer].to(dev), "ind_x": ds["ind_x"].to(dev)}
        for k in ds:
            if k in (layer, "ind_x"):
                continue
            comb[k] = (
                repeat_n(torch.as_tensor(fp[c], dtype=torch.float32), n_obs)
                if k == "compound_fp"
                else ds[k].to(dev)
            )
        with torch.no_grad():
            out, _ = model.module.get_expression(comb)
        pred_perts.append(c)
        pred_means.append(out.mean(0).detach().cpu().numpy().astype(np.float32))

    P = np.vstack(pred_means).astype(np.float32)
    if not np.isfinite(P).all():
        raise RuntimeError("Biolord-C5: non-finite values in the generated profiles")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object), pred_means=P)
    print(
        f"[Biolord-C5] wrote {len(pred_perts)} compound profiles x {P.shape[1]} genes "
        f"from {n_obs} source control cells",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
