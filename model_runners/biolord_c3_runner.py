#!/usr/bin/env python
"""Biolord runner for the UNSEEN-PERTURBATION GENE tasks — `ctx-biolord` conda env.

Invoked by ivcbench.baselines.heavy.BiolordC3 (T3 primary-T CRISPR, T4 Frangieh unseen KO):
    <env python> biolord_c3_runner.py <in.npz> <out.npz>

Why this is NATIVE, not adapted
-------------------------------
biolord (Piran et al., Nat Biotechnol 2024) decomposes a cell into a residual "unknown-attribute"
state plus DISENTANGLED annotated attributes and generates a cell under changed attribute values.
The abstract's claim -- "outperforming state-of-the-art methods in predictions of cellular response
to unseen drugs and genetic perturbations" -- is backed by a released GENETIC-perturbation experiment
that is exactly this task, not only the sci-Plex drug experiment the C5 runner reproduces:

  official code, nitzanlab/biolord_reproducibility
    * scripts/biolord/adamson/base_experiment_adamson.py -- the OOD subgroup it scores is
      `for ood_set in ["unseen_single"]`, i.e. single-gene perturbations ABSENT from training.
    * the model is set up with ONE ordered attribute and no categorical attribute:
          ordered_attributes_key = varying_arg["ordered_attributes_key"]
          biolord.Biolord.setup_anndata(adata_single,
                                        ordered_attributes_keys=[ordered_attributes_key],
                                        categorical_attributes_keys=None)
      and scripts/biolord/adamson/adamson_config_optimal.py fixes that key:
          "ordered_attributes_key": "perturbation_neighbors"
    * an unseen gene is predicted by keeping CONTROL input and swapping in the held gene's attribute
      row -- the same recipe the compound runner uses, with a gene-side vector in place of RDKit:
          dataset_pred = dataset_control.copy()
          dataset_pred[ordered_attributes_key] = repeat_n(
              dataset_reference[ordered_attributes_key][idx_ref, :], n_obs)
          test_preds, _ = model.module.get_expression(dataset_pred)
    * `perturbation_neighbors` is prior knowledge, not measurement. From the released preprocessing
      notebook (notebooks/perturbations/adamson/1_perturbations_adamson_preprocessing.ipynb):
          df = pd.read_csv(go_path)                      # GEARS go.csv: source, target, importance
          df = df.groupby('target').apply(lambda x: x.nlargest(20 + 1, ['importance']))
          def get_map(pert):
              tmp = pd.DataFrame(np.zeros(len(gene_list)), index=gene_list)
              tmp.loc[df[df.target == pert].source.values, :] = \
                  df[df.target == pert].importance.values[:, np.newaxis]
              return tmp.values.flatten()
          adata_single.obsm["perturbation_neighbors"] = df_singleperts_emb
      i.e. for each perturbed gene, the GO-similarity ("importance" = Jaccard index of the two genes'
      GO term sets) to its top-20 GO neighbours over the perturbation vocabulary, zero elsewhere;
      `ctrl` is not a target in go.csv, so the control row's attribute is the ZERO vector.
  So an unseen gene is an unseen VALUE of a continuous attribute the model already trains on, and the
  prediction is defined without any external regression on top of biolord. Same argument as the
  compound side (biolord_c5_runner.py), same released entry point, different prior-knowledge vector.

This runner is that published recipe, verbatim in structure:
  * samples are CONDITION-LEVEL profiles (the authors' `adata_single`: one row per perturbation, the
    mean profile of its cells, plus one `ctrl` row) -- built here from the TRAIN fold only;
  * the single ordered attribute is the gene-side vector (see below), zero for control;
  * the authors' tuned genetic hyper-parameters (adamson_config_optimal.py), including
    `"unknown_attributes": False` -- in the genetic setting the residual/basal latent is switched off
    (biolord/_module.py::RegularizedEmbedding multiplies the embedding by `embed`), so the generated
    profile is a function of the attribute alone and does not depend on which control rows are fed in;
  * prediction swaps the held gene's attribute row onto control input and decodes.

Gene-side representation (in the role of `perturbation_neighbors`), resolved in this order:
  1. `payload["gene_embedding_*"]` -- the harness-supplied representation (side_info['gene_embedding'])
     if a split provides one;
  2. GO  -- the published construction, rebuilt from the cached GEARS gene2go annotation:
     importance = Jaccard(GO(g), GO(g*)) over the perturbation vocabulary (train + held gene names),
     top-20 neighbours + self, all-zero columns dropped (`keep_idx = pert2neighbor.sum(0) > 0`).
     Uses gene NAMES and an external ontology only -- no expression, so nothing about the held cells
     enters. This is the default and what the census entry means;
  3. PCA -- the benchmark's uniform leak-safe fallback (control-only PCA gene loadings, exactly as
     scgen_runner.py / cpa_runner.py / linear-shift-KOemb build it) if no gene2go asset is found.
The mode actually used is printed and is part of the run log.

T3 and T4 hold out the PERTURBATION, so this runner emits ONE PROFILE PER HELD GENE (the adapter
therefore leaves pred_key_is_group False). A degenerate prediction -- identical profiles across all
held genes, i.e. the control-mean floor wearing a model's name -- is a hard error here, not a silent
number.

Leak-safe: only payload["X_train"] is aggregated into training rows; the inference control cells are
registered as split="ood" and are only ever decoded; held-gene cells are not in the payload at all.
Knobs: $IVCBENCH_BIOLORD_C3_{GENESIDE,EPOCHS,LATENT,BATCH,ORDERED_LATENT,ATTR_WIDTH,ATTR_DEPTH,
DECODER_WIDTH,DECODER_DEPTH,MAXSRC,TOPK,MINCELLS,SEED,GENE_EMB}; $IVCBENCH_GENE2GO for the annotation.
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

# GEARS' gene2go asset (dict: gene symbol -> set of GO terms). The repo path is the one
# scripts/c3_nearest_gene_baseline.py already documents; the third entry is the local GEARS install.
_GENE2GO_CANDIDATES = [
    Path(__file__).resolve().parents[1] / "data/_assets/gears/gene2go_all.pkl",
    Path(
        "/data1/home/chlee/projects/single_cell_fm/scFoundation/GEARS/data/gene2go.pkl"
    ),
]


def _resolve_gene2go() -> Path | None:
    env = os.environ.get("IVCBENCH_GENE2GO")
    cands = ([Path(env)] if env else []) + _GENE2GO_CANDIDATES
    for p in cands:
        if p.exists():
            return p
    return None


def _go_neighbor_vectors(
    vocab: list[str], gene2go: dict, topk: int = 20
) -> dict | None:
    """The published `pert2neighbor` construction: for each gene, the GO-Jaccard similarity to its
    top-`topk` GO neighbours (plus itself) over `vocab`, zero elsewhere; all-zero columns dropped.
    """
    import scipy.sparse as sp

    sets = [set(gene2go.get(g, ()) or ()) for g in vocab]
    terms = sorted({t for s in sets for t in s})
    if not terms:
        return None
    tpos = {t: i for i, t in enumerate(terms)}
    rows, cols = [], []
    for i, s in enumerate(sets):
        for t in s:
            rows.append(i)
            cols.append(tpos[t])
    M = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(vocab), len(terms)),
    )
    inter = np.asarray((M @ M.T).todense(), dtype=np.float64)
    sz = np.asarray(M.sum(1)).ravel()
    union = sz[:, None] + sz[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter / union, 0.0)
    A = np.zeros_like(J, dtype=np.float32)
    k = int(min(topk + 1, J.shape[1]))
    for i in range(J.shape[0]):  # nlargest(20 + 1) per TARGET row, as published
        idx = np.argpartition(J[i], -k)[-k:]
        idx = idx[J[i, idx] > 0]
        A[i, idx] = J[i, idx]
    keep = A.sum(0) > 0  # keep_idx = pert2neighbor.sum(0) > 0
    if not keep.any():
        return None
    A = A[:, keep]
    return {g: A[i].astype(np.float32) for i, g in enumerate(vocab)}


def _pca_gene_loadings(vocab, genes, ctrl_X, n_emb: int) -> dict:
    """The benchmark's uniform leak-safe gene-side vector: control-only PCA gene loadings."""
    from sklearn.decomposition import PCA

    k = int(min(n_emb, ctrl_X.shape[0] - 1, ctrl_X.shape[1]))
    comp = PCA(n_components=max(2, k), random_state=0).fit(ctrl_X).components_.T
    gpos = {g: i for i, g in enumerate(genes)}
    dim = comp.shape[1]
    return {
        g: (
            comp[gpos[g]].astype(np.float32)
            if g in gpos
            else np.zeros(dim, dtype=np.float32)
        )
        for g in vocab
    }


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import pandas as pd
    import scanpy as sc  # noqa: F401  (biolord expects the scanpy stack present)
    import torch
    from biolord import Biolord

    ev = os.environ.get
    mode_req = ev(
        "IVCBENCH_BIOLORD_C3_GENESIDE", "auto"
    ).lower()  # auto | payload | go | pca
    epochs = int(
        ev("IVCBENCH_BIOLORD_C3_EPOCHS", "1000")
    )  # adamson_config_optimal max_epochs
    n_latent = int(ev("IVCBENCH_BIOLORD_C3_LATENT", "32"))
    batch = int(ev("IVCBENCH_BIOLORD_C3_BATCH", "32"))
    n_ord = int(ev("IVCBENCH_BIOLORD_C3_ORDERED_LATENT", "512"))
    attr_w = int(ev("IVCBENCH_BIOLORD_C3_ATTR_WIDTH", "64"))
    attr_d = int(ev("IVCBENCH_BIOLORD_C3_ATTR_DEPTH", "6"))
    dec_w = int(ev("IVCBENCH_BIOLORD_C3_DECODER_WIDTH", "64"))
    dec_d = int(ev("IVCBENCH_BIOLORD_C3_DECODER_DEPTH", "1"))
    max_src = int(
        ev("IVCBENCH_BIOLORD_C3_MAXSRC", "256")
    )  # control rows decoded per gene
    topk = int(ev("IVCBENCH_BIOLORD_C3_TOPK", "20"))  # published nlargest(20 + 1)
    min_cells = int(
        ev("IVCBENCH_BIOLORD_C3_MINCELLS", "1")
    )  # cells needed for a training row
    seed = int(ev("IVCBENCH_BIOLORD_C3_SEED", "0"))  # benchmark seed policy (paper: 42)
    n_emb = int(ev("IVCBENCH_BIOLORD_C3_GENE_EMB", "50"))  # PCA-fallback width

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    pert_train = np.asarray([str(p) for p in d["pert_train"]])
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    held = sorted({str(p) for p in d["test_perts"]} - {"control"})
    if not held:
        raise RuntimeError("Biolord-C3: no held perturbation label in the payload")
    if not is_ctrl.any():
        raise RuntimeError(
            "Biolord-C3: training fold has no control cells (needed for the ctrl row)"
        )

    lab_all = np.where(is_ctrl, "control", pert_train)
    train_perts = [
        p
        for p in sorted(set(lab_all) - {"control"})
        if int((lab_all == p).sum()) >= min_cells
    ]
    if len(train_perts) < 3:
        raise RuntimeError(
            f"Biolord-C3: only {len(train_perts)} training perturbations -- the "
            "attribute network has nothing to learn the gene-side map from"
        )
    vocab = sorted(set(train_perts) | set(held))

    # ---------------- gene-side representation (the `perturbation_neighbors` role) ----------------
    repr_map, mode, note = None, None, ""
    if mode_req in ("auto", "payload") and "gene_embedding_keys" in d.files:
        supplied = {
            str(k): np.asarray(v, dtype=np.float32).ravel()
            for k, v in zip(d["gene_embedding_keys"], d["gene_embedding_vals"])
        }
        dim = len(next(iter(supplied.values())))
        repr_map = {g: supplied.get(g, np.zeros(dim, dtype=np.float32)) for g in vocab}
        mode, note = "payload", "harness side_info['gene_embedding']"
    if repr_map is None and mode_req in ("auto", "go"):
        g2g_path = _resolve_gene2go()
        if g2g_path is not None:
            with open(g2g_path, "rb") as fh:
                gene2go = pickle.load(fh)
            repr_map = _go_neighbor_vectors(vocab, gene2go, topk=topk)
            if repr_map is not None:
                cov = sum(1 for g in vocab if g in gene2go)
                mode, note = (
                    "go",
                    f"GEARS gene2go {g2g_path} (annotated {cov}/{len(vocab)} genes)",
                )
        if repr_map is None and mode_req == "go":
            raise RuntimeError(
                "Biolord-C3: GENESIDE=go requested but no gene2go asset was found"
                " (looked at $IVCBENCH_GENE2GO and"
                f" {[str(p) for p in _GENE2GO_CANDIDATES]})"
            )
    if repr_map is None:
        repr_map = _pca_gene_loadings(vocab, genes, X[is_ctrl], n_emb)
        mode = "pca"
        note = "control-only PCA gene loadings" + (
            " (requested)" if mode_req == "pca" else " (no gene2go asset found)"
        )
    attr_dim = len(next(iter(repr_map.values())))
    zero_attr = np.zeros(
        attr_dim, dtype=np.float32
    )  # control == no perturbation attribute
    n_dead_held = sum(1 for g in held if not np.any(repr_map[g]))
    if n_dead_held == len(held):
        raise RuntimeError(
            f"Biolord-C3: all {len(held)} held genes have an all-zero gene-side vector "
            f"under mode='{mode}' -- every prediction would be the same profile, i.e. "
            "the control floor. Refusing to emit it."
        )

    # ---------------- condition-level samples (the authors' `adata_single`), TRAIN fold only ------
    rows_X = [X[is_ctrl].mean(0)]
    rows_lab = ["control"]
    rows_attr = [zero_attr]
    for p in train_perts:
        rows_X.append(X[lab_all == p].mean(0))
        rows_lab.append(p)
        rows_attr.append(repr_map[p])
    n_cond = len(rows_lab)

    src_rows = np.arange(X_ctrl_inf.shape[0])
    if len(src_rows) > max_src:
        src_rows = np.sort(
            np.random.default_rng(seed).choice(src_rows, max_src, replace=False)
        )
    Xsrc = X_ctrl_inf[src_rows]
    n_src = Xsrc.shape[0]

    # validation = a random subset of TRAINING conditions (the published run early-stops on the
    # GEARS validation conditions); too few conditions to hold any out -> train the full schedule
    rng = np.random.default_rng(seed)
    n_val = int(round(0.1 * len(train_perts)))
    n_val = n_val if n_val >= 2 else 0
    val_perts = (
        set(rng.choice(train_perts, n_val, replace=False).tolist()) if n_val else set()
    )
    split = np.array(
        ["test" if lb in val_perts else "train" for lb in rows_lab] + ["ood"] * n_src
    )

    obs = pd.DataFrame(
        {"split": split, "condition": np.array(rows_lab + ["control"] * n_src)},
        index=[f"c{i}" for i in range(n_cond)] + [f"s{i}" for i in range(n_src)],
    )
    adata = ad.AnnData(
        X=np.vstack([np.vstack(rows_X), Xsrc]).astype(np.float32), obs=obs
    )
    adata.var_names = genes
    adata.obs["split"] = adata.obs["split"].astype("category")
    adata.obsm["perturbation_neighbors"] = np.vstack(
        [np.vstack(rows_attr), np.tile(zero_attr, (n_src, 1))]
    ).astype(np.float32)

    print(
        f"[Biolord-C3] gene-side={mode} ({note}) dim={attr_dim}\n[Biolord-C3] condition"
        f" rows={n_cond} (1 ctrl + {len(train_perts)} train perts, {n_val} held for"
        f" validation) source-ctrl rows={n_src} genes={len(genes)}\n[Biolord-C3] held"
        f" genes={len(held)} (all-zero gene-side vector: {n_dead_held})"
        f" latent={n_latent} epochs<={epochs} seed={seed}",
        flush=True,
    )

    module_params = {  # scripts/biolord/adamson/adamson_config_optimal.py
        "decoder_width": dec_w,
        "decoder_depth": dec_d,
        "decoder_activation": False,
        "attribute_nn_width": attr_w,
        "attribute_nn_depth": attr_d,
        "attribute_nn_activation": False,
        "n_latent_attribute_ordered": n_ord,
        "n_latent_attribute_categorical": 16,
        "gene_likelihood": "normal",
        "reconstruction_penalty": 1e3,
        "unknown_attribute_penalty": 1e4,
        "unknown_attribute_noise_param": 1e-1,
        "unknown_attributes": (
            False
        ),  # the genetic setting switches the residual latent OFF
        "attribute_dropout_rate": 0.1,
        "use_batch_norm": False,
        "use_layer_norm": False,
        "seed": seed,
    }
    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": 1e-4,
        "latent_wd": 1e-3,
        "decoder_lr": 1e-3,
        "decoder_wd": 1e-2,
        "attribute_nn_lr": 1e-3,
        "attribute_nn_wd": 4e-8,
        "step_size_lr": 45,
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
    }

    Biolord.setup_anndata(
        adata,
        ordered_attributes_keys=["perturbation_neighbors"],
        categorical_attributes_keys=None,
    )
    model = Biolord(
        adata=adata,
        n_latent=n_latent,
        model_name="ivcbench_c3",
        module_params=module_params,
        train_classifiers=False,
        split_key="split",
        train_split="train",
        valid_split=("test" if n_val else "train"),
        test_split="ood",
    )
    with tempfile.TemporaryDirectory(prefix="biolord_c3_") as td:
        _kw = dict(
            max_epochs=epochs,
            batch_size=batch,
            plan_kwargs=trainer_params,
            check_val_every_n_epoch=5,
            enable_progress_bar=False,
            enable_checkpointing=False,
            num_workers=0,
            default_root_dir=td,
        )
        try:
            model.train(early_stopping=bool(n_val), early_stopping_patience=10, **_kw)
        except RuntimeError as e:
            # On the smaller CRISPR datasets the held-out validation conditions are too few for
            # biolord to emit `val_biolord_metric`, so the early-stopping callback aborts the run.
            # The published fallback is simply to train the full schedule, which is what the
            # no-validation branch above already does when no condition can be spared.
            if "val_biolord_metric" not in str(e):
                raise
            print(
                f"[Biolord-C3] validation metric unavailable ({n_val} held conditions);"
                f" training the full {epochs}-epoch schedule without early stopping",
                flush=True,
            )
            model = Biolord(
                adata=adata,
                n_latent=n_latent,
                model_name="ivcbench_c3",
                module_params=module_params,
                train_classifiers=False,
                split_key="split",
                train_split="train",
                valid_split="train",
                test_split="ood",
            )
            model.train(early_stopping=False, **_kw)
    model.module.eval()

    # ---------------- prediction: control input + the held gene's attribute row swapped in --------
    src = adata[adata.obs["split"] == "ood"].copy()
    ds = model.get_dataset(src)
    layer = "X" if "X" in ds else "layers"
    n_obs = int(ds[layer].size(0))
    dev = model.device

    def repeat_n(x, n):
        return x.to(dev).view(1, -1).repeat(n, 1)

    def decode(vec):
        comb = {layer: ds[layer].to(dev), "ind_x": ds["ind_x"].to(dev)}
        for k in ds:
            if k in (layer, "ind_x"):
                continue
            comb[k] = (
                repeat_n(torch.as_tensor(vec, dtype=torch.float32), n_obs)
                if k == "perturbation_neighbors"
                else ds[k].to(dev)
            )
        with torch.no_grad():
            out, _ = model.module.get_expression(comb)
        return out.mean(0).detach().cpu().numpy().astype(np.float32)

    ctrl_profile = decode(zero_attr)  # the model's own control-attribute output
    pred_perts, pred_means = [], []
    for g in held:
        pred_perts.append(g)
        pred_means.append(decode(repr_map[g]))

    P = np.vstack(pred_means).astype(np.float32)
    if not np.isfinite(P).all():
        raise RuntimeError("Biolord-C3: non-finite values in the generated profiles")

    # ---------------- anti-floor check: one profile PER held gene, and they must differ -----------
    dev_ctrl = float(np.abs(P - ctrl_profile[None, :]).mean())
    spread = float(P.std(0).mean()) if P.shape[0] > 1 else float("nan")
    spread_max = (
        float(np.abs(P[:, None, :] - P[None, :, :]).max())
        if P.shape[0] > 1
        else float("nan")
    )
    if P.shape[0] > 1 and spread_max < 1e-6:
        raise RuntimeError(
            f"Biolord-C3: the {P.shape[0]} held-gene profiles are identical to"
            f" {spread_max:.2e} -- the attribute pathway carries no perturbation"
            " information, so this cell would score as the control-mean floor."
            " Refusing to emit it."
        )
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object), pred_means=P)
    print(
        f"[Biolord-C3] wrote {len(pred_perts)} held-gene profiles x {P.shape[1]} genes"
        f" from {n_obs} control rows | mean|pred-ctrl_pred|={dev_ctrl:.4f} across-gene"
        f" SD={spread:.4f} max pairwise diff={spread_max:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
