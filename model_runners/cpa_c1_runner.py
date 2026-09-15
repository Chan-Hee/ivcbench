#!/usr/bin/env python
"""Categorical CPA using the public CPA.predict counterfactual API.

Run in ivc-cpa: python cpa_c1_runner.py input.npz output.npz.
T1/T2 and seen-compound T5c use a separate learned embedding per actual
condition. Unseen categorical targets are declined, never filled. A held axis
with unseen categorical values is not registered as a random new embedding:
held context enters the published encoder through X_ctrl_inf.

The installed CPA.predict documentation explicitly prescribes control
expression in query.X and target labels in query.obs. Its get_expression
passes that X into the basal encoder. No latent delta or custom decoder is
used here. Model/optimizer code is not patched.

Optional train_indices/inference_input_indices (or train_cell_ids/ctrl_inf_cell_ids)
must contain ORIGINAL identifiers in one common namespace, not local aranges.
Without them original-index disjointness remains unverified; exact expression
row overlap is still rejected. Output averaging uses float64 accumulation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np


def log(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False, sort_keys=True),
          file=sys.stderr, flush=True)


def decline(label, reason):
    print(f"[decline] {label}: {reason}", file=sys.stderr, flush=True)


def row_hashes(matrix):
    return {hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()
            for row in matrix}


def check_disjoint(data, x_train, x_inf):
    checked = False
    for train_key, inf_key in (
        ("train_cell_ids", "ctrl_inf_cell_ids"),
        ("cell_ids_train", "cell_ids_inf"),
        ("train_indices", "inference_input_indices"),
    ):
        if train_key not in data or inf_key not in data:
            continue
        left, right = np.asarray(data[train_key]).astype(str), np.asarray(data[inf_key]).astype(str)
        if left.shape != (len(x_train),) or right.shape != (len(x_inf),):
            raise ValueError(f"original ID shape mismatch: {train_key}/{inf_key}")
        shared = set(left) & set(right)
        log("original_id_overlap", keys=[train_key, inf_key], count=len(shared))
        assert not shared, f"training and inference original IDs overlap: {len(shared)}"
        checked = True
    if not checked:
        log("original_id_overlap", status="UNVERIFIED",
            reason="payload has no paired original global cell IDs; local row numbers cannot prove separation")
    shared_rows = row_hashes(x_train) & row_hashes(x_inf)
    log("exact_expression_row_overlap", count=len(shared_rows),
        limitation="identical measurements can also occur in distinct cells; conservative rejection")
    assert not shared_rows, f"training and inference share {len(shared_rows)} exact expression rows"
    if "group_train" in data and "held_label" in data:
        held = str(np.asarray(data["held_label"]).item())
        groups = np.asarray(data["group_train"]).astype(str)
        if groups.shape != (len(x_train),):
            raise ValueError("group_train shape mismatch")
        overlap = int(np.count_nonzero(groups == held))
        log("held_group_training_overlap", held=held, count=overlap)
        assert overlap == 0, f"held group {held} occurs in training"


def diagnostics(requested, labels, profiles, declined, control_mean):
    missing = sorted(set(requested) - set(labels))
    unexpected = sorted(set(labels) - set(requested))
    assert not unexpected, f"unexpected predictions: {unexpected}"
    assert len(labels) == len(set(labels)), "duplicate predicted labels"
    assert set(missing) == set(declined), "missing predictions without decline reasons"
    log("coverage", requested=requested, predicted=labels,
        missing=missing, unexpected=unexpected, decline_reasons=declined)
    identical = [[labels[i], labels[j]] for i in range(len(labels))
                 for j in range(i + 1, len(labels))
                 if np.array_equal(profiles[i], profiles[j])]
    log("bit_exact_profile_pairs", pairs=identical,
        all_equal=len(labels) > 1 and len(identical) == len(labels) * (len(labels) - 1) // 2)
    for label, profile in zip(labels, profiles):
        distance = float(np.max(np.abs(profile.astype(np.float64) - control_mean)))
        log("training_control_distance", target=label, max_abs=distance,
            near_control=distance <= 1e-5)


def save_predictions(path, labels, profiles, n_genes):
    array = np.asarray(profiles, dtype=np.float32).reshape(len(labels), n_genes)
    # Atomic output replacement; no partial/stale file is presented as a success.
    destination = Path(path)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as handle:
        temp_path = Path(handle.name)
        np.savez(handle, pred_perts=np.asarray(labels, dtype=object), pred_means=array)
    try:
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def main(in_path: str, out_path: str) -> None:
    # NumPy 2 writes object-array pickles using numpy._core; this environment
    # ships NumPy 1. The modules have the same array-reconstruction ABI here.
    aliases = {}
    if int(np.__version__.split(".")[0]) < 2:
        for name, module in (("numpy._core", np.core),
                             ("numpy._core.multiarray", np.core.multiarray),
                             ("numpy._core.numeric", np.core.numeric)):
            if name not in sys.modules:
                aliases[name] = module
                sys.modules[name] = module
    try:
        with np.load(in_path, allow_pickle=True) as archive:
            data = {key: archive[key] for key in archive.files}
    finally:
        # C extensions probe numpy._core to detect the NumPy 2 ABI. Restore
        # the real module layout before importing h5py/scvi/torch extensions.
        for name in aliases:
            del sys.modules[name]
    x = np.asarray(data["X_train"], dtype=np.float32)
    inf = np.asarray(data["X_ctrl_inf"], dtype=np.float32)
    genes = np.asarray(data["genes"]).astype(str)
    ctrl = np.asarray(data["is_control_train"], dtype=bool)
    perts = np.asarray(data["pert_train"]).astype(str)
    requested = sorted(set(np.asarray(data["test_perts"]).astype(str)) - {"control"})
    if x.ndim != 2 or inf.ndim != 2 or x.shape[1] != inf.shape[1]:
        raise ValueError("X_train and X_ctrl_inf must be matrices with identical gene columns")
    if genes.shape != (x.shape[1],) or len(set(genes)) != len(genes):
        raise ValueError("genes must be unique and match the expression columns")
    if ctrl.shape != (len(x),) or perts.shape != (len(x),):
        raise ValueError("training metadata shape mismatch")
    if not len(inf) or not ctrl.any() or not (~ctrl).any():
        raise ValueError("nonempty own controls, training controls and treated cells are required")
    if not np.isfinite(x).all() or not np.isfinite(inf).all():
        raise ValueError("nonfinite expression input")
    if not np.array_equal(ctrl, perts == "control"):
        raise ValueError("is_control_train disagrees with literal control labels")
    if not requested:
        raise ValueError("no non-control target requested")
    check_disjoint(data, x, inf)
    control_mean = x[ctrl].mean(axis=0, dtype=np.float64)
    declined = {}
    targets = []
    seen = set(perts)
    for label in requested:
        if label not in seen:
            declined[label] = "unseen categorical perturbation has no trained CPA embedding"
            decline(label, declined[label])
        else:
            targets.append(label)
    log("gene_vocabulary", status="input columns learned directly; CPA has no fixed gene vocabulary",
        n_genes=len(genes), gene_order_sha256=hashlib.sha256("\n".join(genes).encode()).hexdigest())
    if not targets:
        diagnostics(requested, [], [], declined, control_mean)
        save_predictions(out_path, [], [], len(genes))
        return

    seed = int(os.environ.get("IVCBENCH_SEED", "0"))
    epochs = int(os.environ.get("IVCBENCH_CPA_EPOCHS", "60"))
    cap = int(os.environ.get("IVCBENCH_CPA_MAXCELLS", "60000"))
    if epochs < 1 or cap < 0:
        raise ValueError("positive epochs and nonnegative maxcells required (0 means uncapped)")
    rng = np.random.default_rng(seed)
    # CPA treats '+' as a combination separator. A bijective token map prevents
    # chemical names containing '+' from silently becoming combination inputs.
    tokens = {label: f"p{i:05d}" for i, label in enumerate(sorted(seen - {"control"}))}
    tokens["control"] = "ctrl"
    train_cov, inf_cov = {}, {}
    for axis in ("celltype", "gem"):
        tk, ik = f"{axis}_train", f"{axis}_inf"
        if tk not in data or ik not in data:
            log("covariate_omitted", axis=axis, reason="paired metadata absent")
            continue
        tv, iv = np.asarray(data[tk]).astype(str), np.asarray(data[ik]).astype(str)
        if tv.shape != (len(x),) or iv.shape != (len(inf),):
            raise ValueError(f"{axis} metadata shape mismatch")
        unseen = sorted(set(iv) - set(tv))
        if unseen:
            log("covariate_omitted", axis=axis, unseen=unseen,
                reason="no trained embedding for held categories; context enters through own control X")
            continue
        # Safe tokens also avoid ambiguity in CPA's composite category strings.
        mapping = {value: f"{axis}{i:05d}" for i, value in enumerate(sorted(set(tv)))}
        train_cov[axis] = np.asarray([mapping[v] for v in tv])
        inf_cov[axis] = np.asarray([mapping[v] for v in iv])
        log("covariate_used", axis=axis, mapping=mapping)
    if not train_cov:
        train_cov["population"] = np.full(len(x), "population")
        inf_cov["population"] = np.full(len(inf), "population")
        log("covariate_used", axis="population", reason="single trained population category")

    strata = {}
    for i in range(len(x)):
        key = (perts[i],) + tuple(values[i] for values in train_cov.values())
        strata.setdefault(key, []).append(i)
    selected = []
    if cap and len(x) > cap:
        if cap < 2 * len(strata):
            raise ValueError("maxcells cannot preserve at least two cells per condition/covariate stratum")
        quota = cap // len(strata)
        for indices in strata.values():
            selected.extend(rng.choice(indices, min(len(indices), quota), replace=False).tolist())
        keep = np.asarray(sorted(selected), dtype=int)
    else:
        keep = np.arange(len(x))
    # Validation uses only non-held training data. Every condition/covariate
    # stratum retains a fitting cell; all query expression remains outside fit.
    fit = np.ones(len(keep), dtype=bool)
    selected_strata = {}
    for pos, idx in enumerate(keep):
        key = (perts[idx],) + tuple(values[idx] for values in train_cov.values())
        selected_strata.setdefault(key, []).append(pos)
    for positions in selected_strata.values():
        if len(positions) >= 2:
            n_valid = max(1, min(len(positions) - 1, int(0.1 * len(positions))))
            fit[rng.choice(positions, n_valid, replace=False)] = False
    if fit.sum() < 2 or (~fit).sum() < 2:
        raise ValueError("insufficient train-only cells for CPA fitting and validation")
    log("training_selection", original_rows=len(x), selected_rows=len(keep),
        fitting_rows=int(fit.sum()), validation_rows=int((~fit).sum()),
        selected_original_row_positions=keep.tolist(), inference_rows=len(inf))
    log("condition_token_map", mapping=tokens)
    import anndata as ad
    import cpa
    import pandas as pd
    import scvi
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("native runner requires a real CUDA device")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    scvi.settings.seed = seed
    torch.set_num_threads(int(os.environ.get("IVCBENCH_THREADS", "4")))
    log("runtime", model="CPA", seed=seed, epochs=epochs,
        device=torch.cuda.get_device_name(0), package_file=cpa.__file__,
        use_rdkit_embeddings=False, conditioning="independent learned categorical perturbation embeddings")

    covariates = list(train_cov)
    setup = dict(perturbation_key="condition", control_group="ctrl", dosage_key="dose",
                 is_count_data=False, categorical_covariate_keys=covariates, max_comb_len=1)
    # New CLI invocation normally starts with empty registries. Refuse contaminated
    # state rather than reuse another model's perturbation/covariate indices.
    if cpa.CPA.pert_encoder is not None or cpa.CPA.covars_encoder is not None:
        raise RuntimeError("CPA class registry is already initialized; invoke runner in a fresh process")
    adata = ad.AnnData(x[keep].copy(), var=pd.DataFrame(index=genes))
    adata.obs_names = np.asarray([f"training_row_{i}" for i in keep])
    adata.obs["condition"] = [tokens[p] for p in perts[keep]]
    adata.obs["dose"] = np.where(ctrl[keep], "0.0", "1.0")
    for axis, values in train_cov.items():
        adata.obs[axis] = values[keep]
    adata.obs["split"] = np.where(fit, "train", "valid")
    cpa.CPA.setup_anndata(adata, **setup)
    model = cpa.CPA(adata, split_key="split", train_split="train", valid_split="valid",
                    test_split="unused", use_rdkit_embeddings=False, recon_loss="gauss", seed=seed)
    before = {name: param.detach().cpu().clone()
              for name, param in model.module.named_parameters()
              if "pert" in name and param.requires_grad}
    batch_size = min(int(os.environ.get("IVCBENCH_CPA_BATCH", "256")), int(fit.sum()))
    if batch_size < 2:
        raise ValueError("CPA batch size must be at least 2")
    while batch_size > 2 and int(fit.sum()) % batch_size == 1:
        batch_size -= 1
    digest = hashlib.sha256()
    with open(in_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    held = str(np.asarray(data.get("held_label", "unspecified")).item())
    run_key = hashlib.sha256(
        (held + "|" + "|".join(requested)).encode()).hexdigest()[:12]
    with tempfile.TemporaryDirectory(
        prefix=f"ivc_CPA_{run_key}_seed{seed}_{digest.hexdigest()[:12]}_"
    ) as work:
        model.train(max_epochs=epochs, batch_size=batch_size, use_gpu=True,
                    save_path=False, check_val_every_n_epoch=1,
                    early_stopping_patience=8, plan_kwargs={"lr": 1e-3},
                    logger=False, enable_checkpointing=False,
                    default_root_dir=work, num_sanity_val_steps=0)
        optimized = {id(param)
                     for optimizer in model.runner.trainer.optimizers
                     for group in optimizer.param_groups
                     for param in group["params"]}
        memberships = []
        for name, param in model.module.named_parameters():
            if "pert" in name and param.requires_grad:
                included = id(param) in optimized
                delta = float((param.detach().cpu() - before[name]).abs().max())
                memberships.append({"parameter": name, "in_optimizer": included,
                                    "max_abs_weight_change": delta})
                if not included:
                    raise RuntimeError(f"trainable perturbation parameter absent from actual optimizer: {name}")
        transformation_names = [name for name, _ in model.module.named_parameters()
                                if "pert_transformation" in name]
        log("optimizer_membership", parameters=memberships,
            pert_transformation=transformation_names,
            S2="RDKit path disabled; categorical perturbation parameters checked in actual fitted optimizers")
        if transformation_names:
            raise RuntimeError("unexpected RDKit pert_transformation in categorical CPA")

        labels, profiles = [], []
        for target in targets:
            query = ad.AnnData(inf.copy(), var=pd.DataFrame(index=genes))
            query.obs_names = np.asarray([f"inference_row_{i}" for i in range(len(inf))])
            query.obs["condition"] = tokens[target]
            query.obs["dose"] = "1.0"
            for axis, values in inf_cov.items():
                query.obs[axis] = values
            # source_registry preserves the trained category-to-embedding indices.
            # New COMBINATIONS may be registered for bookkeeping; each actual
            # perturbation and covariate embedding must already be trained.
            cpa.CPA.setup_anndata(
                query, source_registry=model.adata_manager.registry,
                extend_categories=True, **setup)
            manager = cpa.CPA._get_most_recent_anndata_manager(query, required=True)
            for axis in covariates:
                mapping = manager.registry["field_registries"][axis]["state_registry"]["categorical_mapping"]
                for value in set(inf_cov[axis]):
                    assert list(mapping).index(value) == model.covars_encoder[axis][value], \
                        f"query covariate index changed: {axis}/{value}"
            # Instance validation otherwise retries transfer without category
            # extension and rejects legitimate unseen joint bookkeeping labels.
            # Every actual embedding category was checked above and before fit.
            model._register_manager_for_instance(manager)
            assert np.array_equal(query.X, inf), "own control basal input changed"
            assert list(query.var_names) == list(genes), "gene order changed"
            try:
                # Public counterfactual API: query.X is the basal control expression;
                # query.obs contains the requested target condition, not treated X.
                model.predict(query, batch_size=batch_size, n_samples=1, return_mean=True)
                cell_predictions = np.asarray(query.obsm["CPA_pred"])
                if cell_predictions.shape != inf.shape or not np.isfinite(cell_predictions).all():
                    raise RuntimeError("CPA.predict returned invalid shape or nonfinite expression")
                profile = cell_predictions.mean(axis=0, dtype=np.float64).astype(np.float32)
                log("public_prediction", target=target, internal_token=tokens[target],
                    api="cpa.CPA.predict", basal_source="X_ctrl_inf", input_cells=len(inf),
                    predicted_cells=len(cell_predictions),
                    across_cell_mean_gene_variance=float(cell_predictions.var(axis=0, dtype=np.float64).mean()))
                labels.append(target)
                profiles.append(profile)
            except Exception as exc:
                # Do not disguise a broken inference call as structural inapplicability.
                for unfinished in targets[len(labels):]:
                    decline(unfinished, f"native inference aborted: {type(exc).__name__}: {exc}")
                raise
        diagnostics(requested, labels, profiles, declined, control_mean)
        save_predictions(out_path, labels, profiles, len(genes))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: cpa_c1_runner.py input.npz output.npz")
    main(sys.argv[1], sys.argv[2])
