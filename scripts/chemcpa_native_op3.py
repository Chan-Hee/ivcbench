#!/usr/bin/env python
"""NATIVE chemCPA (cpa-tools 0.8.8) on the OP3 PBMC unseen-compound split — leak-safe, one seed.

This is the FAITHFUL chemistry-aware CPA drug-encoder path, distinct from the post-hoc
fingerprint->latent-delta-ridge runner in model_runners/cpa_c5_runner.py. Mechanism (verified in
cpa/_model.py): CPA(adata, use_rdkit_embeddings=True) builds a FROZEN nn.Embedding whose rows are the
2048-bit Morgan fingerprint (RDKit GetMorganFingerprintAsBitVect, radius 2) of every compound in the
perturbation vocabulary (built from setup_anndata + smiles_key), then learns a trainable
Linear(2048 -> n_latent) drug-effect map. An UNSEEN compound's effect is therefore a pure function of
its SMILES fingerprint through the learned linear map — no test expression touches the drug encoder.

Leak-safety via split_key (train / valid / ood):
  * train  = controls + TRAIN-compound treated cells   -> fits encoder, decoder, adversary, drug-map
  * valid  = a held-out subset of TRAIN compounds       -> validation / early-stopping ONLY
  * ood    = the HELD test compounds' cells             -> present ONLY so their FROZEN fingerprint
             rows exist in the embedding table; AnnDataSplitter routes them to test_indices, never to
             the training/validation optimiser. We never read ood expression for fitting or selection.

Prediction (counterfactual, per the cpa.predict docstring): for each held compound C, take the held
context's OWN control cells (the inference-input controls of the split, matched to test cell types),
relabel them with compound C (its frozen FP row), and decode -> the predicted treated profile. This
anchors the decode on the real control state, exactly like cpa_c5_runner.py's anchor.

Usage:  CUDA_VISIBLE_DEVICES=0 python chemcpa_native_op3.py <seed> <out_dir> [max_cells] [epochs]
Writes <out_dir>/chemcpa_native_seed<seed>.npz with per-(compound x cell_type) predicted means +
diagnostics, and prints a JSON line of leak-safety facts.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(REPO, "src"))

DMSO_SMILES = "C[S+](C)[O-]"  # OP3 control = Dimethyl Sulfoxide canonical SMILES (from obs['SMILES'])


def main(seed: int, out_dir: str, max_cells: int, epochs: int, cov_mode: str = "constant") -> None:
    # cov_mode: 'constant' -> single PBMC covariate + pooled control anchor (matches the deposited
    #   OP3 CPA/FP-ridge/cell-mean protocol so the Pearson-Δ is COMPARABLE to 0.172/0.164/0.159 and
    #   isolates the compound axis). 'celltype' -> cell_type covariate + per-lineage anchor (richer but
    #   NOT comparable: the cell-type identity offset dominates the Δ-vs-pooled-control correlation).
    import anndata as ad
    import cpa
    import pandas as pd
    import torch
    from rdkit import Chem

    from ivcbench.clusters import c5
    from ivcbench.clusters.spec import REGISTRY, _c5_held_compounds
    from ivcbench.data.loaders.op3 import load
    from ivcbench.data.schema import CONTROL_TOKEN
    from ivcbench.splits.audit import audit_split
    from ivcbench.splits.builder import build_split

    t0 = time.time()
    facts: dict = {"seed": seed, "path": "native_use_rdkit_embeddings"}

    # ------------------------------------------------------------------ data + leak-safe split
    # SAME loader / HVG panel / fingerprint side-info as every deposited OP3 entrant -> comparable.
    cs = load()
    held = _c5_held_compounds(cs)          # seed-0 ~20% compound holdout (the SAME held set as anchors)
    spec = c5.global_compound_holdout(held)
    split = build_split(cs, spec)
    audit = audit_split(cs, split)         # hard leak gate (raises on violation)
    facts["audit"] = audit

    obs = cs.obs.reset_index(drop=True)
    genes = [str(g) for g in cs.var_names]
    pert_all = obs["perturbation"].astype(str).to_numpy()
    ct_all = obs["cell_type_coarse"].astype(str).to_numpy()
    is_ctrl = obs["is_control"].astype(bool).to_numpy()

    # held compounds actually present in the test cells (labels only)
    test_perts = sorted(set(pert_all[split.test_idx]) - {CONTROL_TOKEN})
    facts["n_held_compounds_in_split"] = len(test_perts)

    # SMILES per compound from the loader's raw file (re-read the on-disk obs map).
    smiles_map = _smiles_map()
    smiles_map[CONTROL_TOKEN] = DMSO_SMILES

    # RDKit-parseability check for every held compound (pre-specified exclusion if any fails).
    bad_held = [c for c in test_perts if c not in smiles_map or Chem.MolFromSmiles(smiles_map[c]) is None]
    facts["held_without_parseable_smiles"] = bad_held
    test_perts = [c for c in test_perts if c not in bad_held]

    train_idx = split.train_idx
    train_perts_all = pert_all[train_idx]
    train_cpds = sorted(set(train_perts_all) - {CONTROL_TOKEN})
    # drop any train compound lacking a parseable SMILES (the embedding builder would crash on it)
    bad_train = [c for c in train_cpds if c not in smiles_map or Chem.MolFromSmiles(smiles_map[c]) is None]
    facts["train_without_parseable_smiles"] = bad_train
    keep_train_cpd = set(train_cpds) - set(bad_train)

    # ------------------------------------------------------------------ assemble train/valid cells
    rng = np.random.default_rng(seed)
    # validation = a held-out subset of TRAIN compounds (NEVER test compounds). 15% of train compounds.
    tc = sorted(keep_train_cpd)
    n_val = max(3, int(round(0.15 * len(tc))))
    val_cpds = set(rng.choice(np.array(tc, dtype=object), size=min(n_val, len(tc)), replace=False).tolist())
    facts["n_train_compounds"] = len(tc)
    facts["n_valid_compounds"] = len(val_cpds)

    ctrl_train_pos = train_idx[is_ctrl[train_idx]]
    treat_train_pos = train_idx[(~is_ctrl[train_idx]) & np.isin(train_perts_all, list(keep_train_cpd))]

    # cap training cells (stratified by compound) so the panel trains in time, like cpa_c5_runner.
    def _cap(pos, per):
        cpds = pert_all[pos]
        keep = []
        for c in pd.unique(cpds):
            ci = pos[cpds == c]
            keep.append(ci if len(ci) <= per else rng.choice(ci, per, replace=False))
        return np.sort(np.concatenate(keep)) if keep else pos

    n_groups = max(1, len(keep_train_cpd) + 1)
    per = max(4, max_cells // n_groups)
    ctrl_keep = ctrl_train_pos if len(ctrl_train_pos) <= max_cells // 3 else rng.choice(
        ctrl_train_pos, max_cells // 3, replace=False)
    treat_keep = _cap(treat_train_pos, per)
    train_keep = np.sort(np.concatenate([ctrl_keep, treat_keep]))

    # ood cells: the held test compounds' treated cells — included in the AnnData ONLY so the frozen
    # FP embedding has their rows. Capped (we never use their expression for anything but vocabulary).
    ood_pos = split.test_idx[np.isin(pert_all[split.test_idx], test_perts)]
    ood_keep = ood_pos if len(ood_pos) <= max_cells // 2 else rng.choice(ood_pos, max_cells // 2, replace=False)

    all_pos = np.concatenate([train_keep, ood_keep])
    X = cs.X[all_pos].astype(np.float32)
    cond = np.where(is_ctrl[all_pos], CONTROL_TOKEN, pert_all[all_pos]).astype(object)
    # covariate: 'constant' uses a single PBMC bucket (deposited-comparable, isolates the compound axis);
    # 'celltype' uses the real lineage (not comparable to anchors — see cov_mode docstring).
    ct = (np.array(["PBMC"] * len(all_pos), dtype=object) if cov_mode == "constant"
          else ct_all[all_pos].astype(object))

    # split_key: train compounds -> 'train' (+ controls), val compounds -> 'valid', held -> 'ood'
    split_col = np.empty(len(all_pos), dtype=object)
    n_tr = len(train_keep)
    for i in range(len(all_pos)):
        c = cond[i]
        if i >= n_tr:
            split_col[i] = "ood"
        elif c == CONTROL_TOKEN:
            split_col[i] = "train"
        elif c in val_cpds:
            split_col[i] = "valid"
        else:
            split_col[i] = "train"
    # controls also seed the valid split a bit so the validation loss is well-defined
    val_ctrl = rng.random(len(all_pos)) < 0.0  # keep controls in train; valid has treated val-cpd cells
    split_col[(cond == CONTROL_TOKEN) & val_ctrl] = "valid"

    smiles_col = np.array([smiles_map[c] for c in cond], dtype=object)

    adata = ad.AnnData(X.copy())
    adata.var_names = genes
    adata.obs["condition"] = pd.Categorical(cond)
    adata.obs["SMILES"] = smiles_col.astype(str)
    adata.obs["cell_type"] = pd.Categorical(ct)
    adata.obs["split"] = pd.Categorical(split_col)
    adata.obs["dose"] = 1.0  # dose NOT used as a conditioning axis (see manifest); flat dosage

    facts["n_train_cells"] = int((split_col == "train").sum())
    facts["n_valid_cells"] = int((split_col == "valid").sum())
    facts["n_ood_cells"] = int((split_col == "ood").sum())
    facts["cov_mode"] = cov_mode
    facts["covariates"] = (["PBMC_constant"] if cov_mode == "constant" else ["cell_type"])
    facts["dose_used"] = False
    facts["n_genes"] = len(genes)

    # ------------------------------------------------------------------ native chemCPA setup + train
    # reset cpa class-level caches (fresh per process anyway, but be explicit)
    cpa.CPA.pert_encoder = None
    cpa.CPA.covars_encoder = None
    cpa.CPA.pert_smiles_map = None

    cpa.CPA.setup_anndata(
        adata,
        perturbation_key="condition",
        control_group=CONTROL_TOKEN,
        smiles_key="SMILES",
        is_count_data=False,
        categorical_covariate_keys=["cell_type"],
        max_comb_len=1,
    )
    model = cpa.CPA(
        adata,
        split_key="split",
        train_split="train",
        valid_split="valid",
        test_split="ood",
        use_rdkit_embeddings=True,
        n_latent=64,
        recon_loss="gauss",
    )
    facts["use_rdkit_embeddings"] = bool(model.module.pert_network.use_rdkit
                                         if hasattr(model.module, "pert_network") else True)

    # confirm the frozen FP embedding (drug encoder NOT a learnable lookup -> a function of FP)
    try:
        emb = model.module.pert_network.pert_embedding
        facts["drug_embedding_frozen"] = (not emb.weight.requires_grad)
        facts["drug_embedding_dim"] = int(emb.weight.shape[1])
    except Exception as e:  # noqa: BLE001
        facts["drug_embedding_introspect_error"] = str(e)[:200]

    model.train(
        max_epochs=epochs,
        batch_size=256,
        early_stopping_patience=8,
        check_val_every_n_epoch=2,
        use_gpu=torch.cuda.is_available(),
        plan_kwargs={"lr": 1e-3},
    )
    try:
        facts["epoch_history_len"] = int(len(model.epoch_history)) if model.epoch_history is not None else None
    except Exception:  # noqa: BLE001
        pass

    # ------------------------------------------------------------------ counterfactual prediction
    # Relabel control anchor cells with each held compound (its FROZEN FP row drives the drug effect)
    # and decode. setup_anndata reuses the trained cls.pert_encoder / pert_smiles_map (NOT reset), so
    # vocabulary + frozen FP map are identical to training. Leak-safe: only control X is the basal
    # input; the held compound's effect comes purely from its fingerprint.
    #   * constant mode  -> anchor = pooled control cells, covariate = PBMC. One profile per compound,
    #     tiled to all that compound's test cells (the DEPOSITED CPA/FP-ridge/cell-mean protocol).
    #   * celltype mode  -> per-lineage anchor + cell_type covariate; one profile per (compound x ct).
    inf_pos = split.inference_input_idx
    n_anchor = 256
    blocks_X, blocks_cond, blocks_ct, blocks_tag = [], [], [], []
    if cov_mode == "constant":
        anchor_pos = inf_pos if len(inf_pos) else train_idx[is_ctrl[train_idx]]
        if len(anchor_pos) > n_anchor:
            anchor_pos = rng.choice(anchor_pos, n_anchor, replace=False)
        Xa = cs.X[anchor_pos].astype(np.float32)
        for cpd in test_perts:
            blocks_X.append(Xa)
            blocks_cond.extend([cpd] * Xa.shape[0])
            blocks_ct.extend(["PBMC"] * Xa.shape[0])
            blocks_tag.extend([f"{cpd}|||PBMC"] * Xa.shape[0])
    else:
        for ctype in sorted(set(ct_all[split.test_idx])):
            apos = inf_pos[ct_all[inf_pos] == ctype]
            if len(apos) == 0:
                apos = inf_pos
            if len(apos) > n_anchor:
                apos = rng.choice(apos, n_anchor, replace=False)
            Xa = cs.X[apos].astype(np.float32)
            for cpd in test_perts:
                blocks_X.append(Xa)
                blocks_cond.extend([cpd] * Xa.shape[0])
                blocks_ct.extend([ctype] * Xa.shape[0])
                blocks_tag.extend([f"{cpd}|||{ctype}"] * Xa.shape[0])

    Xp = np.vstack(blocks_X).astype(np.float32)
    pred_ad = ad.AnnData(Xp.copy())
    pred_ad.var_names = genes
    pred_ad.obs["condition"] = pd.Categorical(blocks_cond,
                                              categories=list(adata.obs["condition"].cat.categories))
    pred_ad.obs["SMILES"] = np.array([smiles_map[c] for c in blocks_cond], dtype=object).astype(str)
    pred_ad.obs["cell_type"] = pd.Categorical(blocks_ct,
                                              categories=list(adata.obs["cell_type"].cat.categories))
    pred_ad.obs["dose"] = 1.0
    tags = np.array(blocks_tag, dtype=object)

    # register scvi fields on the prediction adata (reuses cached encoder/smiles map -> same vocab)
    cpa.CPA.setup_anndata(
        pred_ad,
        perturbation_key="condition",
        control_group=CONTROL_TOKEN,
        smiles_key="SMILES",
        is_count_data=False,
        categorical_covariate_keys=["cell_type"],
        max_comb_len=1,
    )
    model.predict(pred_ad, batch_size=512, n_samples=20, return_mean=True)
    px_all = np.asarray(pred_ad.obsm["CPA_pred"], dtype=np.float32)

    pred_rows = []  # (compound, cell_type, pred_mean_vector)
    for tag in np.unique(tags):
        cpd, ctype = tag.split("|||")
        m = tags == tag
        pred_rows.append((cpd, ctype, px_all[m].mean(0)))

    facts["n_pred_strata"] = len(pred_rows)
    facts["wall_clock_s"] = round(time.time() - t0, 1)

    os.makedirs(out_dir, exist_ok=True)
    tag = "" if cov_mode == "constant" else f"_{cov_mode}"
    out = os.path.join(out_dir, f"chemcpa_native_seed{seed}{tag}.npz")
    np.savez(
        out,
        pred_compounds=np.array([r[0] for r in pred_rows], dtype=object),
        pred_celltypes=np.array([r[1] for r in pred_rows], dtype=object),
        pred_means=np.vstack([r[2] for r in pred_rows]).astype(np.float32),
        genes=np.asarray(genes),
        test_perts=np.array(test_perts, dtype=object),
        seed=seed,
        cov_mode=cov_mode,
    )
    facts["out"] = out
    print("CHEMCPA_FACTS " + json.dumps(facts))


def _smiles_map() -> dict:
    """compound name -> SMILES, read from the on-disk OP3 obs (backed)."""
    import anndata
    import pandas as pd
    path = os.environ.get("IVCBENCH_OP3_PATH",
                          os.path.join(REPO, "data/C5/op3/GSE279945_sc_counts_processed.h5ad"))
    a = anndata.read_h5ad(path, backed="r")
    obs = a.obs
    m = (obs[["sm_name", "SMILES"]].astype(str).drop_duplicates("sm_name")
         .set_index("sm_name")["SMILES"].to_dict())
    return {str(k): str(v) for k, v in m.items() if v not in ("nan", "None", "")}


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "outputs/additional_models")
    max_cells = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    cov_mode = sys.argv[5] if len(sys.argv) > 5 else "constant"
    main(seed, out_dir, max_cells, epochs, cov_mode)
