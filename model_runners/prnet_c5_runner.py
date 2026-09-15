#!/usr/bin/env python
"""PRnet runner (C5 / OP3) — `ivc-prnet` conda env.

PRnet (Qi et al., Nat Commun 2024, s41467-024-53457-1; code
github.com/Perturbation-Response-Prediction/PRnet, Apache-2.0) is a perturbation-conditioned
generative model: a Perturb-adaptor embeds the compound's rFCFP4 fingerprint scaled by log10(dose+1),
a Perturb-encoder maps [unperturbed profile ‖ perturbation embedding] to a latent, and a
Perturb-decoder emits the post-treatment profile as per-gene (mean, variance) under a Gaussian NLL.

Why ONE runner serves both C5 entry points
------------------------------------------
PRnet has NO cell-line/lineage covariate: the cell context enters only through the *unperturbed
profile* it is conditioned on, and the compound enters only through chemistry. Its two published
evaluations map directly onto our two splits with no change of mechanism:
  * `compound_split`  (unseen compounds)  -> C5_global_compound_holdout  (T5u)
  * `cell_line_split` (unseen cell lines) -> C5_loct_<lineage>           (T5c)
so the only difference between the two is which compounds/cells the train fold contains. Both are
NATIVE, not adapted.

Gene space. PRnet has NO landmark-panel requirement: `x_dimension` is a plain hyper-parameter (978
for the L1000 runs, 5000 for Sci-Plex), so the harness's 2000-HVG OP3 panel is passed straight
through. The payload's X is already library-size + log1p normalized by ivcbench.data.preprocess --
the SAME pipeline every other model in the paper consumes -- so the `normalize_total`/`log1p` that
PRnet's own train_sciplex.py applies to raw counts is deliberately NOT repeated here.

Leak-safety. Training consumes the payload's train fold only. Model selection uses a validation arm
drawn along the axis being held out — a subset of TRAIN compounds on T5u, random treated cells on T5c
— so no held compound and no held-lineage treated cell ever enters fitting or early stopping.
Prediction for a held target is a forward pass on (held context's own control cells, chemistry) — the
model never sees held-out response.

Invoked by the harness as:  <ivc-prnet python> prnet_c5_runner.py <in.npz> <out.npz>

Env knobs: IVCBENCH_PRNET_{REPO,EPOCHS,MAXCELLS,BATCH,NDECODE,SMILES,SEED,DOSE}.
"""
from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]  # .../ivcbench
_VENDOR_DEFAULT = _REPO.parent / "benchmark" / "vendor" / "PRnet"
_SMILES_DEFAULT = (
    _REPO / "outputs" / "onboarding" / "PRnet" / "op3_compound_smiles_dose.csv"
)
_CONTROL = "control"


class _FastIndexList(list):
    """Drop-in for PRnet's `dense_adata_index`: same list, but O(1) `.index()`.

    PRnet's DrugDoseAnnDataset resolves each item's paired control with a linear
    `list.index()` scan over EVERY cell in the split, i.e. O(n_cells) per sample. On a 90k-cell OP3
    fold that is ~5e11 comparisons per 100-epoch run. Behaviour is identical (obs names are unique);
    only the lookup cost changes.
    """

    def __init__(self, items):
        super().__init__(items)
        self._pos = {v: i for i, v in enumerate(items)}

    def index(self, value, *args, **kwargs):  # noqa: A003
        return self._pos[value]


def _smiles_dose_tables(compounds):
    """compound -> (SMILES, dose_uM). CSV first, then the on-disk OP3 obs, else empty."""
    import pandas as pd

    path = os.environ.get("IVCBENCH_PRNET_SMILES", str(_SMILES_DEFAULT))
    if os.path.exists(path):
        t = pd.read_csv(path)
        smi = {str(r.sm_name): str(r.SMILES) for r in t.itertuples()}
        dose = {str(r.sm_name): float(r.dose_uM) for r in t.itertuples()}
        return smi, dose
    op3 = os.environ.get(
        "IVCBENCH_OP3_PATH",
        str(_REPO / "data/C5/op3/GSE279945_sc_counts_processed.h5ad"),
    )
    if os.path.exists(op3):
        import anndata

        o = anndata.read_h5ad(op3, backed="r").obs
        g = o[["sm_name", "SMILES", "dose_uM"]].copy()
        g["sm_name"] = g["sm_name"].astype(str)
        smi = (
            g.drop_duplicates("sm_name")
            .set_index("sm_name")["SMILES"]
            .astype(str)
            .to_dict()
        )
        dose = (
            g.groupby("sm_name")["dose_uM"]
            .apply(lambda s: float(pd.to_numeric(s, errors="coerce").median()))
            .to_dict()
        )
        return smi, dose
    return {}, {}


def _drug_dose_matrix(compounds, smiles, doses, fp_by_cpd, n_bits, default_dose):
    """PRnet's perturbation encoding, one row per compound.

    Native path = the authors' `Drug_dose_encoder`: rFCFP4 (Morgan r=2, useFeatures=True, nBits) of
    the SMILES, scaled by log10(dose+1). Fallback for a compound with no/unparseable SMILES = the
    harness's own Morgan fingerprint from side_info, scaled identically, so a compound is never
    silently dropped.
    """
    from PRnet_data_utils import Drug_dose_encoder  # injected alias, see _import_prnet

    rows, used_native, missing = [], 0, []
    for c in compounds:
        d = float(doses.get(c, default_dose))
        if not np.isfinite(d) or d <= 0:
            d = default_dose
        smi = smiles.get(c)
        vec = None
        if smi and smi not in ("nan", "None", ""):
            try:
                vec = Drug_dose_encoder([smi], [d], num_Bits=n_bits, comb_num=1)[0]
                used_native += 1
            except Exception:
                vec = None
        if vec is None:
            fp = fp_by_cpd.get(c)
            if fp is None:
                missing.append(c)
                rows.append(np.zeros(n_bits, dtype=np.float32))
                continue
            v = np.asarray(fp, dtype=np.float32).ravel()[:n_bits]
            if v.shape[0] < n_bits:  # keep the width the model was built with
                v = np.pad(v, (0, n_bits - v.shape[0]))
            vec = v * np.log10(d + 1.0)
        rows.append(np.asarray(vec, dtype=np.float32))
    return np.vstack(rows).astype(np.float32), used_native, missing


def _import_prnet():
    repo = os.environ.get("IVCBENCH_PRNET_REPO", str(_VENDOR_DEFAULT))
    if not os.path.isdir(repo):
        raise RuntimeError(
            f"PRnet: vendored repo not found at {repo} "
            "(clone github.com/Perturbation-Response-Prediction/PRnet)"
        )
    sys.path.insert(0, repo)
    import importlib

    du = importlib.import_module("data._utils")
    sys.modules["PRnet_data_utils"] = du  # stable alias for _drug_dose_matrix
    ds = importlib.import_module("data.Dataset")
    tr = importlib.import_module("trainer.PRnetTrainer")
    return repo, ds, tr


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.distributions import normal
    from torch.nn import functional as F

    epochs = int(os.environ.get("IVCBENCH_PRNET_EPOCHS", "100"))
    maxcells = int(os.environ.get("IVCBENCH_PRNET_MAXCELLS", "60000"))
    batch = int(os.environ.get("IVCBENCH_PRNET_BATCH", "512"))
    n_dec = int(os.environ.get("IVCBENCH_PRNET_NDECODE", "512"))
    seed = int(os.environ.get("IVCBENCH_PRNET_SEED", "0"))
    dflt_dose = float(os.environ.get("IVCBENCH_PRNET_DOSE", "1.0"))

    repo, ds_mod, tr_mod = _import_prnet()
    DrugDoseAnnDataset = ds_mod.DrugDoseAnnDataset
    PRnetTrainer = tr_mod.PRnetTrainer

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    pert_train = np.array([str(p) for p in d["pert_train"]], dtype=object)
    is_ctrl = d["is_control_train"].astype(bool)
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {_CONTROL})
    ct_train = (
        np.array([str(c) for c in d["celltype_train"]], dtype=object)
        if "celltype_train" in d.files
        else np.array(["PBMC"] * X.shape[0], dtype=object)
    )
    gem_train = (
        np.array([str(c) for c in d["gem_train"]], dtype=object)
        if "gem_train" in d.files
        else np.array(["d0"] * X.shape[0], dtype=object)
    )
    fp_by_cpd = (
        {
            str(k): np.asarray(v, dtype=np.float32)
            for k, v in zip(d["fingerprint_keys"], d["fingerprint_vals"])
        }
        if "fingerprint_keys" in d.files
        else {}
    )
    if not test_perts:
        raise RuntimeError("PRnet: payload carries no held perturbation label")
    if not is_ctrl.any():
        raise RuntimeError("PRnet: train fold has no control cells to pair against")

    # PRnet's published perturbation width. It is NOT free: DrugDoseAnnDataset calls
    # Drug_dose_encoder with its default num_Bits=1024 when it builds the training tensors, so the
    # Perturb-adaptor must be built at 1024 or train- and predict-time encodings disagree.
    n_bits = 1024
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- cell cap (stratified)
    if X.shape[0] > maxcells:
        lab = np.where(is_ctrl, "__ctrl__", pert_train).astype(str)
        keep = []
        for u in np.unique(lab):
            idx = np.where(lab == u)[0]
            take = max(2, int(round(maxcells * len(idx) / X.shape[0])))
            keep.append(
                idx if take >= len(idx) else rng.choice(idx, take, replace=False)
            )
        sel = np.sort(np.concatenate(keep))
        X, pert_train, is_ctrl, ct_train, gem_train = (
            X[sel],
            pert_train[sel],
            is_ctrl[sel],
            ct_train[sel],
            gem_train[sel],
        )

    train_cpds = sorted(set(pert_train[~is_ctrl].astype(str)))
    if len(train_cpds) < 3:
        raise RuntimeError(
            f"PRnet: only {len(train_cpds)} training compounds; nothing to condition on"
        )

    # ---------------------------------------------------------------- paired controls
    # PRnet conditions each treated cell on ONE unperturbed cell. Pair within (lineage, donor) so the
    # control carries the same context as the target; fall back to lineage, then to any control.
    ctrl_pos = np.where(is_ctrl)[0]
    by_cd, by_c = {}, {}
    for p in ctrl_pos:
        by_cd.setdefault((ct_train[p], gem_train[p]), []).append(p)
        by_c.setdefault(ct_train[p], []).append(p)
    names = np.array([f"c{i}" for i in range(X.shape[0])], dtype=object)
    paired = np.empty(X.shape[0], dtype=object)
    for i in range(X.shape[0]):
        pool = (
            by_cd.get((ct_train[i], gem_train[i]))
            or by_c.get(ct_train[i])
            or list(ctrl_pos)
        )
        paired[i] = names[pool[int(rng.integers(len(pool)))]]

    # ---------------------------------------------------------------- PRnet AnnData
    smiles, doses = _smiles_dose_tables(train_cpds + test_perts)
    dose_col = np.array(
        [
            0.0 if c else float(doses.get(p, dflt_dose))
            for c, p in zip(is_ctrl, pert_train)
        ],
        dtype=float,
    )
    dose_col[(~is_ctrl) & (~np.isfinite(dose_col) | (dose_col <= 0))] = dflt_dose
    smi_col = np.array(
        [
            str(smiles.get(str(p), "C")) if not c else "C"
            for c, p in zip(is_ctrl, pert_train)
        ],
        dtype=object,
    )

    # split column: 'train' / 'valid' on TREATED cells only; controls stay NaN so PRnet's
    # train_valid_test() (which appends every dose==0 cell to each split) never duplicates a row.
    # The validation axis mirrors the axis being held out, exactly as in PRnet's two published
    # protocols: compound-wise for `compound_split` (T5u), cell-wise for `cell_line_split` (T5c,
    # where every compound is seen and excluding compounds from training would be a self-handicap).
    unseen_cpd_mode = not (set(test_perts) & set(train_cpds))
    split_col = np.array([np.nan] * X.shape[0], dtype=object)
    treated = np.where(~is_ctrl)[0]
    if unseen_cpd_mode:
        n_val = max(2, int(round(0.15 * len(train_cpds))))
        val_cpds = set(
            rng.choice(
                np.array(train_cpds, dtype=object),
                size=min(n_val, max(1, len(train_cpds) - 2)),
                replace=False,
            ).tolist()
        )
        for i in treated:
            split_col[i] = "valid" if str(pert_train[i]) in val_cpds else "train"
    else:
        val_cpds = set()
        vmask = rng.random(len(treated)) < 0.15
        for i, v in zip(treated, vmask):
            split_col[i] = "valid" if v else "train"
        if not (split_col == "valid").any():
            split_col[treated[0]] = "valid"
    # BatchNorm1d dies on a trailing batch of 1 -> drop one treated cell if that would happen
    for tag in ("train", "valid"):
        n = int((split_col == tag).sum())
        if n > 1 and n % batch == 1:
            split_col[[i for i in treated if split_col[i] == tag][-1]] = np.nan

    obs = pd.DataFrame(
        {
            "SMILES": smi_col.astype(str),
            "dose": dose_col,
            "cell_type": ct_train.astype(str),
            "cov_drug": np.array(
                [
                    f"{ct}_{'control' if c else p}"
                    for ct, c, p in zip(ct_train, is_ctrl, pert_train)
                ]
            ),
            "paired_control_index": paired.astype(str),
            "prnet_split": split_col,
        },
        index=list(names),
    )
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = genes

    work = Path(tempfile.mkdtemp(prefix="prnet_"))
    (work / "ckpt").mkdir()
    (work / "res").mkdir()
    print(
        f"[PRnet] repo={repo} cells={X.shape[0]} genes={len(genes)} "
        f"train_cpds={len(train_cpds)} test_cpds={len(test_perts)} "
        f"mode={'unseen-compound (compound_split)' if unseen_cpd_mode else 'seen-compound / held context (cell_line_split)'} "
        f"valid={'%d held-out train compounds' % len(val_cpds) if unseen_cpd_mode else '15% of treated cells'} "
        f"n_train_rows={int((split_col=='train').sum())} n_valid_rows={int((split_col=='valid').sum())} "
        f"epochs={epochs} batch={batch}",
        flush=True,
    )

    torch.manual_seed(seed)
    import random as _random

    _random.seed(seed)
    trainer = PRnetTrainer(
        adata,
        batch_size=batch,
        comb_num=1,
        split_key="prnet_split",
        model_save_dir=str(work / "ckpt") + "/",
        results_save_dir=str(work / "res") + "/",
        x_dimension=len(genes),
        hidden_layer_sizes=[128],
        z_dimension=64,
        adaptor_layer_sizes=[128],
        comb_dimension=64,
        drug_dimension=n_bits,
        dr_rate=0.05,
        n_genes=20,
        loss=["GUSS"],
        obs_key="cov_drug",
        seed=seed,
    )

    # (1) PRnet saves `self.modelPGM.module.state_dict()`, which only exists when >1 GPU made it wrap
    #     the net in DataParallel. Wrap unconditionally so the authors' train() runs verbatim on 1 GPU.
    if not isinstance(trainer.modelPGM, nn.DataParallel):
        trainer.modelPGM = nn.DataParallel(trainer.modelPGM)
    # (2) O(1) paired-control lookup (see _FastIndexList).
    for name in ("train_dataset", "valid_dataset", "test_dataset"):
        dset = getattr(trainer, name, None)
        if dset is not None:
            dset.dense_adata_index = _FastIndexList(dset.dense_adata_index)

    trainer.train(
        n_epochs=epochs,
        lr=1e-3,
        weight_decay=1e-8,
        scheduler_factor=0.5,
        scheduler_patience=5,
    )

    ckpt = work / "ckpt" / "prnet_split_best_epoch_all.pt"
    net = trainer.modelPGM.module
    if ckpt.exists():
        net.load_state_dict(torch.load(str(ckpt), map_location=trainer.device))
        print(f"[PRnet] loaded best checkpoint {ckpt}", flush=True)
    net.eval()

    # ---------------------------------------------------------------- predict held compounds
    # Conditioning = held context's OWN control cells + the compound's chemistry. Nothing else.
    C = X_ctrl_inf if X_ctrl_inf.shape[0] else X[is_ctrl]
    if C.shape[0] > n_dec:
        C = C[np.sort(rng.choice(C.shape[0], n_dec, replace=False))]
    ctrl_t = torch.tensor(C, dtype=torch.float32, device=trainer.device)
    DD, n_native, missing = _drug_dose_matrix(
        test_perts, smiles, doses, fp_by_cpd, n_bits, dflt_dose
    )
    predictable = [c for c in test_perts if c not in set(missing)]
    print(
        f"[PRnet] decoding {len(predictable)}/{len(test_perts)} held compounds from "
        f"{C.shape[0]} control cells (native rFCFP4 for {n_native}; "
        f"no chemistry for {len(missing)}: {missing[:3]})",
        flush=True,
    )
    if not predictable:
        raise RuntimeError(
            "PRnet: no held compound has a usable chemistry representation"
        )

    G = len(genes)
    pred_perts, pred_means = [], []
    with torch.no_grad():
        for c, row in zip(test_perts, DD):
            if c in set(missing):
                continue
            lab = torch.tensor(
                np.repeat(row[None, :], ctrl_t.shape[0], axis=0),
                dtype=torch.float32,
                device=trainer.device,
            )
            noise = torch.randn(ctrl_t.shape[0], 10, device=trainer.device)
            rec = net(ctrl_t, lab, noise)
            mu, var = rec[:, :G], F.softplus(rec[:, G:])
            dist = normal.Normal(
                torch.clamp(mu, 1e-3, 1e3), torch.clamp(var.sqrt(), 1e-3, 1e3)
            )
            pred_perts.append(c)
            pred_means.append(dist.sample().mean(0).cpu().numpy().astype(np.float32))

    np.savez(
        out_path,
        pred_perts=np.array(pred_perts, dtype=object),
        pred_means=np.vstack(pred_means).astype(np.float32),
    )
    print(
        f"[PRnet] wrote {len(pred_perts)} profiles of {G} genes -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
