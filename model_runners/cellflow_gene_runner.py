#!/usr/bin/env python
"""CellFlow runner for the held-GENE tasks — `ivc-cellflow` env.

Invoked by ivcbench.baselines.heavy.CellFlowGene:
    <ivc-cellflow python> cellflow_gene_runner.py <in.npz> <out.npz>

Covers T3 (primary-T CRISPR, held GENES absent from training) and T4 (Frangieh CRISPR, held KNOCKOUT,
RNA and protein readouts). Here the held entity is the PERTURBATION itself, so the categorical entry
point used by `cellflow_c1_runner.py` is undefined: a held gene is a value the one-hot encoder has
never seen. CellFlow's own construct for exactly this is `perturbation_covariate_reps` — the
perturbation enters as a *representation*, so an unseen perturbation is an unseen point of an input
space the model already consumes. That is the same mechanism the compound runner uses for an unseen
drug (`cellflow_c5_runner.py`, fingerprints in `adata.uns`); only the representation changes.

GENE-SIDE REPRESENTATION. The runner takes `gene_embedding_keys`/`gene_embedding_vals` from the
payload when the split supplies them (`side_info['gene_embedding']`). No loader in this repository
populates that key, so — exactly as `scgen_runner.py` and `scripts/run_c4_conditioned.py`
(`LinearShiftKOEmb`) do for the same two tasks — the runner otherwise builds the harness's standard
LEAK-SAFE gene-side embedding itself: PCA gene-loadings fitted on CONTROL cells only, so the
representation of a held gene carries no perturbation information and no held-gene response. Using
the identical construction keeps CellFlow's gene-side input the same as the other conditioned
entrants on this axis. A perturbed gene that is not a feature of the expression panel has no loading
vector; the runner declines it, and heavy.py keeps the control mean for that label — the defined
behaviour when a runner declines an unseen target. Coverage is printed.

ONE PROFILE PER HELD GENE: the model is queried once per held gene with that gene's own
representation, so `pred_key_is_group` stays False. The runner prints the across-gene spread of the
predicted profiles so a collapse onto a single perturbation-agnostic profile (the defect found on
this axis for other entrants) is visible in the log rather than only in the score.

Leak-safe: fit on `X_train` only; the held genes' cells are absent from the training fold by
construction, and the source cells are the split's `inference_input_idx` control cells.

Knobs (env): IVCBENCH_CELLFLOW_{ITERS,BATCH,NPCA,MAXCELLS,NCTRL,COND_DIM,HIDDEN,GENE_EMB}.
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

PERT_COL = "perturbation"
COND_GROUP = "gene"
REPS_KEY = "gene_reps"


def _gene_embedding(d, X, is_ctrl, X_ctrl_inf, genes):
    """-> ({gene -> vector}, source tag). Payload representation if present, else control-only PCA."""
    if "gene_embedding_keys" in d.files and "gene_embedding_vals" in d.files:
        emb = {
            str(kk): np.asarray(v, dtype=np.float32).ravel()
            for kk, v in zip(d["gene_embedding_keys"], d["gene_embedding_vals"])
        }
        if emb:
            return emb, "side_info['gene_embedding']"
    from sklearn.decomposition import PCA

    n_emb = int(os.environ.get("IVCBENCH_CELLFLOW_GENE_EMB", "50"))
    ctrl_X = X[is_ctrl] if is_ctrl.any() else X_ctrl_inf
    k = int(min(n_emb, ctrl_X.shape[0] - 1, ctrl_X.shape[1]))
    gpca = PCA(n_components=max(2, k), random_state=0).fit(ctrl_X)
    loadings = gpca.components_.T  # (n_genes, k): gene g's loading vector
    return (
        {g: loadings[i].astype(np.float32) for i, g in enumerate(genes)},
        "control-only PCA gene-loadings (leak-safe, built in-runner)",
    )


def build_inputs(in_path: str, centered_pca, project_pca):
    """Payload -> (train AnnData in PCA space, control-source AnnData, held genes, genes)."""
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
    if not test_perts:
        raise RuntimeError("CellFlow-gene: payload has no held perturbation label")
    if not is_ctrl.any():
        raise RuntimeError(
            "CellFlow-gene: training fold has no control cells to flow from"
        )

    emb, emb_src = _gene_embedding(d, X, is_ctrl, X_ctrl_inf, genes)
    emb_dim = len(next(iter(emb.values())))

    pred_genes = [g for g in test_perts if g in emb]
    if not pred_genes:
        raise RuntimeError(
            f"CellFlow-gene: none of the {len(test_perts)} held genes has a gene-side "
            f"representation ({emb_src}); the model declines every target."
        )

    # ---- training AnnData ----------------------------------------------------------------------
    pert = np.where(is_ctrl, "control", pert_train).astype(str)
    # a training gene with no representation cannot be encoded -> drop it (never a target here)
    ok = np.array([p == "control" or p in emb for p in pert])
    X, pert = X[ok], pert[ok]
    sel = stratified_cap(X, pert, k["max_cells"])
    X, pert = X[sel], pert[sel]

    adata_tr = ad.AnnData(
        X=np.asarray(X, dtype=np.float32),
        obs=pd.DataFrame(
            {PERT_COL: pert, "is_control": pert == "control"},
            index=[f"tr{i}" for i in range(X.shape[0])],
        ),
    )
    adata_tr.var_names = genes
    adata_tr.obs["is_control"] = adata_tr.obs["is_control"].astype("boolean")
    reps = {g: emb[g] for g in sorted(set(pert) - {"control"})}
    for g in pred_genes:  # held genes: representation only, no cells
        reps[g] = emb[g]
    reps["control"] = np.zeros(emb_dim, dtype=np.float32)
    adata_tr.uns[REPS_KEY] = reps

    # ---- control AnnData used as the SOURCE distribution at inference ---------------------------
    if X_ctrl_inf.shape[0] > k["n_ctrl"]:
        X_ctrl_inf = X_ctrl_inf[
            np.random.default_rng(0).choice(
                X_ctrl_inf.shape[0], k["n_ctrl"], replace=False
            )
        ]
    adata_ct = ad.AnnData(
        X=np.asarray(X_ctrl_inf, dtype=np.float32),
        obs=pd.DataFrame(
            {PERT_COL: ["control"] * X_ctrl_inf.shape[0], "is_control": True},
            index=[f"ct{i}" for i in range(X_ctrl_inf.shape[0])],
        ),
    )
    adata_ct.var_names = genes
    adata_ct.obs["is_control"] = adata_ct.obs["is_control"].astype("boolean")
    adata_ct.uns[REPS_KEY] = reps

    # ---- PCA space fitted on the TRAIN fold only (leak-safe), authors' own helpers --------------
    n_pca = int(min(k["n_pca"], adata_tr.n_obs - 1, adata_tr.n_vars - 1))
    centered_pca(adata_tr, n_comps=n_pca, method="scanpy", keep_centered_data=False)
    project_pca(query_adata=adata_ct, ref_adata=adata_tr)

    print(
        f"[CellFlow-gene] train={adata_tr.n_obs} cells x {adata_tr.n_vars} genes |"
        f" train genes={len(set(pert)) - 1} | held genes with a"
        f" representation={len(pred_genes)}/{len(test_perts)} | gene-side repr:"
        f" {emb_src} (dim {emb_dim}) | ctrl-source={adata_ct.n_obs} | PCA={n_pca}",
        flush=True,
    )
    return adata_tr, adata_ct, pred_genes, genes


def main(in_path: str, out_path: str) -> None:
    import pandas as pd
    from cellflow.model import CellFlow
    from cellflow.preprocessing import centered_pca, project_pca

    k = knobs()
    adata_tr, adata_ct, pred_genes, genes = build_inputs(
        in_path, centered_pca, project_pca
    )
    print(
        "[CellFlow-gene]"
        f" iters={k['iters']} batch={k['batch']} cond_dim={k['cond_dim']} "
        f"hidden={k['hidden']}",
        flush=True,
    )

    cf = CellFlow(adata_tr, solver="otfm")
    cf.prepare_data(
        sample_rep="X_pca",
        control_key="is_control",
        perturbation_covariates={COND_GROUP: (PERT_COL,)},
        perturbation_covariate_reps={COND_GROUP: REPS_KEY},
        max_combination_length=1,
        null_value=0.0,
    )
    prepare_model(cf, k["cond_dim"], k["hidden"], COND_GROUP)
    cf.train(num_iterations=k["iters"], batch_size=k["batch"])

    cov = pd.DataFrame(
        {
            PERT_COL: pred_genes,
            "condition_id": pred_genes,
            "is_control": [True] * len(pred_genes),
        }
    )
    cov["is_control"] = cov["is_control"].astype("boolean")
    preds = cf.predict(
        adata=adata_ct,
        covariate_data=cov,
        sample_rep="X_pca",
        condition_id_key="condition_id",
    )

    pred_perts, pred_means = [], []
    for g in pred_genes:
        if g not in preds:
            continue
        pred_perts.append(g)
        pred_means.append(decode_mean(preds[g], genes, adata_tr))
    if not pred_perts:
        raise RuntimeError(
            "CellFlow-gene: predict() returned no profile for any held gene"
        )

    M = np.vstack(pred_means).astype(np.float32)
    # perturbation specificity: mean over genes of the SD across held perturbations. A value at 0
    # would mean one profile for every held gene (a perturbation-agnostic collapse), which is the
    # failure this axis has produced for other entrants; report it so it is visible in the log.
    spec = float(np.mean(M.std(0))) if M.shape[0] > 1 else float("nan")
    np.savez(out_path, pred_perts=np.array(pred_perts, dtype=object), pred_means=M)
    print(
        f"[CellFlow-gene] wrote {len(pred_perts)} predicted profiles "
        f"(across-held-gene SD per gene, mean = {spec:.5f}) -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
