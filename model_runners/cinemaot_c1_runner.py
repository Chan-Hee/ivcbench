#!/usr/bin/env python
"""CINEMA-OT on a held-GROUP split (T1 lineage, T2 donor) — `scperturbench_eval` env.

Invoked by ivcbench.baselines.heavy.CINEMAOTC1: <env python> cinemaot_c1_runner.py <in.npz> <out.npz>

WHY THIS RUNNER EXISTS. `cinemaot_runner.py` pools every training perturbation into one "treated"
token, averages the returned effect into one global shift, and tiles `control_mean + shift` over
every held label. On the unseen-entity tasks that is a deliberate perturbation-agnostic diagnostic.
On T1/T2 it is simply the wrong call: the stimulus is SEEN (Kang carries exactly two perturbation
values, control and IFN-beta) and the held axis is a group, so the published operation applies
directly and no reduction is needed.

THE PUBLISHED OPERATION, read from the installed official package
(site-packages/cinemaot/cinemaot.py, vandijklab/CINEMA-OT):

    cinemaot_unweighted(adata, obs_label, ref_label, expr_label, ...) -> (cf, ot_matrix, TE)
    :123   te2 = X[obs==ref_label] - (ot_matrix / rowsum) @ X[obs==expr_label]
    :146   TE  = AnnData(te2, obs=adata[obs==ref_label].obs)        # CONTROL-indexed
    :63    "Single-cell differential expression for each cell in control condition,
            of shape (n_refcells, n_genes)"

So for every control cell i, `X_ref[i] - te2[i]` is exactly `(ot_matrix/rowsum) @ X[expr]` -- the
OT-barycentric counterfactual treated profile of that cell, straight out of a published return
value. Averaging it over the held group's own control cells is the group's predicted response.
`expr_label` is a named-condition selector, the direct analogue of scPRAM's `stim_key`.

LEAK BOUNDARY. The control arm is the held group's OWN control cells (payload X_ctrl_inf); the
treated arm is the stimulated cells of the TRAINING groups only. No held-group treated cell enters.
"""
from __future__ import annotations

import os
import sys

import numpy as np

CTRL = "control"


def main(in_path: str, out_path: str) -> None:
    import anndata as ad
    import scanpy as sc
    from cinemaot.cinemaot import cinemaot_unweighted

    d = np.load(in_path, allow_pickle=True)
    X = d["X_train"].astype(np.float32)
    genes = [str(g) for g in d["genes"]]
    is_ctrl = d["is_control_train"].astype(bool)
    pert_train = np.asarray([str(p) for p in d["pert_train"]])
    X_ctrl_inf = d["X_ctrl_inf"].astype(np.float32)
    test_perts = sorted({str(p) for p in d["test_perts"]} - {CTRL})

    cap = int(os.environ.get("IVCBENCH_CINEMAOT_MAXCELLS", "2000"))
    n_pca = int(os.environ.get("IVCBENCH_CINEMAOT_PCA", "20"))
    rng = np.random.default_rng(0)

    if not test_perts:
        raise RuntimeError("CINEMA-OT-C1: no held perturbation label in the payload")
    # The control arm carries the TRAINING controls as well as the held unit's own, and only the
    # held rows are averaged at the end.
    #
    # WHY. With the held unit's controls as the whole reference and the mean taken over all of
    # them, a uniform balanced OT makes the answer an identity: P's row marginals are 1/n and its
    # column marginals 1/m, so
    #     mean_rows( rownorm(P) @ X_treated ) == mean_rows( X_treated )
    # and the prediction collapses to that condition's TRAINING TREATED MEAN, carrying no
    # information about the held unit at all. Measured on the first build of this runner: across
    # the eight held lineages of T1 the predictions correlated at 0.995, against 0.996 for the
    # cell-mean floor and 0.84-0.91 for scGen / CellOT / scPRAM, which do read the held controls.
    # Mixing the training controls into the reference and then averaging only the held rows breaks
    # the identity -- the held cells now have their own transport rows.
    #
    # The held unit's TREATED cells are still absent from every arm, so the leak boundary is
    # unchanged; this is transductive inference over controls the split already allows as the
    # inference input, not a re-labelling of held cells as training data.
    train_ctrl = X[is_ctrl]
    if train_ctrl.shape[0] > cap:
        train_ctrl = train_ctrl[rng.choice(train_ctrl.shape[0], cap, replace=False)]
    held_ref = X_ctrl_inf
    if held_ref.shape[0] > cap:
        held_ref = held_ref[rng.choice(held_ref.shape[0], cap, replace=False)]
    ref = np.vstack([train_ctrl, held_ref]).astype(np.float32)
    held_rows = np.zeros(ref.shape[0], dtype=bool)
    held_rows[train_ctrl.shape[0]:] = True
    print(
        f"[CINEMA-OT-C1] reference = {train_ctrl.shape[0]} training controls "
        f"+ {held_ref.shape[0]} held-unit controls; only the held rows are averaged",
        flush=True,
    )

    labels, profiles, declined = [], [], []
    for stim in test_perts:
        m = (~is_ctrl) & (pert_train == stim)
        if int(m.sum()) < 5:
            declined.append(f"{stim}(train n={int(m.sum())})")
            continue
        Xt = X[m]
        if Xt.shape[0] > cap:
            Xt = Xt[rng.choice(Xt.shape[0], cap, replace=False)]

        adata = ad.AnnData(np.vstack([ref, Xt]).astype(np.float32))
        adata.var_names = genes
        adata.obs["condition"] = [CTRL] * ref.shape[0] + [stim] * Xt.shape[0]
        adata.obs["condition"] = adata.obs["condition"].astype("category")
        dim = int(min(n_pca, adata.n_obs - 1, adata.n_vars - 1))
        sc.pp.pca(adata, n_comps=max(2, dim))

        _cf, _ot, TE = cinemaot_unweighted(
            adata, obs_label="condition", ref_label=CTRL, expr_label=stim, dim=dim
        )
        te2 = np.asarray(TE.X, dtype=np.float64)
        if te2.shape != ref.shape:
            declined.append(f"{stim}(effect shape {te2.shape} != {ref.shape})")
            continue
        # X_ref - te2 IS the barycentric counterfactual of each control cell (cinemaot.py:123)
        counterfactual = ref.astype(np.float64) - te2
        if not np.all(np.isfinite(counterfactual[held_rows])):
            declined.append(f"{stim}(non-finite transport)")
            continue
        labels.append(stim)
        # only the held unit's own control cells -- averaging the whole reference restores the
        # identity described above
        profiles.append(counterfactual[held_rows].mean(0).astype(np.float32))
        print(f"  [CINEMA-OT-C1] {stim}: ref={ref.shape[0]} expr={Xt.shape[0]}", flush=True)

    if declined:
        print(
            "[decline] CINEMA-OT-C1: " + ", ".join(declined),
            file=sys.stderr, flush=True,
        )
    if not labels:
        raise RuntimeError("CINEMA-OT-C1: no held condition could be fitted")
    np.savez(
        out_path,
        pred_perts=np.array(labels, dtype=object),
        pred_means=np.vstack(profiles).astype(np.float32),
    )
    print(f"[CINEMA-OT-C1] wrote {len(labels)} profile(s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
