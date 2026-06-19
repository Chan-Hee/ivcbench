#!/usr/bin/env python
"""chemCPA collapse characterization on the OP3 unseen-compound split.

Computes — entirely from the DEPOSITED data (no fabricated numbers) — how chemCPA collapses:

  (1) Predicted between-compound variance: from the deposited per-seed prediction npz
      (outputs/additional_models/chemcpa_native_seed{0,1,2}.npz), the per-compound predicted
      profiles, and (a) their pairwise Pearson r, (b) per-gene between-compound std/var, (c) the
      between-compound variance of the predicted Δ profiles (pred - control_mean).

  (2) OBSERVED between-compound variance: reconstruct the SAME OP3 loader + leak-safe
      C5_global_compound_holdout split + 2000-HVG panel as every deposited entrant, then take each
      held compound's OBSERVED mean treated profile over its test cells, and the OBSERVED Δ profile
      (obs_treated_mean - control_mean). Between-compound variance of those is the signal a faithful
      chemistry channel would have to reproduce.

  (3) Collapse ratio = predicted between-compound variance / observed between-compound variance
      (in Δ space). A degenerate-prediction signature is ratio << 1 with the model still training
      cleanly (it is not a crash) — i.e. the chemistry channel carries no usable held-out signal.

All control-mean / split / gene-panel conventions are IDENTICAL to scripts/chemcpa_evaluate.py
and src/ivcbench/runner/run.py.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(REPO, "src"))
OUT = os.path.join(REPO, "outputs/additional_models")


def main():
    from ivcbench.clusters import c5
    from ivcbench.clusters.spec import _c5_held_compounds
    from ivcbench.data.loaders.op3 import load
    from ivcbench.splits.audit import audit_split
    from ivcbench.splits.builder import build_split

    # ---------- reconstruct loader + split (identical to deposited entrants) ----------
    cs = load()
    held = _c5_held_compounds(cs)
    spec = c5.global_compound_holdout(held)
    split = build_split(cs, spec)
    audit = audit_split(cs, split)
    assert audit["leak_free"], "leak gate failed"

    obs = cs.obs.reset_index(drop=True)
    genes = np.array([str(g) for g in cs.var_names])
    test_idx = split.test_idx
    test_X = cs.X[test_idx]
    pert_test = obs["perturbation"].astype(str).to_numpy()[test_idx]

    # control mean for Δ — the split's matched inference-input controls (same as run.py / evaluate)
    inf_pos = split.inference_input_idx
    ctrl_mean = (cs.X[inf_pos].mean(0) if len(inf_pos) else test_X.mean(0)).astype(np.float64)

    # ---------- OBSERVED per-compound mean treated profile + Δ ----------
    held_set = set(held)
    obs_compounds = sorted(set(pert_test) & held_set)
    obs_prof = {}
    for c in obs_compounds:
        m = pert_test == c
        obs_prof[c] = test_X[m].mean(0).astype(np.float64)
    OBS = np.vstack([obs_prof[c] for c in obs_compounds])           # (n_held, n_genes)
    OBS_delta = OBS - ctrl_mean[None, :]

    # ---------- PREDICTED per-compound profiles (per seed + seed-averaged) ----------
    seeds = [0, 1, 2]
    pred_by_seed = {}
    pred_compounds_ref = None
    for s in seeds:
        f = os.path.join(OUT, f"chemcpa_native_seed{s}.npz")
        d = np.load(f, allow_pickle=True)
        comps = [str(x) for x in d["pred_compounds"]]
        order = np.argsort(comps)
        comps_sorted = list(np.array(comps)[order])
        P = d["pred_means"][order].astype(np.float64)               # (n_held, n_genes), sorted by compound
        # check predicted genes == loader genes (same panel/order)
        pg = np.array([str(x) for x in d["genes"]])
        same_panel = bool(np.array_equal(pg, genes))
        pred_by_seed[s] = (comps_sorted, P, same_panel)
        if pred_compounds_ref is None:
            pred_compounds_ref = comps_sorted

    # seed-averaged predicted profiles, aligned to the OBSERVED compound ordering
    pred_avg_map = {}
    for c in pred_compounds_ref:
        stack = []
        for s in seeds:
            cs_list, P, _ = pred_by_seed[s]
            i = cs_list.index(c)
            stack.append(P[i])
        pred_avg_map[c] = np.mean(np.vstack(stack), 0)
    common = [c for c in obs_compounds if c in pred_avg_map]
    PRED = np.vstack([pred_avg_map[c] for c in common])             # (n_common, n_genes)
    PRED_delta = PRED - ctrl_mean[None, :]
    OBS_c = np.vstack([obs_prof[c] for c in common])
    OBS_c_delta = OBS_c - ctrl_mean[None, :]

    # ================= statistics =================
    def between_compound_var(M):
        """mean over genes of the across-compound variance (ddof=1)."""
        return float(M.var(axis=0, ddof=1).mean())

    def pairwise_r(M):
        C = np.corrcoef(M)
        iu = np.triu_indices(M.shape[0], k=1)
        return C[iu]

    stats = {"n_held_observed": len(obs_compounds),
             "n_common": len(common),
             "predicted_genes_match_loader_panel": all(pred_by_seed[s][2] for s in seeds)}

    # --- per-seed predicted collapse (raw profile space) ---
    per_seed = {}
    for s in seeds:
        cs_list, P, _ = pred_by_seed[s]
        pwr = pairwise_r(P)
        per_seed[s] = {
            "between_compound_var_pred_profile": between_compound_var(P),
            "between_compound_std_pred_profile_mean_over_genes": float(P.std(axis=0, ddof=1).mean()),
            "pairwise_pearson_r_min": float(pwr.min()),
            "pairwise_pearson_r_mean": float(pwr.mean()),
            "pairwise_pearson_r_max": float(pwr.max()),
        }
    stats["per_seed_predicted_collapse"] = per_seed

    # --- seed-averaged Δ-space comparison (the metric space) ---
    pwr_pred = pairwise_r(PRED)
    var_pred_delta = between_compound_var(PRED_delta)
    var_obs_delta = between_compound_var(OBS_c_delta)
    var_pred_prof = between_compound_var(PRED)
    var_obs_prof = between_compound_var(OBS_c)
    pwr_obs = pairwise_r(OBS_c_delta)

    stats["seed_averaged"] = {
        "between_compound_var_PRED_profile": var_pred_prof,
        "between_compound_var_OBS_profile": var_obs_prof,
        "between_compound_var_PRED_delta": var_pred_delta,
        "between_compound_var_OBS_delta": var_obs_delta,
        "collapse_ratio_delta_pred_over_obs": var_pred_delta / var_obs_delta if var_obs_delta else float("inf"),
        "collapse_ratio_profile_pred_over_obs": var_pred_prof / var_obs_prof if var_obs_prof else float("inf"),
        "pred_delta_pairwise_r_min": float(pwr_pred.min()),
        "pred_delta_pairwise_r_mean": float(pwr_pred.mean()),
        "obs_delta_pairwise_r_min": float(pwr_obs.min()),
        "obs_delta_pairwise_r_mean": float(pwr_obs.mean()),
    }

    # --- total-variance partition (fraction of variance that is between-compound) ---
    # predicted: between-compound RMS deviation vs grand-mean profile magnitude
    def dispersion(M):
        gm = M.mean(0)
        dev = M - gm[None, :]
        rms_dev = float(np.sqrt((dev ** 2).mean()))
        rms_gm = float(np.sqrt((gm ** 2).mean()))
        return rms_dev, rms_gm, (rms_dev / rms_gm if rms_gm else float("inf"))
    stats["seed_averaged"]["pred_profile_dispersion"] = dict(zip(
        ["rms_between_compound_dev", "rms_grand_mean", "ratio"], dispersion(PRED)))
    stats["seed_averaged"]["obs_profile_dispersion"] = dict(zip(
        ["rms_between_compound_dev", "rms_grand_mean", "ratio"], dispersion(OBS_c)))

    print(json.dumps(stats, indent=2))

    # save arrays for the figure
    np.savez(os.path.join(OUT, "chemcpa_collapse_arrays.npz"),
             compounds=np.array(common, dtype=object),
             pred_profile=PRED, obs_profile=OBS_c,
             pred_delta=PRED_delta, obs_delta=OBS_c_delta,
             ctrl_mean=ctrl_mean, genes=genes)
    with open(os.path.join(OUT, "chemcpa_collapse_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\n[saved] {OUT}/chemcpa_collapse_arrays.npz and chemcpa_collapse_stats.json")


if __name__ == "__main__":
    main()
