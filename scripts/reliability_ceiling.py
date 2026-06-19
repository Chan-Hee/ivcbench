"""Noise / reliability ceiling per benchmark cluster.

For each cluster (C1 Kang, C2 Soskic, C3 CRISPR/Shifrut, C4 Frangieh, C5 OP3) we estimate how
reliable the OBSERVED perturbation effect itself is — the ceiling any model could reach on the
benchmark's own Axis-1 metric (Pearson-Delta of the effect vector across genes).

METHOD (split-half pseudo-replicate reliability, matches metrics/response.pearson_delta):
  The benchmark scores a model by correlating, per evaluation UNIT (stratum), the predicted effect
  vector  delta = mean(treated cells in unit) - control_mean  against the observed one, across genes.
  The observed effect is itself a noisy estimate from a finite number of cells. We quantify that noise
  by splitting each unit's treated cells into two random halves A,B, computing the observed effect on
  each half against the SAME control mean, and correlating the two halves across genes (exactly the
  benchmark's Pearson, centred across genes). That split-half r is a pseudo-replicate reliability of
  the observed effect. We repeat over many random partitions and average; macro-average over units to
  get the cluster ceiling, with a unit-bootstrap 95% CI. Genes excluded from the benchmark's score
  (C3/C4 downstream-only: the held KO target gene) are excluded here too.

  A Spearman-Brown full-length correction (2r/(1+r)) gives the reliability of the FULL-sample effect
  (both halves pooled) — that is the true ceiling for a model that sees all the cells; we report both
  the raw split-half (half-sample) and the SB-corrected (full-sample) values.

ALL numbers come from the actual deposited data under benchmark/ via the real loaders. No fabrication.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from ivcbench.data.loaders import kang, soskic, shifrut, frangieh, op3  # noqa: E402
from ivcbench.data.schema import CONTROL_TOKEN  # noqa: E402

N_PARTITIONS = 200       # random half-splits per unit, averaged
MIN_TREATED_PER_UNIT = 6  # need >=3 cells per half for a meaningful mean
RNG = np.random.default_rng(0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Centred-across-genes Pearson, identical convention to metrics/response._pearson."""
    a = a - a.mean(); b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-12:
        return np.nan
    return float(np.dot(a, b) / den)


def split_half_reliability(X_treated: np.ndarray, control_mean: np.ndarray,
                           keep_genes: np.ndarray, n_part: int, rng) -> float:
    """Mean over n_part random half-splits of corr( deltaA , deltaB ) across kept genes."""
    n = X_treated.shape[0]
    if n < MIN_TREATED_PER_UNIT:
        return np.nan
    h = n // 2
    rs = []
    for _ in range(n_part):
        perm = rng.permutation(n)
        a = X_treated[perm[:h]].mean(0) - control_mean
        b = X_treated[perm[h:2 * h]].mean(0) - control_mean
        r = _pearson(a[keep_genes], b[keep_genes])
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else np.nan


def spearman_brown(r: float) -> float:
    """Half-sample reliability -> full-sample reliability (both halves pooled)."""
    if not np.isfinite(r):
        return np.nan
    return 2 * r / (1 + r) if (1 + r) != 0 else np.nan


def summarise(per_unit: dict[str, float], cluster: str, task: str, n_boot: int = 5000):
    vals = np.array([v for v in per_unit.values() if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return None
    macro = float(np.mean(vals))
    sb = spearman_brown(macro)
    # unit-bootstrap 95% CI on the macro-average split-half r
    rng = np.random.default_rng(123)
    boots = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {
        "cluster": cluster, "task": task, "n_units": int(len(vals)),
        "reliability_halfsample": round(macro, 4),
        "reliability_fullsample_SB": round(sb, 4),
        "ci_lo_halfsample": round(lo, 4), "ci_hi_halfsample": round(hi, 4),
        "median_unit_r": round(float(np.median(vals)), 4),
        "min_unit_r": round(float(np.min(vals)), 4), "max_unit_r": round(float(np.max(vals)), 4),
    }


# ----------------------------------------------------------------------------------------------
# C1 — Kang GSE96583 IFN-beta. Headline task: LOCT (predict held lineage's IFN-beta response).
#   Unit = (cell_type_coarse, donor_id) IFN-beta response group; control_mean = SAME-lineage control
#   cells (matched-context control, the C1 LOCT inference input is the held lineage's own controls).
# ----------------------------------------------------------------------------------------------
def c1_kang():
    cs = kang.load(subsample_per_group=80, n_hvg=2000, seed=0)
    obs = cs.obs
    X = cs.X
    is_ctrl = obs["is_control"].to_numpy()
    pert = obs["perturbation"].to_numpy()
    ct = obs["cell_type_coarse"].to_numpy()
    donor = obs["donor_id"].to_numpy()
    keep_genes = np.arange(X.shape[1])  # no gene excluded for cytokine clusters
    per = {}
    for lin in np.unique(ct):
        ctrl_lin = X[(ct == lin) & is_ctrl]
        if len(ctrl_lin) == 0:
            continue
        cmean = ctrl_lin.mean(0)  # matched-context control (held lineage's own controls)
        for d in np.unique(donor[ct == lin]):
            m = (ct == lin) & (donor == d) & (~is_ctrl) & (pert != CONTROL_TOKEN)
            Xt = X[m]
            r = split_half_reliability(Xt, cmean, keep_genes, N_PARTITIONS, RNG)
            if np.isfinite(r):
                per[f"{lin}|{d}"] = r
    return per, summarise(per, "C1", "cytokine/Kang LOCT")


# ----------------------------------------------------------------------------------------------
# C2 — Soskic CD4 0h/16h. Headline task: LODO (predict held donor's 16h response).
#   Unit = (donor, cell_type_coarse) 16h-stim group; control_mean = same-donor 0h resting cells
#   (the LODO inference input). Matches metrics exactly. ALSO report deposited bootstrap fallback.
# ----------------------------------------------------------------------------------------------
def c2_soskic():
    cs = soskic.load(cap_per_donor_cond=300, seed=0)
    obs = cs.obs; X = cs.X
    is_ctrl = obs["is_control"].to_numpy()
    donor = obs["donor_id"].to_numpy()
    ct = obs["cell_type_coarse"].to_numpy()
    keep_genes = np.arange(X.shape[1])
    per = {}
    for d in np.unique(donor):
        for c in np.unique(ct[donor == d]):
            ctrl = X[(donor == d) & (ct == c) & is_ctrl]
            treat = X[(donor == d) & (ct == c) & (~is_ctrl)]
            if len(ctrl) == 0:
                # fall back to all same-celltype 0h if this donor lacks 0h of this celltype
                ctrl = X[(ct == c) & is_ctrl]
            if len(ctrl) == 0:
                continue
            r = split_half_reliability(treat, ctrl.mean(0), keep_genes, N_PARTITIONS, RNG)
            if np.isfinite(r):
                per[f"{d}|{c}"] = r
    return per, summarise(per, "C2", "donor/Soskic LODO")


# ----------------------------------------------------------------------------------------------
# C3 — Shifrut GSE119450 CRISPR-KO (the C1-status real anchor for the gene axis). Headline task:
#   leave-one-gene-out; downstream-only metric (held KO target gene excluded). Unit = KO gene's
#   treated cells; control_mean = NTC/control cells (matched context). Exclude the target gene.
# ----------------------------------------------------------------------------------------------
def c3_shifrut():
    cs = shifrut.load()
    obs = cs.obs; X = cs.X
    var = list(cs.var_names)
    gpos = {g: i for i, g in enumerate(var)}
    is_ctrl = obs["is_control"].to_numpy()
    pert = obs["perturbation"].to_numpy()
    cmean_all = X[is_ctrl].mean(0)
    all_genes = np.arange(X.shape[1])
    per = {}
    for g in np.unique(pert):
        if g == CONTROL_TOKEN:
            continue
        Xt = X[(pert == g) & (~is_ctrl)]
        keep = all_genes
        if g in gpos:  # downstream-only: drop the perturbed target gene
            keep = all_genes[all_genes != gpos[g]]
        r = split_half_reliability(Xt, cmean_all, keep, N_PARTITIONS, RNG)
        if np.isfinite(r):
            per[g] = r
    return per, summarise(per, "C3", "gene/CRISPR-Shifrut LO-gene")


# ----------------------------------------------------------------------------------------------
# C4 — Frangieh RNA. Headline: unseen-KO modality axis; downstream-only (held KO gene excluded).
#   Unit = KO gene's treated cells (IFN-gamma condition, the canonical Frangieh context);
#   control_mean = control cells. Exclude target gene.
# ----------------------------------------------------------------------------------------------
def c4_frangieh():
    cs = frangieh.load(modality="rna", condition="IFNγ", subsample_per_group=60, n_hvg=2000, seed=0)
    obs = cs.obs; X = cs.X
    var = list(cs.var_names)
    gpos = {g: i for i, g in enumerate(var)}
    is_ctrl = obs["is_control"].to_numpy()
    pert = obs["perturbation"].to_numpy()
    cmean_all = X[is_ctrl].mean(0)
    all_genes = np.arange(X.shape[1])
    per = {}
    for g in np.unique(pert):
        if g == CONTROL_TOKEN:
            continue
        Xt = X[(pert == g) & (~is_ctrl)]
        keep = all_genes if g not in gpos else all_genes[all_genes != gpos[g]]
        r = split_half_reliability(Xt, cmean_all, keep, N_PARTITIONS, RNG)
        if np.isfinite(r):
            per[g] = r
    return per, summarise(per, "C4", "complex/Frangieh unseen-KO (RNA)")


# ----------------------------------------------------------------------------------------------
# C5 — OP3 GSE279945. Headline: unseen-compound. Unit = (compound, cell_type_coarse) treated group;
#   control_mean = matched-context (same-celltype) DMSO/control cells.
# ----------------------------------------------------------------------------------------------
def c5_op3():
    cs = op3.load(subsample_per_group=40, n_hvg=2000, seed=0)
    obs = cs.obs; X = cs.X
    is_ctrl = obs["is_control"].to_numpy()
    pert = obs["perturbation"].to_numpy()
    ct = obs["cell_type_coarse"].to_numpy()
    keep_genes = np.arange(X.shape[1])
    # matched-context control mean per cell type
    cmean_by_ct = {}
    for c in np.unique(ct):
        cc = X[(ct == c) & is_ctrl]
        if len(cc):
            cmean_by_ct[c] = cc.mean(0)
    glob_ctrl = X[is_ctrl].mean(0)
    per = {}
    for cpd in np.unique(pert):
        if cpd == CONTROL_TOKEN:
            continue
        for c in np.unique(ct[pert == cpd]):
            Xt = X[(pert == cpd) & (ct == c) & (~is_ctrl)]
            cmean = cmean_by_ct.get(c, glob_ctrl)
            r = split_half_reliability(Xt, cmean, keep_genes, N_PARTITIONS, RNG)
            if np.isfinite(r):
                per[f"{cpd}|{c}"] = r
    return per, summarise(per, "C5", "small-mol/OP3 unseen-compound")


def main():
    out = {}
    summaries = []
    for name, fn in [("C1", c1_kang), ("C2", c2_soskic), ("C3", c3_shifrut),
                     ("C4", c4_frangieh), ("C5", c5_op3)]:
        print(f"[{name}] computing split-half reliability ...", flush=True)
        per, summ = fn()
        out[name] = {"per_unit": {k: round(v, 4) for k, v in per.items()}, "summary": summ}
        if summ:
            summaries.append(summ)
            print(f"  -> n_units={summ['n_units']}  half={summ['reliability_halfsample']}  "
                  f"full(SB)={summ['reliability_fullsample_SB']}  "
                  f"CI=[{summ['ci_lo_halfsample']},{summ['ci_hi_halfsample']}]", flush=True)
    outdir = ROOT / "results" / "_paper" / "immune_novelty"
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summaries)
    df.to_csv(outdir / "reliability_ceiling.csv", index=False)
    json.dump(out, open(outdir / "reliability_ceiling_perunit.json", "w"), indent=1)
    print("\nWROTE", outdir / "reliability_ceiling.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
