"""C2 (donor-generalization) — Soskic CD4-activation leave-one-donor-out (LODO).

The Fig-1 OT-STRONG paired-stimulation DONOR axis: control = 0h resting, perturbed = 16h stimulated.
Hold ONE donor's 16h cells out entirely; predict its stimulated response from its OWN 0h cells. Lineage
(CD4 Naive/Memory) is a within-donor stratum. Leak-safe: the held donor's stim cells never enter
train/val/norm/model-selection (control_inference_only) — enforced by audit_split.

Re-wraps the leak-safe Soskic logic from scripts/c2_soskic_donor.py into the framework:
  * response_gene_idx — training-only control-vs-stim response panel for the held-donor fold. Required
    for evaluation but FORBIDDEN for model selection, so selected inside each fold from TRAINING donors
    only and passed only to the metric (run_job's response_gene_fn hook). This is the framework-native
    home of the bespoke leak-safe response-gene rule; it makes the C2 Pearson-Δ reproduce the bespoke
    per-donor value.
  * SOSKIC_PROGRAMS — the immune-program (AUCell Axis-3) vocabulary for activated CD4 T cells.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..data.schema import CellSet
from ..splits.builder import Split
from ..splits.spec import SplitSpec

# Same immune-program vocabulary as the bespoke pipeline: T-cell activation + IL2/STAT5 (from C3),
# type-I IFN (from C1/C5), plus a compact type-II IFN module for activated CD4 T cells. run.py
# intersects each set with the dataset HVG panel, so a program contributes only measured genes.
SOSKIC_PROGRAMS: dict[str, list[str]] = {
    "T_cell_activation": ["CD69", "IL2RA", "CD40LG", "TNFRSF9", "NR4A1", "NR4A2", "NR4A3",
                          "EGR1", "EGR2", "IRF4", "REL", "NFKBIA", "CD28", "TNFRSF4"],
    "IL2_STAT5": ["IL2RA", "IL2RB", "IL2RG", "STAT5A", "STAT5B", "CISH", "SOCS1", "SOCS3",
                  "BCL2", "MYC", "IL2"],
    "type_I_IFN": ["ISG15", "IFI6", "MX1", "MX2", "OAS1", "OAS2", "OAS3", "OASL", "IFIT1",
                   "IFIT2", "IFIT3", "IFITM1", "IFITM3", "ISG20", "IRF7", "STAT1", "STAT2",
                   "RSAD2", "USP18", "IFI44", "IFI44L", "BST2", "XAF1", "HERC5", "LY6E"],
    "type_II_IFN": ["IFNG", "STAT1", "IRF1", "CXCL9", "CXCL10", "CXCL11", "GBP1", "GBP2",
                    "GBP5", "TAP1", "PSMB8", "PSMB9", "HLA-DRA", "HLA-DRB1", "SOCS1"],
}


def donor_lodo(held_donor: str) -> SplitSpec:
    """DONOR axis — leave-one-donor-out: hold out one donor's 16h-stimulated cells entirely, predict
    its response from that donor's own 0h control cells. Leak-proof (the honest measure). Cell-axis
    applicability pattern (no perturbation-side rep needed); registry_task=C2_LODO."""
    return SplitSpec(
        name=f"C2_lodo_{held_donor}",
        cluster="C2",
        key_col="donor_id",
        held_values=[held_donor],
        control_inference_only=True,
        strata_cols=["cell_type_coarse"],            # CD4 Naive/Memory → ≥2 strata → CI + AUCell-Δ defined
        registry_task="C2_LODO",
        note=("held donor's 16h-stim cells hidden from train/val/norm/model-selection; only its 0h "
              "control cells are inference input (control_inference_only); scored per lineage stratum."),
    )


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.ones_like(p)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranks = np.arange(1, len(order) + 1)
    vals = p[order] * len(order) / ranks
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    q[order] = np.clip(vals, 0, 1)
    return q


def response_gene_idx(cs: CellSet, split: Split, max_genes: int = 200, min_genes: int = 50) -> np.ndarray:
    """Training-only control-vs-stim response genes for the held-donor fold (leak-safe).

    These genes are EXCLUDED from the C2 Pearson-Δ metric (run_job feeds them to pearson_delta's
    exclude_genes): the strongly stimulation-driven response panel is removed so the score measures
    direction recovery on the remaining transcriptome rather than being saturated by the obvious
    activation genes. The panel is selected inside each fold from TRAINING donors only and used by the
    metric alone (never by any model), so it is FORBIDDEN for model selection and constitutes no leak.
    Identical numerics to the bespoke scripts/c2_soskic_donor.py:response_gene_idx, re-wrapped to the
    framework Split interface.
    """
    train_idx = split.train_idx
    tr_obs = cs.obs.iloc[train_idx]
    is_ctrl = tr_obs["is_control"].to_numpy().astype(bool)
    ctrl = cs.X[train_idx[is_ctrl]]
    stim = cs.X[train_idx[~is_ctrl]]
    if len(ctrl) < 3 or len(stim) < 3:
        return np.arange(cs.X.shape[1])
    t = stats.ttest_ind(stim, ctrl, axis=0, equal_var=False, nan_policy="omit")
    p = np.nan_to_num(t.pvalue, nan=1.0, posinf=1.0, neginf=1.0)
    q = _bh_qvalues(p)
    effect = np.abs(stim.mean(0) - ctrl.mean(0))
    ranked = np.argsort(-effect)
    sig = ranked[q[ranked] < 0.05]
    if len(sig) < min_genes:
        sig = ranked[: min(min_genes, len(ranked))]
    return np.asarray(sig[: min(max_genes, len(sig))], dtype=int)
