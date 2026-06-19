"""C1 (cytokine-response) — the leak-proof splits from the OnePager (§5.1).

Axis 1 — cell resolution / state:
  * within-resource fine sub-lineage LOCT (cell-resolution penalty)
  * Cano-Gamez naive -> memory CD4 state transfer
  * (Kang lineage-level IFN-β is a separate reproduction anchor)
Axis 2 — unseen-cytokine (LOcyt), gated to cytokine-side-conditioned models.
"""
from __future__ import annotations

from ..splits.spec import SplitSpec

TYPE_I_IFN_KEY = "type_i_ifn"


def donor_lodo(held_donor: str) -> SplitSpec:
    """DONOR axis — leave-one-donor-out: hold out one donor's IFN-β-treated cells entirely, predict its
    response from that donor's own control cells. Demonstrates donor-generalization under a leak-proof
    split (the honest measure). Reuses the cell-axis applicability pattern (no perturbation-side rep)."""
    return SplitSpec(
        name=f"C1_lodo_{held_donor}",
        cluster="C1",
        key_col="donor_id",
        held_values=[held_donor],
        control_inference_only=True,
        strata_cols=["cell_type_coarse"],            # ≥2 strata → bootstrap CI + AUCell-Δ well-defined
        registry_task="C1_LOCT",
        note=("held donor's IFN-β cells hidden from train/val/norm/model-selection; only its control "
              "cells are inference input (control_inference_only); scored per lineage stratum."),
    )


def random_cell_split(fold_label: str) -> SplitSpec:
    """DONOR axis control — a RANDOM-cell split (donor identity NOT honored): held cells span all donors,
    so the held set's donor neighbours leak into train. Matched (equal held-N) to a donor_lodo fold; the
    LODO-vs-random score gap quantifies random-split optimism (the leakage finding). The leak AUDIT still
    passes (mechanically a valid control_inference_only split on a random fold) — the gap is statistical,
    not a mechanical leak. Needs a transient obs column `_rand_fold` (assigned in the dispatcher)."""
    return SplitSpec(
        name=f"C1_randsplit_{fold_label}",
        cluster="C1",
        key_col="_rand_fold",
        held_values=[fold_label],
        control_inference_only=True,
        strata_cols=["cell_type_coarse"],
        registry_task="C1_LOCT",
        note=("RANDOM-cell-split control: held cells drawn across ALL donors (donor identity ignored), "
              "matched on held-N to the donor-LODO folds → quantifies random-split inflation vs leak-safe LODO."),
    )


def resolution_fine_loct(held_fine: str = "CD8_memory") -> SplitSpec:
    """Predict a held-out FINE sub-lineage's cytokine response (coarse lineage still seen)."""
    return SplitSpec(
        name="C1_resolution_fine_loct",
        cluster="C1",
        key_col="cell_type_fine",
        held_values=[held_fine],
        control_inference_only=True,
        strata_cols=["perturbation"],
        registry_task="C1_LOCT",
        note=("fine sub-lineage held out from train/val/norm/model-selection; only its control "
              "(PBS) cells are inference input. Coarse lineage label remains seen."),
    )


def coarse_loct(held_lineage: str = "NK") -> SplitSpec:
    """Kang IFN-β: predict a held-out coarse lineage's stimulated response from its own control cells
    (seen cytokine, unseen cell type). The classic cross-cell-type cytokine-response transfer."""
    return SplitSpec(
        name=f"C1_loct_{held_lineage.replace(' ', '_')}",
        cluster="C1",
        key_col="cell_type_coarse",
        held_values=[held_lineage],
        control_inference_only=True,
        strata_cols=["donor_id"],                  # macro-average the held lineage's stim response over
                                                   # donors (≥2 strata → meaningful CI + AUCell-Δ Axis 3)
        registry_task="C1_LOCT",
        note=("held lineage's IFN-β-stimulated cells hidden from train/val/norm/model-selection; only "
              "its control cells are inference input (control_inference_only); scored per donor."),
    )


def cd4_state_transfer(held_state: str = "CD4_memory") -> SplitSpec:
    """Naive -> memory CD4 state transfer (train on naive + others, predict memory)."""
    return SplitSpec(
        name="C1_cd4_state_transfer",
        cluster="C1",
        key_col="cell_type_fine",
        held_values=[held_state],
        control_inference_only=True,
        strata_cols=["perturbation"],
        registry_task="C1_state",
        note="held CD4 state's stimulated cells hidden; only its control cells are inference input.",
    )


def locyt(held_cytokine: str = "IL17") -> SplitSpec:
    """Leave-one-cytokine-out. Control (PBS) is NEVER held out. Gated to cytokine-side models."""
    return SplitSpec(
        name="C1_locyt",
        cluster="C1",
        key_col="perturbation",
        held_values=[held_cytokine],
        control_inference_only=False,
        inference_context_cols=["cell_type_fine", "donor_id"],
        requires_side_info="cytokine_prior",
        registry_task="C1_LOcyt",
        note=("held cytokine removed from every lineage/donor; predicted from receptor/pathway "
              "side-info. Tanimoto-like similarity used POST-HOC to stratify error, never as input."),
    )
