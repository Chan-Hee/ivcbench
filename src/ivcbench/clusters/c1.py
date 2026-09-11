"""C1 cytokine-response splits; upstream feature processing is shared.

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
    response from that donor's own control cells. Membership checks do not establish
    that upstream normalization or feature selection was training-only."""
    return SplitSpec(
        name=f"C1_lodo_{held_donor}",
        cluster="C1",
        key_col="donor_id",
        held_values=[held_donor],
        control_inference_only=True,
        strata_cols=["cell_type_coarse"],  # count alone does not ensure estimability
        registry_task="C1_LOCT",
        note=(
            "held donor's IFN-β cells hidden from fitting/validation; upstream"
            " processing shared; only its control cells are inference input"
            " (control_inference_only); scored per lineage stratum."
        ),
    )


def random_cell_split(fold_label: str) -> SplitSpec:
    """Random-cell control with donors shared across training and evaluation.

    The held-cell count is matched to donor_lodo. The score gap describes split
    sensitivity, not a pure leakage effect: target composition also changes.
    The membership audit checks the transient ``_rand_fold`` label, not donors.
    """
    return SplitSpec(
        name=f"C1_randsplit_{fold_label}",
        cluster="C1",
        key_col="_rand_fold",
        held_values=[fold_label],
        control_inference_only=True,
        strata_cols=["cell_type_coarse"],
        registry_task="C1_LOCT",
        note=(
            "RANDOM-cell-split control: held cells drawn across ALL donors (donor"
            " identity ignored), matched on held-N to donor-LODO folds; descriptive"
            " split-sensitivity comparison."
        ),
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
        note=(
            "fine sub-lineage responses held out from fitting/validation; upstream"
            " processing shared; only its control (PBS) cells are inference input."
            " Coarse lineage label remains seen."
        ),
    )


def coarse_loct(held_lineage: str = "NK") -> SplitSpec:
    """Kang IFN-β: predict a held-out coarse lineage's stimulated response from its own control cells
    (seen cytokine, unseen cell type). The classic cross-cell-type cytokine-response transfer.
    """
    return SplitSpec(
        name=f"C1_loct_{held_lineage.replace(' ', '_')}",
        cluster="C1",
        key_col="cell_type_coarse",
        held_values=[held_lineage],
        control_inference_only=True,
        strata_cols=["donor_id"],  # macro-average the held lineage's stim response over
        # donors (≥2 strata → meaningful CI + AUCell-Δ Axis 3)
        registry_task="C1_LOCT",
        note=(
            "held lineage's IFN-β-stimulated cells hidden from fitting/validation;"
            " upstream processing shared; only its control cells are inference input"
            " (control_inference_only); scored per donor."
        ),
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
        note=(
            "held CD4 state's stimulated cells hidden; only its control cells are"
            " inference input."
        ),
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
        note=(
            "held cytokine removed from every lineage/donor; predicted from"
            " receptor/pathway side-info. Tanimoto-like similarity used POST-HOC to"
            " stratify error, never as input."
        ),
    )
