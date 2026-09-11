"""C4: held-KO RNA prediction and separate within-protein marker checks.

Frangieh's RNA and 20-marker surface readouts are fitted within their respective
modalities. No RNA-to-protein predictor or cross-modal transfer is evaluated.
Held KOs are excluded from fitting, but upstream feature processing is shared.
"""

from __future__ import annotations

import numpy as np

from ..splits.spec import SplitSpec


def held_ko_fraction(genes, frac: float = 0.5, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    g = sorted(genes)
    k = max(1, int(round(frac * len(g))))
    return sorted(rng.choice(g, size=k, replace=False).tolist())


def modality_lo_ko(held_genes, frac_label: str = "50") -> SplitSpec:
    return SplitSpec(
        name=f"C4_modality_lo_ko_{frac_label}",
        cluster="C4",
        key_col="perturbation",
        held_values=list(held_genes),
        control_inference_only=False,  # unseen KO: predict from control + train-mean shift
        inference_context_cols=["condition"],
        strata_cols=["perturbation"],
        registry_task="C4_Axis2",
        note=(
            "held KO gene's cells removed from fitting/validation; upstream processing"
            " shared. Non-targeting controls provide the baseline. RNA and protein fits"
            " are separate; this split does not define RNA-to-protein prediction."
        ),
    )
