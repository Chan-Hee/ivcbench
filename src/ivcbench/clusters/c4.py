"""C4 (complex-context) — Axis-2: RNA → targeted-protein modality generalization (Frangieh 2021).

Axis 1 (in vitro → in vivo, Belk/Zhou) is data-access-pending (preprint-stage author resources) and is
deferred. Axis 2 runs here: the SAME leave-one-KO-gene-out split is evaluated on the matched RNA and
24→20-marker CITE readouts of the identical cells, so the modality axis (transcriptome vs surface
proteome recoverability of an unseen KO) is read off the per-modality leaderboards + per-protein
difficulty. Leak-safe by construction (held KO's cells removed from train; predicted from non-targeting
control + the training-mean shift).
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
        control_inference_only=False,            # unseen KO: predict from control + train-mean shift
        inference_context_cols=["condition"],
        strata_cols=["perturbation"],
        registry_task="C4_Axis2",
        note=("held KO gene's cells removed from train/val/norm/model-selection; predicted from the "
              "non-targeting control baseline. Evaluated identically on the matched RNA and protein "
              "(CITE) readouts of the same cells → modality-recoverability comparison."),
    )
