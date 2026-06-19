"""C3 (gene-intervention prediction) — true leave-one-gene-out split.

A held-out target gene's ALL sgRNAs are removed from train/val/norm/model-selection across every
donor/condition/batch (guide-level holdout is forbidden); the gene is predicted from gene-side info,
with non-targeting control cells as the inference baseline. Pearson-Δ is reported downstream-only
(the perturbed target gene is excluded) so on-target knockdown does not inflate the score.
"""
from __future__ import annotations

import numpy as np

from ..splits.spec import SplitSpec

# Dataset-aware immune programs for AUCell on C3 (OnePager Supp Table S3). Five canonical
# primary-T-cell modules; every program is scored on every dataset and the panel reports the
# per-program predicted-vs-observed program-Δ correlation (modality-stratified). The relevant
# module differs by dataset/modality (e.g. Treg/FOXP3 for Chen, effector for KO panels), so the
# panel is "dataset-aware" by construction rather than by hand-picking one set per dataset.
C3_PROGRAMS: dict[str, list[str]] = {
    "TCR_activation": ["CD69", "IL2RA", "CD40LG", "TNFRSF9", "NR4A1", "NR4A2", "NR4A3",
                       "EGR1", "EGR2", "IRF4", "REL", "NFKBIA", "CD28", "TNFRSF4"],
    "IL2_STAT5": ["IL2RA", "IL2RB", "IL2RG", "STAT5A", "STAT5B", "CISH", "SOCS1", "SOCS3",
                  "BCL2", "MYC", "IL2"],
    "proliferation": ["MKI67", "TOP2A", "PCNA", "CCNB1", "CCNB2", "CDK1", "BIRC5", "TYMS",
                      "CENPF", "UBE2C", "STMN1"],
    "effector_cytokine": ["IFNG", "TNF", "GZMB", "GZMA", "PRF1", "NKG7", "GNLY", "CCL5",
                          "CCL4", "FASLG", "TBX21"],
    "Treg_exhaustion": ["FOXP3", "IKZF2", "CTLA4", "TIGIT", "IL10", "PDCD1", "LAG3", "HAVCR2",
                        "TNFRSF18", "ENTPD1", "IL2RA"],
}

# Back-compat single set (union, used as the headline program_key fallback for fixtures/legacy).
TCELL_ACTIVATION = sorted({g for genes in C3_PROGRAMS.values() for g in genes})


def held_gene_fraction(genes, frac: float = 0.5, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    g = sorted(genes)
    k = max(1, int(round(frac * len(g))))
    return sorted(rng.choice(g, size=k, replace=False).tolist())


def true_lo_gene(held_genes, frac_label: str = "50") -> SplitSpec:
    return SplitSpec(
        name=f"C3_true_lo_gene_{frac_label}",
        cluster="C3",
        key_col="perturbation",
        held_values=list(held_genes),
        control_inference_only=False,           # unseen gene: predict from gene-side, baseline = NT controls
        inference_context_cols=["donor_id", "condition"],
        strata_cols=["perturbation"],
        requires_side_info="gene_embedding",
        registry_task="C3_LO_gene",
        note=("held target gene's ALL sgRNAs removed from train/val/norm/model-selection across every "
              "donor/condition/batch; predicted from gene-side info. Pearson-Δ is downstream-only "
              "(perturbed target gene excluded)."),
    )
