"""Vendored PertAdapt (Bai et al. 2025, scFoundation variant) — minimal, GPU-free-importable surface.

Public API (the runner imports these):
    GOMaskedPertAdapter, loss_adapt, build_additive_go_mask        (from .pertadapt_modules)
    additive_go_mask, weighted_jaccard_adjacency, load_gene2go     (from .build_go_mask)
"""
from .pertadapt_modules import GOMaskedPertAdapter, build_additive_go_mask, loss_adapt  # noqa: F401
from .build_go_mask import additive_go_mask, weighted_jaccard_adjacency, load_gene2go  # noqa: F401
