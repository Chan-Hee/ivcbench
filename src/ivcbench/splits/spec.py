"""Declarative train, inference-input and held-response membership contracts.

With ``control_inference_only=True``, inference uses controls from the held
context (for example, the query donor or lineage). Otherwise the builder uses
pooled controls, with intervention-side inputs supplied by the adapter.
``inference_context_cols`` records intended context metadata; the current
builder does not use it to filter that control pool.

These contracts apply to the supplied CellSet. They do not establish that
upstream normalization and feature selection were fitted on training cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SplitSpec:
    name: str  # e.g. "C5_cross_celltype_loct"
    cluster: str  # "C5"
    key_col: str  # obs column whose values are held out (the generalization axis)
    held_values: list[str]  # values of key_col removed from train (the test axis)
    control_inference_only: bool  # see module docstring
    strata_cols: list[str] = field(default_factory=list)  # macro-average granularity
    inference_context_cols: list[str] = field(
        default_factory=list
    )  # descriptive metadata only
    requires_side_info: str | None = (
        None  # e.g. "fingerprint" (chemistry), "gene_embedding"
    )
    registry_task: str | None = (
        None  # runtime applicability key, not final census admission
    )
    # C5 uses C5_unseen_cpd for compound holdout and C5_LOCT for lineage transfer.
    note: str = ""  # human-readable forbidden-set description (paper provenance)

    def stratum_key(self, obs_row) -> str:
        cols = self.strata_cols or [self.key_col]
        return "|".join(f"{c}={obs_row[c]}" for c in cols)
