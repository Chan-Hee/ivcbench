"""Declarative leak-proof split specifications.

Each (cluster, split) on page 3 of the Master Overview defines train / inference-input / test /
forbidden boundaries. We encode that contract as data so it is auditable and reproducible.

Two inference-input regimes cover all 9 split tasks:
  * control_inference_only=True  — the held-out group's CONTROL cells are the only inference input
    (C2 LODO: held donor 0h cells; C2 temporal: earlier timepoints; C4: in-vivo control cells;
     C5 LOCT: held lineage DMSO cells; C1 state: held-state controls). The held group's *treated*
     cells are forbidden everywhere in train/val/norm/model-selection.
  * control_inference_only=False — the held-out label has no controls of its own and is predicted
    from perturbation-side info (C1 LOcyt cytokine side-info; C3 gene-side meta; C5 unseen-compound
    chemistry). Inference baseline = control cells from matched contexts (inference_context_cols).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SplitSpec:
    name: str                      # e.g. "C5_cross_celltype_loct"
    cluster: str                   # "C5"
    key_col: str                   # obs column whose values are held out (the generalization axis)
    held_values: list[str]         # values of key_col removed from train (the test axis)
    control_inference_only: bool   # see module docstring
    strata_cols: list[str] = field(default_factory=list)        # macro-average granularity
    inference_context_cols: list[str] = field(default_factory=list)  # used when not control-only
    requires_side_info: str | None = None  # e.g. "fingerprint" (chemistry), "gene_embedding"
    registry_task: str | None = None  # column in the 13x9 applicability matrix this split maps to.
    # NOTE: C5 has two splits; the matrix encodes only the unseen-compound column ("C5_unseen_cpd").
    #   - global compound holdout -> registry_task="C5_unseen_cpd" (compound-side rep gated)
    #   - cross-cell-type LOCT (seen compound, unseen lineage) -> registry_task="C1_LOCT" (cell-axis
    #     pattern: all families applicable, Graph inapplicable) since no compound-side rep is needed.
    note: str = ""                 # human-readable forbidden-set description (paper provenance)

    def stratum_key(self, obs_row) -> str:
        cols = self.strata_cols or [self.key_col]
        return "|".join(f"{c}={obs_row[c]}" for c in cols)
