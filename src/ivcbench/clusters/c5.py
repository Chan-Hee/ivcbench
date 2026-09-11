"""C5 (Szałata/OP3) — compound/context splits with shared upstream preprocessing.

Axis 1 — Tanimoto-stratified global compound holdout (chemistry is a post-hoc difficulty axis).
Axis 2 — cross-cell-type LOCT (seen compound, unseen lineage, control-inference-only).
"""

from __future__ import annotations

from ..splits.spec import SplitSpec

# Manually curated IFN, inflammatory and effector marker panels, not unmodified
# MSigDB collections. Only measured genes contribute. Mean-profile rank scores
# do not establish biological pathway activation or single-cell heterogeneity.
C5_PROGRAMS: dict[str, list[str]] = {
    "type_I_IFN": [
        "ISG15",
        "IFI6",
        "MX1",
        "MX2",
        "OAS1",
        "OAS2",
        "OAS3",
        "OASL",
        "IFIT1",
        "IFIT2",
        "IFIT3",
        "IFITM1",
        "IFITM3",
        "ISG20",
        "IRF7",
        "STAT1",
        "STAT2",
        "RSAD2",
        "USP18",
        "IFI44",
        "IFI44L",
        "BST2",
        "XAF1",
        "HERC5",
        "LY6E",
    ],
    "inflammatory_NFkB": [
        "NFKBIA",
        "NFKB1",
        "NFKB2",
        "REL",
        "RELB",
        "TNFAIP3",
        "TNF",
        "IL1B",
        "CXCL8",
        "CCL2",
        "CCL20",
        "CXCL2",
        "CXCL3",
        "PTGS2",
        "NFKBIE",
        "BIRC3",
        "TNFAIP2",
        "CD83",
        "IER3",
        "SOD2",
        "NFKBID",
        "TNFAIP1",
    ],
    "effector_lymphocyte": [
        "IFNG",
        "GZMB",
        "GZMA",
        "GZMK",
        "PRF1",
        "NKG7",
        "GNLY",
        "KLRD1",
        "CCL5",
        "CST7",
        "TBX21",
        "FASLG",
        "CD69",
        "IL2RA",
        "TNFRSF9",
        "CCL4",
        "CCL3",
    ],
    # NB: a cell-cycle/proliferation program was intentionally dropped — none of its canonical genes
    # (MKI67, TOP2A, PCNA, CCNB1/2, …) survive OP3's 2000-HVG selection, so it is not assessable on
    # this panel; the three retained programs each have ≥7 measured members.
}


def global_compound_holdout(held_compounds: list[str]) -> SplitSpec:
    return SplitSpec(
        name="C5_global_compound_holdout",
        cluster="C5",
        key_col="perturbation",
        held_values=list(held_compounds),
        control_inference_only=False,
        strata_cols=["perturbation", "cell_type_coarse"],
        inference_context_cols=["donor_id", "cell_type_coarse"],
        requires_side_info="fingerprint",
        registry_task="C5_unseen_cpd",
        note=(
            "held compound removed from every donor/cell-type/plate/replicate;"
            " predicted only from chemistry side-info. Tanimoto distance used post-hoc"
            " to stratify error, never as input."
        ),
    )


def cross_celltype_loct(held_lineage: str = "NK") -> SplitSpec:
    return SplitSpec(
        name=f"C5_loct_{held_lineage.replace(' ', '_')}",  # per-lineage so all-lineage LOCT splits coexist
        cluster="C5",
        key_col="cell_type_coarse",
        held_values=[held_lineage],
        control_inference_only=True,
        strata_cols=["perturbation"],
        registry_task="C5_LOCT",  # cell-axis pattern (seen compound) — see SplitSpec docstring
        note=(
            "held lineage's treated cells hidden from fitting/validation; upstream"
            " processing shared; only its DMSO/control cells are inference input"
            " (control_inference_only)."
        ),
    )
