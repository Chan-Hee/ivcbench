"""The panel mask must agree with the runners that motivated it.

scfoundation_gene_runner and pertadapt_c3_runner each resolve the evaluation panel against the
released OS_scRNA_gene_index.19264.tsv and print how many genes they could not represent. If
ivcbench.eval.panel_mask ever drifts from that resolution, the census would exclude a different
gene set than the models actually failed on, which is worse than not masking at all.

The expected counts are not invented here. Three come from NATIVE_102_FINAL_REVIEW.md section
247, which tabulated them against the real payloads, and all five were printed by the running
PertAdapt units as `[panel] modelled=N/2000`.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

pytest.importorskip("numpy")

# unit -> (bundle glob, genes the checkpoint cannot represent)
CASES = {
    "shifrut": ("predictions/**/C3_LO_gene__*__C3_true_lo_gene_10__shifrut.npz", 31),
    "schmidt": ("predictions/**/C3_LO_gene__*__C3_true_lo_gene_10__schmidt.npz", 49),
    "chen": ("predictions/**/C3_LO_gene__*__C3_true_lo_gene_10__chen.npz", 59),
    "mccutcheon_CRISPRa": (
        "predictions/**/C3_LO_gene__*__C3_true_lo_gene_10__mccutcheon_CRISPRa.npz",
        54,
    ),
    "frangieh_ko_25": ("predictions/**/C4_Axis2__*__C4_modality_lo_ko_25.npz", 26),
}


def _panel(pattern: str) -> list[str] | None:
    for path in sorted(glob.glob(pattern, recursive=True)):
        data = np.load(path, allow_pickle=True)
        if "genes" in data.files:
            return [str(g) for g in np.asarray(data["genes"])]
    return None


@pytest.mark.skipif(
    not os.environ.get("IVCBENCH_SCFOUNDATION_DIR"),
    reason="released vocabulary not configured; runs/env.sh sets IVCBENCH_SCFOUNDATION_DIR",
)
@pytest.mark.parametrize("unit", sorted(CASES))
def test_mask_matches_what_the_runners_reported(unit: str) -> None:
    from ivcbench.eval.panel_mask import unrepresentable

    pattern, expected = CASES[unit]
    panel = _panel(pattern)
    if panel is None:
        pytest.skip(f"no deposited bundle with a gene list for {unit}")
    assert len(panel) == 2000, f"{unit}: panel is {len(panel)} genes, expected the 2,000-gene panel"
    assert len(unrepresentable(panel)) == expected


@pytest.mark.skipif(
    not os.environ.get("IVCBENCH_SCFOUNDATION_DIR"),
    reason="released vocabulary not configured",
)
def test_mask_is_a_property_of_the_panel_not_of_a_model() -> None:
    """Every model on a cell must get the same mask, or the comparison stops being common."""
    from ivcbench.eval.panel_mask import unrepresentable

    paths = [
        p
        for p in sorted(
            glob.glob(
                "predictions/**/C3_LO_gene__*__C3_true_lo_gene_10__schmidt.npz", recursive=True
            )
        )
        if "genes" in np.load(p, allow_pickle=True).files
    ]
    if len(paths) < 2:
        pytest.skip("need at least two models' bundles on the same unit")
    masks = set()
    for path in paths:
        panel = [str(g) for g in np.asarray(np.load(path, allow_pickle=True)["genes"])]
        masks.add(tuple(unrepresentable(panel)))
    assert len(masks) == 1, "the same unit produced different masks for different models"
