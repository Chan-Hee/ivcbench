"""Auto-generated Materials & Methods, factored OUT of the results subsection.

Two documents, so the results draft stays results-focused:
  * framework methods  — shared across C1–C5 (data representation, leak-proof splits + auditor,
    baselines + applicability gating, the four metric axes with formulas, statistics, reproducibility)
  * cluster methods    — per-cluster data sources, **preprocessing / data-handling**, the cluster's
    split definitions (auto-filled n_train/n_test/strata + provenance notes), side-info, compute.

Run-specific values (software versions, split sizes, data source, git commit) are pulled from the
manifest + results so the Methods stay in sync with the actual run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---- cluster-specific prose (datasets + compute); split prose comes from SplitSpec.note ----
CLUSTER_META = {
    "C1": {
        "title": "Cytokine-response prediction",
        "datasets": [
            "**Kang 2018** (GSE96583; public): 29,065 PBMCs, 8 SLE donors, control vs recombinant "
            "IFN-β (single concentration, 6 h; 10x). Used only as the lineage-level IFN-β "
            "reproduction anchor, not as a cell-resolution endpoint.",
            "**Cano-Gamez 2020** (EGAS00001003215; controlled access): sorted naïve/memory CD4⁺ T "
            "cells ± Th-polarizing cytokines (Th0/Th2/Th17/iTreg). Used for naïve→memory state transfer.",
            "**Oesinghaus 2025** (preprint; bioRxiv 2025.12.12.693897): PBMC, 12 donors, 90 "
            "recombinant cytokines + PBS, 24 h (Parse). Used for within-resource cell-resolution "
            "penalty and unseen-cytokine breadth.",
        ],
        "side_info": "Cytokine-side conditioning for the LOcyt split uses receptor-family, "
                     "JAK–STAT, and pathway priors as external side information, applied at "
                     "evaluation time only (never for model selection).",
        "compute": "~280 A100-equivalent GPU-hours (≈30% of the Section-3 budget).",
        "program": "Type-I interferon response module (AUCell).",
    },
    "C3": {
        "title": "Gene-intervention prediction in primary immune cells",
        "datasets": [
            "**Shifrut 2018** (GSE119450; public): primary human CD8⁺ T-cell CRISPR-KO Perturb-seq "
            "(SLICE), 2 donors × resting/stimulated, ~20 target genes + non-targeting controls.",
            "**Schmidt 2022** (GSE190604; CRISPRa, CD4⁺ T) and **McCutcheon 2023** (GSE218985; "
            "CRISPRi+CRISPRa epigenome-editing of transcriptional/epigenetic regulators, CD8⁺ T) — "
            "loaded into the Q1 leaderboard via the same ClusterSpec.",
            "**Chen 2025** (Nature 642:191; human FOXP3 Perturb-icCITE-seq) — single-cell data at "
            "**DDBJ PRJDB16517 / GEA E-GEAD-648** (`processed.zip`, 5.9 GB; no GEO), being ingested; "
            "*not* GSE255832 (that accession is Pretto 2025, a mouse in-vivo dataset that belongs to C4).",
            "**Zhu 2025** (GSE314342 / 22M cells) and **Moonen 2026** (4.1M cells + 1,032 CREs) — "
            "genome-scale resources on the CZI Virtual Cells Platform; used for the scaling sub-axis (Q2).",
        ],
        "side_info": "Gene-side conditioning (Gene Ontology / co-expression / learned gene embeddings) "
                     "is the representation an unseen gene is predicted from; required for the "
                     "Latent/OT families to be defined on the true-LO-gene split.",
        "compute": "~250 A100-equivalent GPU-hours (≈27% of the Section-3 budget).",
        "program": "T-cell activation / effector module (AUCell), dataset-aware.",
    },
    "C5": {
        "title": "Small-molecule perturbation prediction",
        "datasets": [
            "**Szałata 2024 / OP3** (GSE279945; public): PBMC, 3 donors, 144 benchmark compounds + "
            "DMSO and positive controls, 24 h, T/B/NK/myeloid readout. The only PBMC chemical "
            "single-cell perturbation resource at benchmark scale.",
        ],
        "side_info": "Compound-side conditioning uses RDKit Morgan fingerprints, Murcko scaffolds, "
                     "and LINCS L1000 MOA/target annotation as external side information. Tanimoto "
                     "distance is used only post-hoc to stratify error, never as a model input.",
        "compute": "~70 A100-equivalent GPU-hours (≈8% of the Section-3 budget).",
        "program": "Immunomodulatory MOA module (AUCell; exploratory).",
    },
}

_FAMILIES = ("Simple (ctrl-pred, cell-mean, donor-shift, linear-PCA) · Latent (scGen, CPA/chemCPA) · "
             "Graph (GEARS, AttentionPert) · Foundation (scGPT, UCE) · Optimal-transport "
             "(CellOT, CINEMA-OT) · Hybrid (STATE)")


def _banner(data_source: str) -> str:
    if "synthetic" in (data_source or "").lower():
        return ("> **⚠ PRELIMINARY.** Quantities below (split sizes, etc.) come from a synthetic "
                f"fixture (`{data_source}`) used to validate the cycle; preprocessing prose describes "
                "the intended pipeline for the real data.\n\n")
    return f"> Data source: {data_source}.\n\n"


def framework_methods(manifest: dict) -> str:
    pk = manifest.get("packages", {})
    ver = ", ".join(f"{k} {v}" for k, v in pk.items() if v not in (None, "absent"))
    seeds = manifest.get("seeds", [0, 1, 2])
    return f"""# Benchmark Methods (shared across clusters C1–C5)

*Materials & Methods common to every cluster. Cluster-specific datasets, preprocessing, and split
definitions are given in each cluster's Supplementary Methods.*

## Unified data representation
Every dataset is ingested into a common cell × gene matrix with a harmonized cell-level annotation
schema (coarse and fine cell type, perturbation label, condition, donor, timepoint, batch, and a
control flag). Control/vehicle populations (DMSO, PBS, NTC, or 0 h) are mapped to a single control
token and never held out as a perturbation. Expression is log-normalized; cluster-specific filtering,
gene selection, and subsampling are described per cluster.

## Leak-proof splits and the leak auditor
For each (cluster, split) we define four disjoint roles — *train*, *inference input*, *test*, and a
*forbidden* set — such that the held-out group's **perturbed (treated) response never enters
training, validation, normalization, or model selection**. Two inference regimes cover all split
tasks: (i) *control-inference-only*, where only the held-out group's control cells are provided at
inference (e.g. held-out donor 0 h cells, held-out lineage DMSO cells); and (ii) *side-info
inference*, where an unseen label with no controls of its own (e.g. an unseen cytokine or compound)
is predicted from a perturbation-side representation, with control cells from matched contexts as the
baseline state. A programmatic **leak auditor** verifies these boundaries — train/test disjointness,
absence of the held-out label from train, control-only inference inputs, and treated-only test
cells — and **must pass before any metric is computed**.

## Baselines and applicability gating
Thirteen baselines span six families: {_FAMILIES}. Because a vanilla label-conditioned model cannot
construct a representation for a perturbation it never saw in training, each (baseline, split) pair is
assigned one of four applicability states — *applicable* (native conditioning fits the split),
*adapted* (defined only with an explicit side-info/conditioning extension, recorded as such),
*not defined* (vanilla form undefined for the unseen label), or *inapplicable* (family mismatch).
**Only *applicable* cells enter the headline ranking**; *not-defined* cells are reported as reference
floors and excluded; *inapplicable* cells are not run. This gating is the structural safeguard
against the most common over-claim in perturbation-prediction benchmarks — scoring undefined models
on unseen-perturbation tasks as if they were defined.

## Evaluation metrics (four axes)
All metrics are computed per stratum (donor × timepoint × cell-state, or perturbation × context) and
then macro-averaged.

1. **Response-direction fidelity.** With observed control mean x̄_ctrl, observed perturbed mean
   x̄_pert, and predicted perturbed mean x̂_pert, the effect vectors are δ_obs = x̄_pert − x̄_ctrl and
   δ_pred = x̂_pert − x̄_ctrl; we report Pearson(δ_pred, δ_obs) over perturbation-responsive genes
   (defined for evaluation only, not model selection). For CRISPR (C3) this is computed
   downstream-only, excluding the perturbed target gene.
2. **Distributional fidelity.** Energy distance E = 2·E‖P−T‖ − E‖P−P′‖ − E‖T−T′‖ between predicted
   (P) and observed (T) perturbed-cell clouds in a PCA-50 space fit on the training expression.
3. **Immune-program fidelity.** A per-cell AUCell score s on a predefined immune module; the
   program-level effect (s̄ − s̄_ctrl) is compared between prediction and observation across strata.
4. **Generalization robustness.** The gap between an easy and a leak-proof split (random↔LODO,
   within↔across class, in vitro↔in vivo, Tanimoto-near↔far); lower is better.

## Statistical protocol
Each (baseline, split, dataset) is run with {len(seeds)} seeds. We report 95% bootstrap confidence
intervals on the macro-averaged scores (B = 2000, resampling strata), apply Benjamini–Hochberg FDR
correction (α = 0.05) to within-cluster baseline-pair comparisons, and report failed runs while
excluding them from the headline ranking.

## Reproducibility and software
Each cluster cycle writes a `manifest.json` recording the git commit, software versions, seeds, data
provenance, and the per-job leak-audit summary; `make cluster C=<id>` regenerates the results table,
figure, and draft from raw data. Software: {ver or 'recorded per run in manifest.json'}.
"""


def _split_sizes(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("n_train", "n_test", "n_test_strata") if c in raw_df.columns]
    return raw_df[raw_df["ran"]].groupby("split", as_index=False)[cols].first()


def cluster_methods(cluster: str, raw_df: pd.DataFrame, specs, manifest: dict) -> str:
    meta = CLUSTER_META.get(cluster, {"title": cluster, "datasets": [], "side_info": "",
                                      "compute": "", "program": ""})
    sizes = _split_sizes(raw_df).set_index("split")
    data_lines = "\n".join(f"- {d}" for d in meta["datasets"])

    split_blocks = []
    for s in specs:
        sz = sizes.loc[s.name] if s.name in sizes.index else None
        n = (f" (n_train = {int(sz['n_train'])}, n_test = {int(sz['n_test'])}, "
             f"{int(sz['n_test_strata'])} strata)" if sz is not None else "")
        regime = ("control-inference-only" if s.control_inference_only else "side-info inference")
        split_blocks.append(
            f"- **{s.name}** — held out: `{s.key_col}` ∈ {s.held_values}; regime: {regime}; "
            f"applicability column: `{s.registry_task}`{n}.\n  {s.note}")
    splits_md = "\n".join(split_blocks)

    return f"""# Supplementary Methods — {cluster} ({meta['title']})

{_banner(manifest.get('data_source', 'unknown'))}*Shared metrics, split philosophy, applicability
gating, and statistics are in **Benchmark Methods (shared)**; this document covers {cluster}-specific
datasets, data handling, and split definitions.*

## Datasets
{data_lines}

## Preprocessing and data handling
Each dataset is ingested into the unified schema (Benchmark Methods). For {cluster} we (i) restrict to
the official cluster scope; (ii) log-normalize counts and select highly variable genes shared across
datasets within the cluster; (iii) map control/vehicle populations to the control token; (iv) harmonize
cell-type annotations to a common coarse/fine hierarchy; and (v) for resource-scale data, subsample to
a fixed number of cells per condition before splitting. All fitting and normalization use train cells
only; no statistic is estimated on held-out test cells (enforced by the leak auditor).

## Split definitions ({cluster})
{splits_md}

## Side information
{meta['side_info']}

## Compute
{meta['compute']}
{_c3_extra() if cluster == "C3" else ""}"""


def _c3_extra() -> str:
    """C3-specific Supplementary Table S3 content: dataset-aware immune programs, the main vs
    secondary response metric, the modality-stratified meta-rank, and the AUCell degeneracy note."""
    from ..clusters.c3 import C3_PROGRAMS
    rows = "\n".join(f"| {name} | {', '.join(genes)} |" for name, genes in C3_PROGRAMS.items())
    return f"""
## Supplementary Table S3 — dataset-aware immune programs (C3)
Five canonical primary-T-cell modules are scored with AUCell on every dataset; the immune-program
axis reports, per program, the Pearson correlation between predicted and observed program-Δ
(s̄_pred − s̄_ctrl vs s̄_obs − s̄_ctrl) across strata, modality-stratified. The headline AUCell-Δ is
the mean over the five programs.

| Program | Genes |
|---|---|
{rows}

**AUCell-Δ degeneracy (definition note).** The program-Δ correlation requires non-zero predicted
variance across strata. Constant-profile baselines (ctrl-pred, cell-mean, donor-shift, linear-PCA)
predict a single profile for every held-out gene, so their predicted program-Δ is constant and the
correlation is undefined; we report it as 0. This is the correct, principled value — a baseline with
no per-gene resolution cannot recover differential program modulation — and the axis discriminates
only gene-side (conditioned) models. Illustrative non-zero floor values in the planning mock are not
achievable by a constant predictor.

## Response-direction: main vs secondary (C3)
The headline Pearson-Δ is **downstream-only** (perturbed target gene excluded) so on-target
knockdown/activation cannot inflate the score. We additionally report a **target-gene-inclusive**
secondary Pearson-Δ (`pearson_delta_ontarget`) in the results table for completeness.

## Integrated ranking — modality-stratified meta-rank (O★)
Baselines are ranked by downstream-only Pearson-Δ **within each perturbation modality**
(KO / CRISPRi / CRISPRa / CRE), averaged across that modality's datasets; a baseline's meta-rank is
the mean of its per-modality ranks. Ranks are never computed on raw-pooled scores, which would let
the most-represented modality dominate. The 13-baseline 4-status taxonomy (applicable / adapted /
not-defined / failed) is shown alongside so undefined models are visible, not silently dropped.
"""


def write_methods(cluster: str, raw_df, specs, manifest: dict, cluster_out: Path,
                  shared_out: Path) -> tuple[Path, Path]:
    shared_out = Path(shared_out)
    shared_out.parent.mkdir(parents=True, exist_ok=True)
    shared_out.write_text(framework_methods(manifest))

    cluster_out = Path(cluster_out)
    cluster_out.parent.mkdir(parents=True, exist_ok=True)
    cluster_out.write_text(cluster_methods(cluster, raw_df, specs, manifest))
    return cluster_out, shared_out
