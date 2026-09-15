# Benchmark Methods (shared across clusters C1–C5)

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
Thirteen baselines span six families: Simple (ctrl-pred, cell-mean, donor-shift, linear-PCA) · Latent (scGen, CPA/chemCPA) · Graph (GEARS, AttentionPert) · Foundation (scGPT, UCE) · Optimal-transport (CellOT, CINEMA-OT) · Hybrid (STATE). Because a vanilla label-conditioned model cannot
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
Each (baseline, split, dataset) is run with 1 seeds. We report 95% bootstrap confidence
intervals on the macro-averaged scores (B = 2000, resampling strata), apply Benjamini–Hochberg FDR
correction (α = 0.05) to within-cluster baseline-pair comparisons, and report failed runs while
excluding them from the headline ranking.

## Reproducibility and software
Each cluster cycle writes a `manifest.json` recording the git commit, software versions, seeds, data
provenance, and the per-job leak-audit summary; `make cluster C=<id>` regenerates the results table,
figure, and draft from raw data. Software: numpy 2.4.6, scipy 1.17.1, sklearn 1.8.0, pandas 2.3.3, matplotlib 3.10.9.
