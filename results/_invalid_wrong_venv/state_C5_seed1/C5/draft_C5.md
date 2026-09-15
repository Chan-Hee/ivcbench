> **Real data — OP3 / Szałata 2024 (op3_GSE279945).** Roster (1 baselines): STATE. Splits: an unseen-compound holdout + all-lineage LOCT (B, Mono, NK, T cells). chemCPA is the gene→compound (Morgan-fingerprint) variant of CPA; scGen/STATE are fingerprint-conditioned (adapted*); CINEMA-OT runs perturbation-agnostically (one global OT shift). CellOT (software unavailable), UCE (encoder-only, no decoder) and scGPT-compound (foundation, conditioning port pending) are not run.

### 3.5 Small-molecule perturbation prediction

OP3 (Szałata et al., 2024) is at present the only PBMC chemical single-cell perturbation dataset at benchmark scale. A small molecule is a chemical structure, not a categorical label, so the task is intrinsically chemistry-conditioned and we evaluate two axes: unseen-compound prediction and cross-cell-type transfer. Simple baselines are first-class comparators; details in the Supplementary Methods.

**Unseen compounds.** A held compound is removed from every donor, cell type, plate and replicate, so a model without a compound representation cannot place it — the simple baselines are reference floors (run_floor), and the headline-eligible (applicable) models are the chemistry-aware ones: none (Pearson-Δ), against the no-chemistry floor cell-mean nan. **No applicable chemistry model exceeds the floor** — Morgan-fingerprint conditioning (FP-ridge, chemCPA) adds no measurable value over a mean shift on these immune readouts, so unseen-compound generalization is **not demonstrated** here. **Tanimoto stratification** (28 fingerprint-diverse held compounds, distance 0.25–0.88 to nearest training compound): per-compound Pearson-Δ shows **no dependence on chemical distance** — every baseline's slope vs nearest-train Tanimoto is ≈0 (max R²=0.02), so a chemical-similarity difficulty axis does not manifest on these data (Figure 7a–b). 

**Cross-cell-type generalization (LOCT, 4 lineages: B, Mono, NK, T cells).** Each lineage's treated cells are withheld; its response to seen compounds is predicted from its own DMSO control. Best (mean over lineages): **STATE 0.06**. **Cell-type specificity:** NK is the hardest held-out lineage and B the most robust (mean Pearson-Δ over ranked baselines; Figure 7c–d). 

**Immune-program engagement (AUCell-Δ, Axis 3).** Across the three curated immunomodulatory-MoA programs (type-I IFN, NF-κB/inflammation, lymphocyte effector), the conditioned models — FP-ridge and the latent/hybrid decoders (scGen, chemCPA, STATE), scGen strongest — engage the programs (nonzero AUCell-Δ, largest on type-I IFN), while the constant-profile simple baselines and the perturbation-agnostic CINEMA-OT are structurally 0. So the conditioned models capture *some* program-level direction even where they do not beat the mean on the global response axis.

**Conclusion.** On real OP3, C5 is a **null for compound-side conditioning**: neither applicable chemistry model (FP-ridge, chemCPA) beats the no-chemistry mean on unseen compounds, echoing the cross-cluster finding that simple baselines are hard to beat. Cross-lineage transfer is recovered by control-anchored shifts but fails for a target-agnostic pooled mean and for from-scratch latent decoders. CINEMA-OT is perturbation-agnostic (≈ donor-shift) and is shown as a reference, not a per-compound OT prediction. Remaining ports (scGPT-compound foundation, CellOT) and a Tanimoto-stratified analysis are noted next steps (Figure 7; Supplementary Table S5).

#### Pearson-Δ leaderboard (this run)

| baseline | unseen-cpd | B | Mono | NK | T cells |
| --- | --- | --- | --- | --- | --- |
| STATE * | 0.00 | 0.11 | 0.04 | 0.03 | 0.06 |

*adapted (fingerprint-conditioned), †reference floor (no compound rep), ‡ CINEMA-OT is perturbation-agnostic (one global OT shift ≈ donor-shift), not a per-compound OT prediction. On the unseen-compound split only the applicable chemistry models (FP-ridge, chemCPA) are headline-ranked; on LOCT the compound is seen so all models are ranked. Energy-distance and per-program AUCell-Δ are in Supplementary Table S5.
