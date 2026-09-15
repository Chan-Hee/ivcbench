# INTERNALLY PRESPECIFIED ANALYSIS PLAN — Immune-Aware Perturbation-Prediction Benchmark

**Status:** Internally prespecified analysis plan, deposited with the revision. There is no
independently verifiable pre-result public timestamp for this document, so it is not presented as a
public preregistration. The available repository history does not independently establish whether this
text preceded every result it describes. Post-review changes are recorded in the revision log below.

**Version:** 1.0
**Plan date recorded by the authors:** 2026-06-05 (self-recorded; not an independently verifiable
pre-result public timestamp)
**Scope governed:** the immune-perturbation benchmark in `benchmark/`, clusters C1 (immune stimulation —
Kang IFN-β), C2 (immune stimulation — Soskic CD4 activation), C3 (CRISPR — primary-T Perturb-seq),
C4 (CRISPR / modality — Frangieh), C5 (drug — OP3 compounds), and any later cluster of the same three
perturbation classes added under the same shared contracts.

---

## (1) DESIGN CHARTER

We test whether current deep and specialized perturbation-prediction methods beat simple baselines on
immune-cell perturbation tasks, across the three perturbation classes (immune stimulation, CRISPR, and
drug), scoring predictions with immune-aware metrics under leak-safe held-out splits; a method works
only if it exceeds the simple baseline on its task.

---

## (2) UNIVERSAL SIMPLE FLOOR

The **universal simple floor** applied to EVERY cluster is the set

> **{ cell-mean shift, linear-PCA shift }**

(baseline identifiers `cell-mean` and `linear-PCA`). This identical two-member floor is computed for
every cluster, every metric axis, and every split, so the headline question — does a conditioned method
exceed the simple baseline on its task — is posed against the same reference everywhere.

Cluster-specific baselines — **FP-ridge** (C5 compound fingerprint ridge) and **donor-shift** (donor /
batch mean shift) — are **reported as context only**. They are NOT part of the headline floor and a
method is never credited or discredited by comparison to them in the headline contrasts. They appear in
tables and figures as additional context columns, explicitly labelled as context baselines, so that a
reader can see how a method fares against a stronger task-tailored shift without that comparison
entering the internally prespecified decision rule.

(`ctrl-pred`, the control-as-prediction baseline, is a degeneracy check / sanity reference and is
likewise not part of the headline floor.)

---

## (3) METRIC AXES

There are **exactly three** scored metric axes. Generalization-robustness is NOT a fourth scored axis —
it is encoded in the SPLIT DESIGN (Section 3a) and is therefore a property of how every axis below is
evaluated, not a separate score.

1. **Response-direction** — Pearson correlation of the predicted versus observed perturbation **delta**
   (mean shift relative to control) over **training-selected response genes** (genes chosen on the
   training split only; for CRISPR clusters the on-target perturbed gene is excluded, the
   downstream-only variant). Column: `pearson_delta` (CRISPR downstream-only: `pearson_delta_ontarget`).
   This is the **headline axis** for the fit-recommendation rule (Section 5).

2. **Distributional** — energy distance (**E-distance**) between predicted and observed perturbed-cell
   distributions in a **PCA space fit on training data only** (PCA-50). Column: `e_distance` (lower is
   better). Reported with its bootstrap CI (`e_distance_lo`, `e_distance_hi`).

3. **Immune-program** — AUCell-delta on **curated immune gene sets**: the correlation of predicted
   versus observed AUCell-score deltas over the cluster's curated immune programs. Column:
   `aucell_program_corr`, with the per-program AUCell columns (`aucell::<program>`) retained for the
   per-program analyses in Section 6.

Each axis is computed per stratum, then macro-averaged, with 95% bootstrap CIs; multiplicity is handled
per Section 4.

### (3a) SPLIT DESIGN (the generalization-robustness dimension — not a scored axis)

Robustness lives entirely here and is shared across all three axes above:

- **Cell-context transfer** (leave-one-cell-type / lineage out, "LOCT"): predict a SEEN perturbation in
  an UNSEEN cell type / lineage.
- **Unseen-perturbation extrapolation** (true leave-one-gene-out / leave-one-compound-out): predict an
  UNSEEN perturbation label.
- **Donor / batch transfer** (leave-one-donor-out, "LODO"), contrasted with random splits to expose
  donor-driven inflation.

Every split is passed through the leak auditor (`splits/` + `audit_split()`) before any metric is
computed; a split that does not pass the auditor for a given (baseline, seed) is not scored.

---

## (4) STATISTICS

Two layers of multiplicity control, both reporting **adjusted p-values**:

- **Per-(cluster, metric) family.** Within each (cluster × metric-axis) cell, the pre-specified family
  of contrasts is **every model family vs the universal floor** (Section 2), using the per-axis
  paired/cluster-bootstrap statistic of Section 7. Apply **Benjamini–Hochberg (BH)** across that family
  and report BH-adjusted p-values. These are the per-cell exploratory contrasts.

- **Global headline contrasts.** In addition, a small internally prespecified set of **headline contrasts**
  (Section 4a) is corrected ONCE, globally, with **BH and Holm** (both reported), computed on the
  **maximal non-ragged submatrix** — i.e. restricted to the (cluster, family) combinations that were
  actually run on the headline response-direction axis so that the correction is not diluted or biased
  by structurally-missing cells. Report BH- and Holm-adjusted p-values for every headline contrast.

The maximal non-ragged submatrix is determined mechanically from the results table (rows with
`ran == True` and `leak_free == True` on the headline axis); its membership is logged alongside the
adjusted p-values, not chosen by hand.

### (4a) INTERNALLY PRESPECIFIED HEADLINE CONTRASTS

These are the only contrasts entering the global BH/Holm correction. Each is "best eligible conditioned
model of the relevant family vs the universal simple floor" on the **headline response-direction axis**,
under the split that defines the task. "Conditioned" EXCLUDES the perturbation-agnostic OT floors
(family `ot`) when assessing whether conditioning helps.

| # | Cluster | Task / split | Family contrast |
|---|---------|--------------|-----------------|
| H1 | C1 (Kang IFN-β) | cell-context (LOCT) | latent (scGen/CPA) vs universal floor |
| H2 | C3 (primary-T CRISPR) | unseen-perturbation (true LO-gene) | best conditioned (latent/graph/foundation/hybrid) vs universal floor |
| H3 | C4 (Frangieh) | unseen-KO (modality, RNA & protein) | latent (scGen) vs universal floor |
| H4 | C5 (OP3 compounds) | cell-context (LOCT) | chemistry (FP-ridge as conditioned chemistry model) / latent vs universal floor |
| H5 | C5 (OP3 compounds) | unseen-compound | chemistry / latent vs universal floor |

(C2 enters as an additional headline contrast of the same form — latent vs floor on the CD4-activation
cell-context split — only once its results are produced under this registration; until then it is listed
as a planned headline contrast and is not implied to have run.)

---

## (5) FIT-RECOMMENDATION RULE (descriptive / exploratory — NOT a formal hypothesis test)

This rule is **descriptive**. It is not a significance test and is not corrected for multiplicity; it is
a transparent, mechanical reading of the table for the end-of-paper fit-matrix figure.

> A model **family "works" on a task** if and only if **at least one of its models exceeds the universal
> floor with a cluster-bootstrap CI_low > 0 on the headline response-direction metric** (Section 3 axis
> 1) for that task. **Ties are broken toward simplicity** (if a conditioned family and the simple floor
> are statistically indistinguishable — overlapping CIs / CI_low not above 0 — the simple floor is
> recommended).

- The CI is the biological-unit cluster bootstrap of Section 7 on the **family-minus-floor gap**.
- "Exceeds" means the gap point estimate is positive AND its cluster-bootstrap CI_low > 0.
- The end-of-paper **fit-matrix figure is labelled "descriptive / exploratory"** in its caption and is
  explicitly stated NOT to be a hypothesis test; the formal contrasts with adjusted p-values
  (Section 4) are the inferential claims.
- `scripts/fit_recommendation.py` implements this rule mechanically (input: a results table; output: the
  descriptive fit matrix). The figure is built from that script's output, not from hand-entered verdicts.

---

## (6) PRE-SPECIFIED IMMUNE-SPECIFIC ANALYSES

These are specified now and run ONCE results are in. They are immune-interpretive read-outs, reported
descriptively with CIs; they do not redefine the headline decision.

1. **Per-surface-marker protein recovery (Frangieh / C4).** Per-CITE-seq surface marker, predicted vs
   observed protein-level recovery, **including the PD-1 / PD-L1 sign** (whether the predicted direction
   of PD-1 and PD-L1 change matches observed). Report per-marker, on-target-excluded where applicable.

2. **Per-immune-program AUCell predictability map.** Across curated programs — **type-I IFN, type-II
   IFN, IL2-STAT5, T-cell activation, cytotoxicity, exhaustion, and Th subsets** — a map of how
   predictable each program's perturbation delta is (AUCell-delta correlation), per cluster and per
   method family, to localize the program-vs-magnitude dissociation.

3. **Per-donor and per-lineage predictability.** Response-direction predictability broken out per donor
   and per lineage, to expose which biological units drive (or fail) each method, and to support the
   donor-inflation contrast (random vs LODO).

---

## (7) SEED / CI POLICY

- **Cluster CIs are biological-unit bootstraps.** For each cluster the resampled unit is the named
  biological unit — **lineage** (C5 cell-context, C2), **dataset** (C3 across the primary-T datasets),
  **donor** (donor/LODO analyses), **marker** (C4 protein recovery) — resampled with replacement; the
  reported CI is over that unit, computed on **seed-0 predictions**. Hierarchical (cluster) bootstrap,
  not naive per-cell bootstrap, so non-independence within a unit is respected.

- **3-seed replication is reported ONLY where it exists** — namely **CellOT on Soskic (C2)** and
  **scGen on Kang (C1)** — and is reported there as an explicit **seed range** (min–max across the three
  seeds). It is **never implied for any other cluster, model, or contrast.** Everywhere else the headline
  number is the seed-0 point estimate with the biological-unit bootstrap CI, and no multi-seed spread is
  stated or implied.

- All bootstraps are deterministic (fixed RNG seed in the analysis scripts), so CIs are reproducible.

---

## DOCUMENT STATUS AND LIMITATION

This is an **internally prespecified analysis-plan document deposited with the revision**, not an
independently timestamped public preregistration and not a results document. The plan records the
design, floor, metric axes, multiplicity control, descriptive fit-recommendation rule, immune-specific
analyses, and seed/CI policy. Because no independently verifiable pre-result public timestamp is
available, the temporal ordering of this text and every result cannot be established from the public
record. Outcomes and post-review corrections are reported separately in the Results artifacts and the
revision log below.

---

*Author-recorded plan date: 2026-06-05. This is not an independently verifiable pre-result public
timestamp. Revision changes are dated and documented below.*

## AMENDMENT 1 — 2026-09-01 (revision of BIB-26-1553, in response to peer review)

This log records changes made after the reviews of 2026-08-31. In this revision, the status language
above was also corrected to remove unsupported public-preregistration, committed-before-results, and
frozen/immutable claims; that reclassification does not alter the substantive analysis rules.

1. **Census extended from 35 to 39 cells.** Answering Reviewer 2's first comment on uneven
   method-by-task coverage, the T1 cell-context column was filled by running CellOT, scGPT,
   scFoundation and Biolord there. Task definitions, the universal floor, the metric axes, the
   multiplicity control and the seed/CI policy recorded above are unchanged; only the roster grew.
   None of scGPT, scFoundation or Biolord publishes an interface for a seen stimulus with a held-out
   cell type, so each runs through an adaptation written for this study and is recorded as **adapted**
   (Supplementary Table S15 states what each adaptation changes).

2. **Descriptive fit-recommendation rule — fixed member.** The rule scores a family by whether at
   least one of its models beats the floor. The implementation had selected the best member separately
   within each biological unit, which is a unit-wise oracle that no single model achieves. It now
   scores ONE fixed member per task — the member with the best macro value among the models in the
   39-cell headline census, named in the table. Off-census sensitivity bundles do not enter S3.
   No verdict changes. An exact comparison of the 15 rows shared by the earlier and revised fit-matrix
   CSVs finds eight changed family gaps: C1 latent −0.024 → −0.031; C2 hybrid −0.040 → −0.089;
   C2 latent −0.068 → −0.078; C3 foundation −0.284 → −0.328; C3 graph −0.258 → −0.267;
   C3 hybrid −0.479 → −0.495; C3 latent −0.372 → −0.389; and C4 latent −0.102 → −0.105.

3. **Conditioned optimal transport is scored by the rule.** The rule excludes the *perturbation-agnostic*
   OT floors from the conditioned-family comparison. The implementation had excluded the entire OT
   family, including the conditioned predictors CellOT and scPRAM that main Table 2 and the census
   classify as conditioned. Only the perturbation-agnostic comparator CINEMA-OT is excluded now, as
   the rule is written.

4. **Reporting precision.** Answering Reviewer 1's first minor comment, table values are re-printed at
   a uniform precision. This is a display convention; no value changed.
