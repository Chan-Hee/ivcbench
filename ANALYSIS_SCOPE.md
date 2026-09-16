# Analysis scope for the revised panel

This is a review-stage analysis specification, not a prospective preregistration.
The final panel contains 58 evaluations: 46 native, eight adapted and four diagnostic.
The executed-interface inventory and bundle-selection manifest are authoritative.

The eight adapted entries are scGPT and scFoundation on T1, T2, T5c and T5u: each
reaches its split through a task interface written for this study, because the
published interface does not express the held entity. A shortfall by one of them
therefore bounds that interface as well as the architecture, and a clearance does
too. Their classification changes provenance, not prediction values or the
comparison family. (PertAdapt carried this note in the submitted panel, as an
adapted T2 donor head; it was re-run and is native on T3 and T4, and its T2 cell
now carries a reason in Supplementary Table S15b instead.)

Pearson-Δ is macro-averaged over the recorded biological analysis units. The two
simple references are cell-mean and linear-PCA shifts. The stronger reference is
selected at task level, then held fixed for paired unit-level margins. This differs
from taking a different better reference for each donor/gene; auxiliary analyses
using that older convention are identified separately.

Analysis units are eight T1 lineages, 106 T2 donors, five T3 dataset-arms, two T4
holdout fractions, four T5c lineages and 28 T5u compounds. T3's 43 held genes are
not treated as 43 independent datasets; T4 fractions overlap. Cell counts are not
biological replicate counts.

All 27 entries with at least eight units enter one two-sided Wilcoxon family, with
both BH and Holm adjustments, regardless of the sign of their observed margin.
Those entries receive conditional unit-bootstrap intervals (4,000 resamples).
The other 31 entries receive descriptive observed ranges, not inferential P values,
confidence intervals or power claims. Approximate MDE values for the 27 eligible
entries describe margin scale at 80% power; they are not a prospective design audit.

Intervals condition on fitted models, fixed masks, selected floor and saved seeds.
They omit model-selection, optimization-seed and independent-dataset uncertainty.
The common family was defined during revision; it was not registered before data
inspection. Eight point estimates exceed the task reference, seven of them conditioned
predictors and the eighth the FP-ridge diagnostic, but only CellOT T2 has
positive support from both its conditional interval and the adjusted family.

Dataset-wise winners, observed-effect bins, donor-pool curves, chemical-distance
equivalence, program concordance and checkpoint-normalization checks are separate
sensitivity or descriptive analyses. Their conditioning, references and missing
values are disclosed in Supplementary Notes S3–S7. They do not enlarge the main
family silently or turn a small/undefined correlation into proof of equivalence.

Each row of `panel_precision_attenuation.csv` has explicit power-scale and
attenuation statements. Exact held-target/score-mask repeatability uses 100
disjoint treated/control partitions, compared with the same treated halves and
a pooled shared control. Strata are averaged within the original primary units;
estimable coverage is explicit. Partition ranges describe cell allocation, not
biological replication. Neither statistic is a ceiling or attenuation correction.

T3/T5c program correlations compare scores of predicted and observed mean
expression, never a mean-profile score to an average of per-cell rank scores.
The identity control has zero error across 37 targets; 12 variable targets have
correlation one and 25 constant targets have undefined correlation. Of T3's 25
targets, 22 are constant. NA-O distinguishes this from prediction-constant NA-P.
These numerical controls do not validate biological pathway activation.

Shared preprocessing limits all end-to-end inductive claims. Soskic additionally
uses condition-specific residualized, scaled and clipped source matrices, with
no raw-count layer. T2 scores measure donor transfer in those coordinates, not
unregressed activation. Training-only additional scaling does not reverse source
processing or recover unmeasured genes; the input-space audit records this limit.
