# Unseen-cytokine extrapolation (pseudobulk-DE) — supplementary test of the extrapolation law

**Question.** Does the benchmark's no-unseen-perturbation-extrapolation law (established on genes,
C3, and compounds, C5) extend to **cytokines**? I.e. on a held-out cytokine, does ANY predictor beat
the cytokine-mean floor?

**Data.** `data/human_cytokine_dict/hcd_mini.csv` — the Cytokine Dictionary pseudobulk
differential-expression summary table: per `(gene × celltype × cytokine)` the `log_fc` of the
cytokine response vs PBS control. 371,424 DE rows, 24 celltypes, 87 cytokines, 13,834 genes. The
table is **sparse (significant-DE rows only)**; a `(celltype, cytokine)` DE vector is built on that
celltype's gene **universe** (union of its DE genes) with genes that were not significantly perturbed
**0-filled** (= no detected response).

**Task.** Leave-one-cytokine-out **within each celltype** (the cytokine analog of C3 leave-one-gene-
out): predict the held cytokine's log-fc vector over the celltype gene universe, score
**response-direction Pearson** vs the observed held DE. The held cytokine's effect *in the held
celltype* is never used (leak-safe; numerically verified that the neighbour-selection profile excludes
the held celltype). 1,810 scoreable held instances; 99 dropped uniformly across all methods because
the held DE vector is degenerate (<5 nonzero genes).

**Predictors.**
- `zero` — no-response baseline (constant 0 → undefined direction → scored 0 by convention).
- `cytokine-mean` — **the floor**: mean DE across training cytokines in that celltype.
- `feature-nearest` — conditioned, **annotation only**: nearest training cytokine by a receptor-family
  / structural-class feature set parsed from the cytokine name (IFN-I/II/III, common γ-chain, IL-1
  superfamily, gp130/IL-6, IL-10, IL-12, IL-17, common β-chain, TNF superfamily, RTK growth factors,
  …). This is the **truly-novel** regime: the cytokine's effect is never observed in any context.
- `DE-profile-nearest` — conditioned, **observed-elsewhere transfer**: nearest training cytokine by
  DE-profile similarity computed **in OTHER celltypes** (the cytokine's identity is observable from
  contexts that are not the held one), transferring that neighbour's DE in the held celltype.

## Result (real numbers)

Pooled mean response-direction Pearson over all 1,810 held cytokine instances:

| method | mean Pearson | median | vs floor |
|---|---|---|---|
| zero (no response) | 0.000 | 0.000 | — |
| **cytokine-mean (floor)** | **0.195** | 0.169 | — |
| feature-nearest (annotation only) | 0.153 | 0.089 | **−0.041** (does NOT beat floor) |
| DE-profile-nearest (observed-elsewhere transfer) | 0.325 | 0.302 | **+0.131** (beats floor) |

Paired floor-vs-conditioned gap (resampling unit = held cytokine, 95% CI over held cytokines):
- DE-profile-nearest − floor: **+0.131 [+0.122, +0.139]**, beats floor on **1429/1810 (79%)** of held
  cytokines, and in **19/24 celltypes** (the 5 losses are the small-n / under-sampled lineages:
  Plasmablast n=15, Granulocyte n=26, HSPC, pDC, ILC n=9).
- feature-nearest − floor: **−0.041 [−0.048, −0.035]**, beats floor on only **539/1810 (30%)** of held
  cytokines, and in only **2/24 celltypes**.

## Interpretation — a regime split, not a simple "law breaks"

The law holds in the **truly-novel** regime and is escaped only by **cross-context transfer**:

- **Annotation-only conditioning fails the floor** (−0.04). A purely a-priori representation of an
  unseen cytokine (its receptor family) does NOT beat the average-cytokine prior — exactly the
  no-extrapolation law seen for unseen genes (C3) and unseen compounds (C5).
- **A cytokine observed in OTHER celltypes can be transferred** (+0.13, 79% of cases). Because the
  cytokine response manifold is low-dimensional and family-structured (shared receptor → JAK/STAT
  programmes), the nearest *observed* neighbour is a much better prior than the average. Examples of
  the neighbours chosen: IL-15↔IL-2 (common γ-chain), IL-3→GM-CSF (common β-chain myeloid),
  IL-24/OSM (gp130/STAT3). IL-10's average-cytokine floor is actually **negative** (−0.32, the average
  cytokine points the wrong way), which its nearest neighbour recovers to +0.32.

So the honest framing: the extrapolation law extends to cytokines **in the strict unseen sense**
(annotation-only conditioning cannot beat the floor), but a cytokine that is "unseen here, seen
elsewhere" is a *cross-celltype-transfer* problem, not a true-novel-perturbation problem, and there
conditioning clearly helps. This sharpens the law's boundary: it is about *never-observed*
perturbations, not perturbations observed in a different context.

## LIMITATION (state plainly)

This is a **pseudobulk-DE-level** test on the **summary table** (`hcd_mini.csv`), **not single-cell**,
and not the full single-cell Cytokine Dictionary. DE vectors are significance-thresholded and
0-filled, so direction is scored on the union-of-significant gene universe per celltype. It is a
**direct supplementary test of the extrapolation law**, not a full benchmark cluster (no single-cell
distributional metrics, no trained deep baselines). The DE-profile-nearest "transfer" predictor is
leak-safe but is a cross-context transfer, not a true-novel-perturbation predictor — that distinction
is the point of the finding, not a confound.

## Files
- `cytokine_loco_per_held.csv` — per (celltype, held cytokine, method) Pearson + chosen neighbour.
- `cytokine_loco_per_celltype.csv` — per-celltype mean Pearson per method.
- `cytokine_loco_unscoreable.csv` — 99 degenerate held cytokines (dropped equally for all methods).
- `cytokine_loco_summary.json` — pooled means, gaps, CIs, beat-counts, verdict.
- `figS_newdata_cytokine_loco.png` — figure (also `results/_paper/figS_newdata_cytokine_loco.{png,pdf}`).
- scripts: `scripts/newdata_cytokine_loco.py`, `scripts/figure_newdata_cytokine_loco.py`.
