# Within-family CONSISTENCY table

Generated from `within_family_consistency.csv`. For each family with ≥2 models on a task: do the members **agree** on the beat-floor verdict, and how correlated are their per-unit Pearson-Δ vectors (Spearman ρ)?  Floor = universal {cell-mean, linear-PCA}.

| cluster | task | split | family | models | n beat both floor | n models | verdict agreement | Spearman ρ (per-unit) | flag |
|---|---|---|---|---|---|---|---|---|---|
| C1 | cytokine/Kang | cell-context (LOCT) | Latent | CPA+scGen | 0 | 2 | agree | 0.476 |  |
| C2 | donor/Soskic | donor (LODO) | Latent | CPA+scGen | 0 | 2 | agree | 0.442 |  |
| C2 | donor/Soskic | donor (LODO) | OT | CellOT+scPRAM | 1 | 2 | **split** | — (scPRAM per-donor vector not deposited element-wise; n=106) | C2 OT SPLIT: CellOT beats both floor members, scPRAM fails (delta-vs-floor -0.108, beats floor in 12/106 donors); within-family verdict disagreement -> the lone donor win is CellOT-specific. Paired scPRAM-vs-CellOT (n=106): scPRAM wins 1/106, gap -0.210, Wilcoxon p=4.3e-19 |
| C3 | gene/CRISPR | unseen-perturbation (LO-gene 10%) | Foundation | scFoundation+scGPT | 0 | 2 | agree | 1.000 |  |
| C3 | gene/CRISPR | unseen-perturbation (LO-gene 10%) | Graph | AttentionPert+GEARS | 0 | 2 | agree | 0.900 |  |
| C3 | gene/CRISPR | unseen-perturbation (LO-gene 10%) | Hybrid | PertAdapt+STATE | 0 | 2 | agree | 0.700 |  |
| C3 | gene/CRISPR | unseen-perturbation (LO-gene 10%) | Latent | CPA+scGen | 0 | 2 | agree | 0.700 |  |
| C4 | complex/Frangieh | unseen-KO (modality, RNA) | Graph | AttentionPert+GEARS | 0 | 2 | agree | — (n<3 units) | C4: 2 modality folds only (rho undefined, <3 units); CellOT/scPRAM single-seed per split -> re-run for CI |
| C4 | complex/Frangieh | unseen-KO (modality, RNA) | Latent | CPA+scGen | 0 | 2 | agree | — (n<3 units) | C4: 2 modality folds only (rho undefined, <3 units); CellOT/scPRAM single-seed per split -> re-run for CI |
| C4 | complex/Frangieh | unseen-KO (modality, RNA) | OT | CellOT+scPRAM | 0 | 2 | agree | — (n<3 units) | C4: 2 modality folds only (rho undefined, <3 units); CellOT/scPRAM single-seed per split -> re-run for CI |
| C5 | small-mol/OP3 | cell-context (LOCT) | Latent | CPA+scGen | 0 | 2 | agree | −0.800 |  |
| C5 | small-mol/OP3 | unseen-compound | Chemistry | FP-ridge+chemCPA | 0 | 2 | agree | — (n<3 units) |  |
| C5 | small-mol/OP3 | unseen-compound | Latent | CPA+scGen | 0 | 2 | agree | — (n<3 units) |  |

## Notes

- **Verdict agreement holds in 12 of the 13 family/task cells**: paired members reach the SAME beat-floor verdict (all beat, or none beat). The **one within-family verdict SPLIT is the C2 donor OT pair** (CellOT beats both floor members; scPRAM fails) — exactly the cell that carries the headline donor win, so the lone positive splits its own family. (ρ is not printed for that cell: scPRAM's per-donor vector is summarized in `scpram_vs_cellot_donor_paired.csv`, n=106, not deposited element-wise.)
- **C3 ρ is high within family** (Foundation ρ=1.00, Graph 0.90, Hybrid 0.70, Latent 0.70 across the 5 primary-T datasets): family members rank datasets the same way even though both members sit below floor — consistent failure, not noise.
- **C1 / C2 Latent ρ moderate** (0.48 / 0.44 across lineages / 106 donors): scGen and CPA agree directionally but not tightly.
- **C5 Latent ρ = −0.80** on the unseen-compound split (only 1 unit there → computed across the LOCT lineages instead; small n, treat as indicative).
- **C4 ρ undefined**: only 2 modality folds (LO-KO 25% / 50%) → <3 units, so cross-model ρ is not computable. **CellOT and scPRAM on Frangieh ran a single seed per split (the two LO-KO fractions, no CI / no multi-seed) — FLAGGED for re-run** to obtain a marker-bootstrap CI before any inferential claim.
