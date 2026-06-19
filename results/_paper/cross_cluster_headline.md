# Cross-cluster HEADLINE table — response-direction delta vs the UNIVERSAL floor

Generated mechanically from `cross_cluster_headline.csv` (`scripts/assemble_cross_cluster.py`). **Real, already-computed results only.**

**Metric:** response-direction = Pearson-Δ (`pearson_delta`), PREREG axis 1 (headline).  
**Reference:** the UNIVERSAL simple floor = {`cell-mean`, `linear-PCA`} (PREREG §2), NOT cluster-specific floors (donor-shift / FP-ridge are context-only and excluded here).  
`delta_vs_floor_mean` = family Pearson-Δ − mean(cell-mean, linear-PCA).  
`beats_both` = point estimate exceeds BOTH floor members. This is the **point-estimate direction/magnitude** read; the CI-gated fit verdict (CI_low>0 on the gap, PREREG §5) is the separate descriptive fit-matrix and is NOT asserted here.  
Biological unit macro-averaged per cluster (PREREG §7): C1 lineage, C2 donor, C3 dataset, C4 modality-fold (RNA), C5 lineage / compound.

## C1 — cytokine/Kang — cell-context (LOCT)
Universal floor: cell-mean = 0.5896, linear-PCA = 0.7803, floor-mean = 0.6849  (unit = lineage, n = 8)

| family | model | Pearson-Δ | Δ vs floor-mean | Δ vs cell-mean | Δ vs linear-PCA | beats both? |
|---|---|---|---|---|---|---|
| Latent | scGen | 0.7645 | 0.0795 | 0.1749 | −0.0158 | no |
| Latent | CPA | 0.7111 | 0.0262 | 0.1215 | −0.0692 | no |

## C2 — donor/Soskic — donor (LODO)
Universal floor: cell-mean = 0.2598, linear-PCA = 0.0362, floor-mean = 0.1480  (unit = donor, n = 106)

| family | model | Pearson-Δ | Δ vs floor-mean | Δ vs cell-mean | Δ vs linear-PCA | beats both? |
|---|---|---|---|---|---|---|
| OT | CellOT | 0.3691 | 0.2211 | 0.1093 | 0.3329 | **yes** |
| Latent | CPA | 0.1966 | 0.0485 | −0.0633 | 0.1603 | no |
| Hybrid | STATE | 0.1830 | 0.0350 | −0.0768 | 0.1468 | no |
| OT | scPRAM | 0.1592 | 0.0112 | −0.1006 | 0.1230 | no |
| Latent | scGen | 0.1436 | −0.0044 | −0.1162 | 0.1074 | no |

## C3 — gene/CRISPR — unseen-perturbation (LO-gene 10%)
Universal floor: cell-mean = 0.4937, linear-PCA = 0.2966, floor-mean = 0.3952  (unit = dataset, n = 5)

| family | model | Pearson-Δ | Δ vs floor-mean | Δ vs cell-mean | Δ vs linear-PCA | beats both? |
|---|---|---|---|---|---|---|
| OT | CINEMA-OT | 0.4579 | 0.0627 | −0.0359 | 0.1613 | no |
| Graph | AttentionPert | 0.2139 | −0.1813 | −0.2798 | −0.0827 | no |
| Graph | GEARS | 0.2004 | −0.1948 | −0.2934 | −0.0962 | no |
| Latent | scGen | 0.0796 | −0.3156 | −0.4141 | −0.2170 | no |
| Latent | CPA | 0.0735 | −0.3217 | −0.4203 | −0.2231 | no |
| Foundation | scGPT | 0.0582 | −0.3369 | −0.4355 | −0.2383 | no |
| Foundation | scFoundation | 0.0437 | −0.3514 | −0.4500 | −0.2528 | no |
| Hybrid | PertAdapt | −0.0014 | −0.3965 | −0.4951 | −0.2979 | no |
| Hybrid | STATE | −0.0206 | −0.4157 | −0.5143 | −0.3171 | no |

## C4 — complex/Frangieh — unseen-KO (modality, RNA)
Universal floor: cell-mean = 0.6612, linear-PCA = 0.2833, floor-mean = 0.4723  (unit = modality-fold, n = 2)

| family | model | Pearson-Δ | Δ vs floor-mean | Δ vs cell-mean | Δ vs linear-PCA | beats both? |
|---|---|---|---|---|---|---|
| Deterministic shift | linear-shift-KOemb | 0.6177 | 0.1454 | −0.0436 | 0.3343 | no |
| OT | CellOT | 0.5917 | 0.1195 | −0.0695 | 0.3084 | no |
| Latent | scGen | 0.5455 | 0.0732 | −0.1157 | 0.2622 | no |
| Latent | CPA | 0.5348 | 0.0625 | −0.1265 | 0.2514 | no |
| Graph | AttentionPert | 0.5113 | 0.0391 | −0.1499 | 0.2280 | no |
| Graph | GEARS | 0.4430 | −0.0293 | −0.2182 | 0.1597 | no |
| OT | scPRAM | 0.3152 | −0.1571 | −0.3460 | 0.0319 | no |
| Hybrid | STATE | 0.0254 | −0.4469 | −0.6358 | −0.2579 | no |

## C5 — small-mol/OP3 — unseen-compound
Universal floor: cell-mean = 0.1722, linear-PCA = 0.1498, floor-mean = 0.1610  (unit = compound, n = 28)

| family | model | Pearson-Δ | Δ vs floor-mean | Δ vs cell-mean | Δ vs linear-PCA | beats both? |
|---|---|---|---|---|---|---|
| OT | CINEMA-OT | 0.1718 | 0.0108 | −0.0004 | 0.0220 | no |
| Chemistry | FP-ridge | 0.1642 | 0.0032 | −0.0080 | 0.0144 | no |
| Latent | CPA | 0.1587 | −0.0023 | −0.0135 | 0.0089 | no |
| Chemistry | chemCPA | 0.1116 | −0.0494 | −0.0606 | −0.0382 | no |
| Latent | scGen | 0.0862 | −0.0748 | −0.0860 | −0.0636 | no |
| Hybrid | STATE | −0.0032 | −0.1642 | −0.1754 | −0.1530 | no |

## C5 — small-mol/OP3 — cell-context (LOCT)
Universal floor: cell-mean = 0.0250, linear-PCA = 0.2694, floor-mean = 0.1472  (unit = lineage, n = 4)

| family | model | Pearson-Δ | Δ vs floor-mean | Δ vs cell-mean | Δ vs linear-PCA | beats both? |
|---|---|---|---|---|---|---|
| Chemistry | FP-ridge | 0.3874 | 0.2402 | 0.3625 | 0.1180 | **yes** |
| OT | CINEMA-OT | 0.2533 | 0.1061 | 0.2283 | −0.0161 | no |
| Latent | scGen | 0.1770 | 0.0298 | 0.1520 | −0.0924 | no |
| Hybrid | STATE | 0.0555 | −0.0917 | 0.0305 | −0.2140 | no |
| Latent | CPA | 0.0536 | −0.0936 | 0.0286 | −0.2158 | no |

## Read (mechanical)

- Conditioned models that beat BOTH universal-floor members (point estimate): 2 of 35 (family,model)×task cells — CellOT@C2/donor (donor, +0.221); FP-ridge@C5/small-mol (cell-context, +0.240).
- Pattern matches the integrated finding: conditioning helps on **cell/donor-context transfer** (C2 CellOT donor-LODO; C5 FP-ridge LOCT) but **fails on unseen-perturbation extrapolation** (C3 LO-gene: every conditioned family is below floor; C5 unseen-compound: chemCPA/scGen below floor).
