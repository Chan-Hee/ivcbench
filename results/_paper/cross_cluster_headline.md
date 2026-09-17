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
| Flow | CellFlow | 0.7985 | 0.1135 | 0.2089 | 0.0181 | **yes** |
| Flow | PerturbNet | 0.7868 | 0.1019 | 0.1973 | 0.0065 | **yes** |
| OT | CellOT | 0.7865 | 0.1016 | 0.1969 | 0.0062 | **yes** |
| OT | scPRAM | 0.7507 | 0.0658 | 0.1612 | −0.0296 | no |
| Latent | scGen | 0.7497 | 0.0647 | 0.1601 | −0.0307 | no |
| Foundation | scFoundation | 0.6773 | −0.0076 | 0.0878 | −0.1030 | no |
| Hybrid | STATE | 0.6416 | −0.0434 | 0.0520 | −0.1388 | no |
| Foundation | scGPT | 0.5855 | −0.0995 | −0.0041 | −0.1949 | no |
| Latent | CPA | 0.3864 | −0.2985 | −0.2031 | −0.3939 | no |

## C2 — donor/Soskic — donor (LODO)
Universal floor: cell-mean = 0.2598, linear-PCA = 0.0362, floor-mean = 0.1480  (unit = donor, n = 106)

| family | model | Pearson-Δ | Δ vs floor-mean | Δ vs cell-mean | Δ vs linear-PCA | beats both? |
|---|---|---|---|---|---|---|
| OT | CellOT | 0.3666 | 0.2186 | 0.1068 | 0.3304 | **yes** |
| Flow | PerturbNet | 0.2546 | 0.1065 | −0.0053 | 0.2183 | no |
| Foundation | scGPT | 0.2501 | 0.1021 | −0.0097 | 0.2139 | no |
| Flow | CellFlow | 0.2373 | 0.0893 | −0.0225 | 0.2011 | no |
| Latent | CPA | 0.2082 | 0.0601 | −0.0516 | 0.1719 | no |
| OT | scPRAM | 0.1592 | 0.0111 | −0.1006 | 0.1229 | no |
| Latent | scGen | 0.1470 | −0.0010 | −0.1128 | 0.1108 | no |
| Hybrid | STATE | 0.1359 | −0.0121 | −0.1239 | 0.0997 | no |
| Foundation | scFoundation | 0.1197 | −0.0283 | −0.1401 | 0.0835 | no |

## C3 — gene/CRISPR — unseen-perturbation (LO-gene 10%)
Universal floor: cell-mean = 0.5020, linear-PCA = 0.3012, floor-mean = 0.4016  (unit = dataset, n = 5)

| family | model | Pearson-Δ | Δ vs floor-mean | Δ vs cell-mean | Δ vs linear-PCA | beats both? |
|---|---|---|---|---|---|---|
| Deterministic shift | linear-shift-KOemb | 0.4941 | 0.0925 | −0.0079 | 0.1929 | no |
| Latent | Biolord | 0.4019 | 0.0003 | −0.1001 | 0.1007 | no |
| Flow | PerturbNet | 0.3712 | −0.0304 | −0.1308 | 0.0700 | no |
| Flow | CellFlow | 0.3271 | −0.0745 | −0.1749 | 0.0259 | no |
| Hybrid | PertAdapt | 0.2401 | −0.1615 | −0.2619 | −0.0611 | no |
| Graph | AttentionPert | 0.2356 | −0.1660 | −0.2664 | −0.0656 | no |
| Graph | GEARS | 0.2129 | −0.1887 | −0.2891 | −0.0883 | no |
| Foundation | scGPT | 0.1688 | −0.2328 | −0.3332 | −0.1324 | no |
| Hybrid | STATE | 0.1435 | −0.2581 | −0.3585 | −0.1577 | no |
| Foundation | scFoundation | 0.1303 | −0.2713 | −0.3717 | −0.1709 | no |

## C4 — complex/Frangieh — unseen-KO (modality, RNA)
Universal floor: cell-mean = 0.6598, linear-PCA = 0.2776, floor-mean = 0.4687  (unit = modality-fold, n = 2)

| family | model | Pearson-Δ | Δ vs floor-mean | Δ vs cell-mean | Δ vs linear-PCA | beats both? |
|---|---|---|---|---|---|---|
| Deterministic shift | linear-shift-KOemb | 0.6143 | 0.1456 | −0.0455 | 0.3368 | no |
| Latent | Biolord | 0.6115 | 0.1428 | −0.0483 | 0.3339 | no |
| Flow | PerturbNet | 0.5958 | 0.1272 | −0.0640 | 0.3183 | no |
| Foundation | scGPT | 0.5682 | 0.0995 | −0.0916 | 0.2907 | no |
| Hybrid | PertAdapt | 0.5154 | 0.0467 | −0.1444 | 0.2378 | no |
| Graph | AttentionPert | 0.5032 | 0.0345 | −0.1566 | 0.2256 | no |
| Flow | CellFlow | 0.4876 | 0.0189 | −0.1722 | 0.2100 | no |
| Graph | GEARS | 0.4376 | −0.0311 | −0.2223 | 0.1600 | no |
| Foundation | scFoundation | 0.3028 | −0.1659 | −0.3570 | 0.0253 | no |
| Hybrid | STATE | 0.2326 | −0.2361 | −0.4272 | −0.0449 | no |

## C5 — small-mol/OP3 — unseen-compound
Universal floor: cell-mean = 0.1722, linear-PCA = 0.1498, floor-mean = 0.1610  (unit = compound, n = 28)

| family | model | Pearson-Δ | Δ vs floor-mean | Δ vs cell-mean | Δ vs linear-PCA | beats both? |
|---|---|---|---|---|---|---|
| Foundation | scFoundation | 0.1768 | 0.0159 | 0.0047 | 0.0270 | **yes** |
| Flow | CellFlow | 0.1661 | 0.0051 | −0.0061 | 0.0163 | no |
| Chemistry | FP-ridge | 0.1642 | 0.0032 | −0.0080 | 0.0144 | no |
| Flow | PerturbNet | 0.1613 | 0.0003 | −0.0109 | 0.0115 | no |
| Foundation | scGPT | 0.1605 | −0.0005 | −0.0117 | 0.0107 | no |
| Chemistry | PRnet | 0.1343 | −0.0267 | −0.0379 | −0.0155 | no |
| Latent | Biolord | 0.1267 | −0.0343 | −0.0455 | −0.0231 | no |
| Hybrid | STATE | 0.1150 | −0.0460 | −0.0572 | −0.0348 | no |
| Latent | CPA | 0.1000 | −0.0610 | −0.0722 | −0.0498 | no |

## C5 — small-mol/OP3 — cell-context (LOCT)
Universal floor: cell-mean = 0.0250, linear-PCA = 0.2694, floor-mean = 0.1472  (unit = lineage, n = 4)

| family | model | Pearson-Δ | Δ vs floor-mean | Δ vs cell-mean | Δ vs linear-PCA | beats both? |
|---|---|---|---|---|---|---|
| Chemistry | FP-ridge | 0.3874 | 0.2402 | 0.3625 | 0.1180 | **yes** |
| Foundation | scGPT | 0.3467 | 0.1995 | 0.3218 | 0.0773 | **yes** |
| Foundation | scFoundation | 0.3291 | 0.1819 | 0.3041 | 0.0597 | **yes** |
| OT | CellOT | 0.1936 | 0.0464 | 0.1687 | −0.0758 | no |
| OT | scPRAM | 0.1812 | 0.0340 | 0.1562 | −0.0882 | no |
| Latent | scGen | 0.1804 | 0.0332 | 0.1554 | −0.0890 | no |
| Flow | PerturbNet | 0.1491 | 0.0019 | 0.1241 | −0.1203 | no |
| Flow | CellFlow | 0.1038 | −0.0434 | 0.0788 | −0.1656 | no |
| Chemistry | PRnet | 0.0944 | −0.0528 | 0.0695 | −0.1750 | no |
| Latent | CPA | 0.0606 | −0.0866 | 0.0356 | −0.2088 | no |
| Hybrid | STATE | 0.0417 | −0.1055 | 0.0167 | −0.2278 | no |

## Read (mechanical)

- Conditioned models that beat BOTH universal-floor members (point estimate): 8 of 58 (family,model)×task cells — CellFlow@C1/cytokine (cell-context, +0.114); CellOT@C1/cytokine (cell-context, +0.102); PerturbNet@C1/cytokine (cell-context, +0.102); CellOT@C2/donor (donor, +0.219); scFoundation@C5/small-mol (unseen-compound, +0.016); FP-ridge@C5/small-mol (cell-context, +0.240); scFoundation@C5/small-mol (cell-context, +0.182); scGPT@C5/small-mol (cell-context, +0.200).
- Pattern matches the integrated finding: conditioning helps on **cell/donor-context transfer** (C2 CellOT donor-LODO; C5 FP-ridge LOCT) but **fails on unseen-perturbation extrapolation** (C3 LO-gene: every conditioned family is below floor; C5 unseen-compound: chemCPA/scGen below floor).
