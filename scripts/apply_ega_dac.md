# EGA Data Access Committee (DAC) applications — **send Day 1**

EGA controlled-access approval typically takes **1–4 weeks** and is the single largest schedule risk
(PLAN.md §3, §8). These two block C1 (partially) and C2 (fully). Public clusters proceed meanwhile.

| Cluster | Dataset | Accession | What you predict / need it for |
|---|---|---|---|
| C1 | Cano-Gamez 2020 | **EGAS00001003215** | naive→memory CD4 state transfer (Axis 1) |
| C2 | Soskic 2022 | **EGAD00001008197** | 119-donor LODO + temporal (the whole cluster) |

## Checklist (per dataset)

1. Open the EGA page for the accession → find the **DAC** and its contact / application portal.
2. Prepare:
   - PI / institution, project title, **research use statement** (immune perturbation-prediction
     benchmark; model-evaluation, non-clinical, no re-identification).
   - Data-handling: secure on-prem storage (`/data1`, access-controlled), no redistribution.
   - Signed **Data Access Agreement** (institutional signatory may be required — start early).
3. Submit; record application date + ticket id in `data/manifest.csv`.
4. On approval: download via EGA download client (pyega3) into `data/<cluster>/<dataset>/`,
   checksum, then ingest with the dataset loader.

## Status tracker

| Dataset | Applied (date) | Approved (date) | Downloaded | Notes |
|---|---|---|---|---|
| Cano-Gamez EGAS00001003215 | | | | |
| Soskic EGAD00001008197 | | | | |

## Preprint / resource-stage (request from authors, not EGA)

Zhu 2025 · Moonen 2026 · Belk 2022 · Zhou 2023 — email corresponding authors; **Week-1 QC gate**
decides main vs supplementary (PLAN.md §8). Subsample large dumps to 100–200k cells/perturbation.
