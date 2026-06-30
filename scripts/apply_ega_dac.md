# EGA Data Access Committee (DAC) Applications

EGA controlled-access approval typically takes **1–4 weeks** and is the single largest schedule risk
for affected datasets. Public datasets can be downloaded independently while approval is pending.

> **Soskic 2022 is NOT DAC-gated for this benchmark:** the scored donor-axis analysis uses the public processed CD4 activation h5ads from the Trynka-lab object store (106 paired donors); the EGA raw reads (EGAD00001008197) are controlled-access but are not required to reproduce the benchmark. The remaining genuine EGA-DAC dataset is the surveyed-only Cano-Gamez 2020.

| Cluster | Dataset | Accession | What you predict / need it for |
|---|---|---|---|
| C1 | Cano-Gamez 2020 | **EGAS00001003215** | naive→memory CD4 state transfer (Axis 1) |

## Checklist (per dataset)

1. Open the EGA page for the accession → find the **DAC** and its contact / application portal.
2. Prepare:
   - PI / institution, project title, **research use statement** (immune perturbation-prediction
     benchmark; model-evaluation, non-clinical, no re-identification).
   - Data-handling: secure institutional storage, access controls, no redistribution.
   - Signed **Data Access Agreement** (institutional signatory may be required — start early).
3. Submit; record application date + ticket id in `data/manifest.csv`.
4. On approval: download via EGA download client (pyega3) into `data/<cluster>/<dataset>/`,
   checksum, then ingest with the dataset loader.

## Status tracker

| Dataset | Applied (date) | Approved (date) | Downloaded | Notes |
|---|---|---|---|---|
| Cano-Gamez EGAS00001003215 | | | | |

## Preprint / resource-stage (request from authors, not EGA)

Zhu 2025, Moonen 2026, Belk 2022, and Zhou 2023 are not fetched by this EGA
note. See `data/README.md` and `scripts/datasets.csv` for their current access
routes and benchmark role.
