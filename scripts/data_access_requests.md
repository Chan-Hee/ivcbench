# Data access — routes + ready-to-send request drafts

> I (the assistant) **cannot send email or drive a browser**: no Gmail/email MCP is connected (only
> Slack/Notion/Drive), and there is no browser-automation tool. WebFetch is read-only and fails on
> authenticated pages (EGA/SCP/DUOS logins, DAC web forms). Sending requests to DACs / authors is
> also an outward action that should go through you. So below: the **exact access route per dataset**,
> and **copy-paste email drafts** for the ones that genuinely need a request. Send from your own mail
> (lch951022@gmail.com, or institutional), cc your PI (jyryu@ssu.ac.kr).

## Access routes — ALL 14 datasets (updated after web check, 2026-05-26)

| Cluster | Dataset | Route | Status |
|---|---|---|---|
| C1 | Kang (GSE96583) | public GEO | ✅ downloaded |
| C1 | Oesinghaus | **public** — Parse Biosciences + Allen Institute portal (~9.7M cells) | portal download (large) |
| C1 | Cano-Gamez (EGAS00001003215) | **EGA DAC** | ⚠ only dataset needing an email (Draft A) |
| C2 | Soskic | **public** processed h5ad (trynkalab.cog.sanger.ac.uk) | ⏳ downloading — no DAC |
| C3 | Shifrut / Schmidt | public GEO | ✅ downloaded |
| C3 | McCutcheon (GSE218985) | public GEO | ⏳ downloading |
| C3 | Chen (GSE255832) | public GEO (Perturb-icCITE-seq) | ⏳ downloading |
| C3 | Zhu (22M cells) | **public** — CZI Virtual Cells Platform | portal download (very large) |
| C3 | Moonen (4.1M cells) | **public** — CZI Virtual Cells Platform | portal download (large) |
| C4 | Frangieh (SCP1064) | **public** — Broad SCP, free Google login | login download (Draft D) |
| C4 | Belk (GSE203592) | public GEO (Seurat .rds, 2.7 GB) | ⏳ downloading |
| C4 | Zhou | published Nature 2023; accession TBD | confirm GEO accession |
| C5 | OP3 (GSE279945) | public GEO (14 GB) | ⏳ downloading |

**Net effect of the web check:** 13 of 14 datasets are publicly accessible. **Only Cano-Gamez still
needs an EGA DAC**, and it backs just one C1 sub-axis (naïve→memory state transfer) — so every cluster
can proceed without waiting on any committee or author. The preprint datasets (Zhu, Moonen) are open
on the CZI Virtual Cells Platform, and Oesinghaus is open via Parse/Allen — so the "email the authors"
path (Draft C) is now only a fallback. The large portal datasets (Oesinghaus 9.7M, Zhu 22M, Moonen
4.1M, OP3 14 GB) should be **subsampled to ~100–200k cells/perturbation on ingest** (PLAN.md §3).

### What still needs YOUR credentials (I can't: no browser/login/DAC)

These 5 are public-or-gated but require a login/account/DAC I cannot complete. Everything else is
already downloading with no auth (incl. **Frangieh**, now grabbed from the scPerturb Zenodo mirror —
no SCP login needed).

**1. Zhu (C3) + Moonen (C3) — CZI Virtual Cells Platform CLI (free account).**
Raw Zhu on GEO is `GSE314342_RAW.tar` = **160 GB** (impractical; we only need a subsample). Use the
VCP CLI on the processed AnnData instead:
```bash
pip install vcp-cli            # https://chanzuckerberg.github.io/vcp-cli/
vcp login                      # free account at virtualcellmodels.cziscience.com
vcp data search "Primary Human CD4+ T Cell Perturb-seq" --exact
vcp data download --query "<dataset id from search>"   # pick per-donor/condition files; subsample on ingest
```

**2. Oesinghaus (C1) — Parse Biosciences portal (registration).** The 9.7M-cell expression matrix is
at https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/ (sign-up).
The Allen page only exposes DEG/GSEA/CellTypist derivatives, not the count matrix.

**3. Cano-Gamez (C1) — EGA DAC.** Email Draft A above. (Backs only one C1 sub-axis.)

**4. Zhou (C4) — confirm GEO accession** from the Nature 2023 paper's Data Availability
(https://www.nature.com/articles/s41586-023-06733-x); candidates GSE216800 / GSE216909. External
in-vivo reference only (sensitivity point), so lowest priority.

> Hand me any of these once downloaded (drop files under `data/<cluster>/<dataset>/`) and I'll write
> the loader. Or share a CZI account token / SCP export and I can script the rest.

---

## Draft A — EGA DAC application (Cano-Gamez, EGAS00001003215)

> Get the exact Data Access Committee contact from the EGA dataset page
> (https://ega-archive.org/studies/EGAS00001003215 → "Data Access Committee"). Many Sanger datasets
> route through a study DAC; the page names the contact/email or links an application form.

**Subject:** Data access request — EGAS00001003215 (Cano-Gamez et al., 2020)

Dear Data Access Committee,

I am Chanhee Lee, a researcher at the School of Systems Biomedical Science, Soongsil University
(PI: Prof. Jae Yong Ryu, jyryu@ssu.ac.kr). We are preparing a benchmarking study of single-cell
perturbation-prediction models in immunology ("Toward Immune Virtual Cells"), evaluating how well
current models predict cytokine responses of CD4+ T cells.

We would like to request access to the single-cell RNA-seq data deposited under accession
EGAS00001003215 / EGAD00001005290 (Cano-Gamez et al., Nat Commun 2020), to evaluate naïve→memory CD4+
T-cell state-transfer prediction. The use is non-clinical, computational model evaluation only; we
will not attempt re-identification and will not redistribute the data.

Data handling: the data will be stored on an access-controlled on-premises server at Soongsil
University, used only by the named researchers, and deleted on project completion. We are happy to
complete and sign your Data Access Agreement and provide any institutional documentation required.

Could you please advise on the application procedure and the Data Access Agreement? Thank you for
your time.

Best regards,
Chanhee Lee — Soongsil University (lch951022@gmail.com)
PI: Prof. Jae Yong Ryu (jyryu@ssu.ac.kr)

---

## Draft C — Preprint / author data request (Zhu, Moonen, Belk, Zhou, Oesinghaus)

> One template; fill the [DATASET]/[REFERENCE] and the corresponding author from the preprint's
> "Contact"/"Lead contact" or the bioRxiv author list. Check GEO first — some (e.g. Belk, Zhou) may
> already have a public accession, in which case no email is needed.

**Subject:** Data request — [DATASET] for an immune perturbation-prediction benchmark

Dear Dr. [Corresponding Author],

I am Chanhee Lee (Soongsil University; PI Prof. Jae Yong Ryu). We are benchmarking single-cell
perturbation-prediction models on immune datasets for a study titled "Toward Immune Virtual Cells,"
and your dataset [REFERENCE] is one of the most relevant resources for the [gene-perturbation /
in vitro→in vivo] axis we evaluate.

Would it be possible to access the processed single-cell data (count matrix + cell/perturbation
metadata) from this study? We will use it solely for computational model evaluation, cite your work,
and not redistribute the data. We are glad to follow any data-use terms or acknowledgement you prefer.

Thank you very much for considering this request.

Best regards,
Chanhee Lee — Soongsil University (lch951022@gmail.com)
PI: Prof. Jae Yong Ryu (jyryu@ssu.ac.kr)

---

## Draft D — Frangieh (SCP1064) download how-to (no email needed)

1. Create a free account / sign in at https://singlecell.broadinstitute.org (Google login).
2. Open study SCP1064 (Frangieh 2021 Perturb-CITE-seq).
3. "Download" tab → agree to terms → download the RNA count matrix, the 20-protein (ADT) matrix, and
   the cell metadata (perturbation/guide assignments). (Bulk download needs the SCP auth token shown
   in the Download tab — not scriptable without login, so do this once in the browser.)
4. Place files under `data/C4/frangieh/`; raw sequencing (if needed) is via DUOS-000124.
