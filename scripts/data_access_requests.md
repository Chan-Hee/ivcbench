# Data Access Notes

Raw datasets remain under the terms of their original archives. This repository
does not redistribute controlled-access data.

For the benchmark datasets and their loaders, start with:

- [`data/README.md`](../data/README.md)
- [`scripts/datasets.csv`](datasets.csv)
- `bash scripts/download_all.sh --list`

## Manual Access

The public download scripts fetch only datasets that can be downloaded without a
data-access committee. The remaining datasets must be obtained directly from
their archives:

| Dataset | Accession | Access route | Local target |
|---|---|---|---|
| Cano-Gamez CD4+ effectorness | `EGAS00001003215` / `EGAD00001005290` | EGA DAC approval | `data/C1/cano_gamez/` |
| Chen FOXP3 Perturb-icCITE-seq | `PRJDB16517` / `E-GEAD-648` | DDBJ/GEA login | `data/C3/chen/` |

After manual download, record the archive, file names, checksums, and access
date in your local provenance notes. Do not commit raw controlled-access files.

## Request Template

Use this as a starting point when an archive or data provider asks for a free-text
research-use statement.

```text
We request access for non-clinical research on immune single-cell perturbation
prediction. The data will be used to evaluate model generalization across immune
cell context, donor, perturbation, and readout modality. We will not attempt
participant re-identification, will not redistribute controlled-access data, and
will store the data on access-controlled institutional compute/storage. Derived
aggregate benchmark results may be reported in publications and software
releases.
```
