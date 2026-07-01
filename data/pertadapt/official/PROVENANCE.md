# PertAdapt Official Artifact Provenance

This directory is the default local location audited by
`scripts/pertadapt_validate.py` for the optional PertAdapt published-anchor
validation tier.

The code release does not ship the official PertAdapt/scFoundation artifacts.
To run the gate, obtain the required files from the upstream sources and either
place them here or set the corresponding environment variables:

| Artifact | Default path | Override |
|---|---|---|
| PertAdapt official gene-similarity mask | `data/pertadapt/official/go_mask_19264.npz` | `IVCBENCH_PA_GO_MASK_NPZ` |
| 19264-gene Adamson GEARS dataset directory | `data/pertadapt/official/gse90546_k562_63587_19264_10k_log1p_withtotalcount/` | `IVCBENCH_PA_ADAMSON_DIR` |
| scFoundation `cell` checkpoint | `scFoundation/models.ckpt` | `IVCBENCH_SCFOUNDATION_CKPT` |
| PertAdapt upstream repository | `vendor/pertadapt_repo/` | `IVCBENCH_PA_REPO` |

Expected Adamson file checked by the validation gate:

```text
data/pertadapt/official/gse90546_k562_63587_19264_10k_log1p_withtotalcount/perturb_processed.h5ad
```

`pertadapt_validate.py` deliberately returns `READY=false` when these artifacts
are absent. That is a provenance guard, not a GPU-free reproduction failure: the
paper's public reproduction path uses the deposited prediction bundles and
result tables described in `REPRODUCE.md`.
