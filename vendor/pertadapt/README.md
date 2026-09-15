# Study-local PertAdapt-inspired components

These are the exact three local source modules used by the retained **adapted T2**
runner, not an installation of native PertAdapt. They are included so the runner's
custom imports resolve inside the submission archive; installing the upstream
repository alone would not provide this study-specific module surface.

`__init__.py`, `pertadapt_modules.py`, and `build_go_mask.py` are preserved
byte-for-byte from the historical study-local extraction. Original explanatory
comments describe that extraction, not an audited claim that the complete
prediction operation is native or that a sampled mask check proves whole-file
identity. The current operation-level scope below takes precedence for interpreting
the submitted results. `UPSTREAM_COMMIT.txt` identifies the upstream source used
when the extraction was authored.

## Operation and provenance

Published method: Bai D, Song L, Xing EP. *PertAdapt: unlocking single-cell
foundation models for genetic perturbation prediction via condition-sensitive
adaptation*. Bioinformatics 2026;42(Supplement_1):btag307.
[Article](https://doi.org/10.1093/bioinformatics/btag307);
[upstream code](https://github.com/BaiDing1234/PertAdapt/tree/53bb7f05b5b21837d5851b7aea51ee29eae5fa3b).

The original predictor combines per-gene frozen representations with a genetic
condition encoder and trains on control-input/perturbed-target pairs. The study's
T2 runner instead projects pooled cell embeddings across learned gene embeddings,
uses a single stimulation token and lineage embedding, a binary GO co-membership
mask, and a shared decoder with a pooled training-control anchor. It reconstructs
stimulated profiles from their own embeddings during training, then applies the
head to held-donor controls. Genes outside its 256-gene response panel retain the
training-control mean. This result does not measure native PertAdapt capability.
The historical T3 execution remains excluded for missing held-gene conditioning.

The public module surface used by T2 is `GOMaskedPertAdapter`, `loss_adapt`,
`additive_go_mask`, and `load_gene2go`. `model_runners/pertadapt_soskic_runner.py`
places this archive's `vendor/` on its import path. Torch and the original
scFoundation environment/checkpoint remain external requirements. Set
`IVCBENCH_SCFOUNDATION_DIR`, `IVCBENCH_SCFOUNDATION_CKPT`, and `IVCBENCH_GENE2GO`
explicitly for a fitting run; historical machine-local defaults are not portable.
No model fitting is needed for CPU result replay.

## Attribution and redistribution status

These study-local files derive their attention/loss design and parts of their
implementation from the cited PertAdapt source. The study's integration, mask
builder and T2 head are distinct from that source. No upstream repository, model
weights, annotations or raw cells are bundled here.

No standalone upstream code license was found in the pinned local source or by
the GitHub license endpoint checked on 2026-09-09. **The parent project's MIT
license is not asserted to relicense upstream-derived material.** These files are
preserved as implementation evidence in this private revision snapshot. Any
permission needed for a subsequent public redistribution remains to be resolved;
this package and its internal review do not establish such permission. No public
upload has been performed.
