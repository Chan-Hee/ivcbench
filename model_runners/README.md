# Heavy-baseline model runners

Each file here is a standalone script run **inside a pinned conda env** (not the GPU-free core
`.venv`) by `ivcbench.baselines.heavy.SubprocessAdapter`. The adapter serialises a leak-safe payload
and calls:

```
<env_python> model_runners/<model>_runner.py  <in.npz>  <out.npz>
```

## Env map (existing envs, discovered 2026-05-26)
| runner | conda env | key package |
|---|---|---|
| `gears_runner.py` | `scgpt` | cell-gears 0.0.2 (+ torch-geometric) |
| `scgpt_runner.py` | `scgpt` | scgpt 0.2.4 |
| `attentionpert_runner.py` | `scgpt` | (torch-geometric) |
| `uce_runner.py` | `scgpt` | UCE weights |
| `scgen_runner.py` | `scperturbench_eval` | pertpy 0.10 → `pt.tl.Scgen` |
| `cpa_runner.py` | `scperturbench_eval` | cpa |
| `cinemaot_runner.py` | `scperturbench_eval` | pertpy → `pt.tl.Cinemaot` |
| `cellot_runner.py` | `scperturbench_eval` | cellot repo |
| `scpram_runner.py` / `scpram_soskic_runner.py` | `ivc-scpram` | scpram 0.0.3 (`--no-deps` on the ivc-cpa torch2.0/cu117 stack) — 2nd conditioned OT model (Kang C1 LOCT / Soskic C2 LODO) |
| `state_runner.py` | `scfoundation` | arc state |

## Input payload (`in.npz`, allow_pickle)
- `X_train` (n_train, n_genes) float32 — training expression (log-normalized HVG)
- `is_control_train` (n_train,) bool
- `pert_train` (n_train,) str — perturbation label per training cell (target gene; `control` for NTC)
- `X_ctrl_inf` (n_ctrl, n_genes) float32 — control cells the model may see at inference (the Δ baseline)
- `genes` (n_genes,) str
- `test_perts` (n_test,) str — held-out perturbation label per test cell (LABELS ONLY; never expression)
- `gene_embedding_keys` / `gene_embedding_vals` — optional gene-side representation (for `adapted` models)

## Output (`out.npz`)
- `pred_perts` (k,) str — the held perturbations predicted
- `pred_means` (k, n_genes) float32 — predicted post-perturbation mean profile per held perturbation

The adapter tiles `pred_means` onto the test cells by `test_perts`, then the GPU-free core scores the
four axes. The runner must NOT receive or read held-out treated expression — the leak boundary holds
on the model side too.

## Support assets
- **gene2go (GEARS/AttentionPert)**: `$IVCBENCH_GENE2GO` or `benchmark/data/_assets/gears/gene2go_all.pkl`
  (the canonical GEARS gene2go_all.pkl, 9.46 MB). Seed it once from the GEARS dataverse, or copy from a
  prior GEARS run. The runner copies it into each per-run PertData dir so cell-gears does not try to
  download it.
- **Perturbed genes must be in the HVG panel** for any gene-side model: `data/preprocess.py` force-keeps
  the perturbed target genes (`select_hvg(..., force_idx=...)`) — required so GEARS `get_pert_idx` can
  locate the perturbed gene.
- Tune epochs via `$IVCBENCH_GEARS_EPOCHS` (default 15; smoke tests use 2).

## Status (2026-05-27)
3 heavy models proven end-to-end (leak-safe, sane Pearson-Δ vs floors on Schmidt 50% LO-gene):
- **GEARS** (`gears_runner.py`, scgpt env) — 0.254. In the C3 roster; full sweep QC-GREEN.
- **scGPT** (`scgpt_runner.py`, scgpt env, pretrained `$IVCBENCH_SCGPT_MODEL_DIR`) — 0.064 @2ep (more epochs ↑).
- **scGen** (`scgen_runner.py`, scperturbench_eval env) — 0.251 (`adapted`: latent-δ on control-only PCA
  gene-embedding → decode `module.as_bound().generative`). Needs `train(accelerator="cpu")` (CPU-only JAX).

All three sit below cell-mean (0.549) → the O1 finding (gene-side models don't beat the mean-shift on
focused primary-T panels) holds across graph/foundation/latent families.

### Remaining roster (each a focused build)
- **CPA** — dedicated env `ivc-cpa`: `conda create -n ivc-cpa python=3.10` + `pip install cpa-tools` +
  `pip install 'pyarrow<17'` (ray 2.9 needs the old pyarrow PyExtensionType). Do NOT install into
  scperturbench_eval (cpa-tools downgrades torch/scvi/anndata and breaks scGen). DECODE PATH FOUND:
  `cpa._module.CPAModule.generative(z, library)` takes a single latent like scGen → reuse the scGen
  adapted template (latent-δ regression on gene-embedding; get z+library via get_latent_representation /
  module.inference).
- **AttentionPert** — graph; `attnpert` (no setup.py → sys.path the source dir) needs per-dataset
  `gene2vec.npy` (gen from `gene2vec_dim_200_iter_9_w2v.txt`, Gaussian fallback) + a leak-safe-by-
  construction predict (its test-loader design does not fit cleanly — needs design).
- **UCE / STATE** — external weight downloads (UCE ckpt; arc-state) not present locally.
