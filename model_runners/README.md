# Model-family execution interfaces

These scripts run inside their model-family environments, separately from the CPU evaluation environment. `ivcbench.baselines.heavy.SubprocessAdapter` serializes the model input and invokes the corresponding runner.

```text
<model_env_python> model_runners/<runner>.py <input.npz> <output.npz>
```

The typical payload contains training expression/conditions, inference controls, gene identities and held-target labels. The ordinary prediction interface must not receive held treated expression; any transductive diagnostic operation or upstream shared feature processing is identified separately. Mean outputs are paired with observed targets in the evaluation layer.

Native/adapted/diagnostic status belongs to an **executed model–task operation**, not to the filename or model family. The final 58-entry panel and all omissions are documented in Table 2 and Supplementary Tables S15a-S15c. Folder membership alone is not evidence that a runner is evaluated.

Important execution distinctions:

- Native scGen T1/T2 remain. Historical CPA context and CPA/scGen fingerprint-to-latent interfaces are excluded under the native-only rule.
- Native chemCPA T5u uses `scripts/chemcpa_native_op3.py`, cpa-tools 0.8.8, molecular embeddings and the counterfactual prediction call; its existing three-seed mean profiles supply the reported CPA/chemCPA result.
- STATE T3/T4/T5 historically recovered `adata_real.h5ad` rather than model predictions. Those saved results are excluded. The corrected shared selector now requires exactly one `adata_pred.h5ad`; fixing source does not repair old results. Separate T1/T2 runners selected the correct prediction artifact.
- scFoundation genetic executions lacking a held-target-specific input and the pooled CellOT genetic map are excluded by their executed operation, not by a low-variance threshold.
- PertAdapt T2 is a study-written adaptation, not the published genetic predictor. It reconstructs stimulated profiles from their own frozen pooled cell embeddings, then takes held-control embeddings at inference. The stimulation/lineage inputs, binary GO mask and shared decoder differ from the published operation. Its exact local modules are in `vendor/pertadapt/` in the submission archive, with provenance and separate attribution; see `vendor/pertadapt/README.md` there. The T3 execution remains excluded.

Historical/excluded interfaces remain as compact source-level audit evidence, not an invitation to count their saved outputs in the panel. See [EXECUTION_AUDIT.md](../EXECUTION_AUDIT.md).

Model environments have incompatible Python, neural-framework and package requirements. Configure the runner paths, pretrained checkpoints and side inputs for the relevant original environment. Checkpoints and raw cell data are not redistributed. The exact recorded optimizer, trainable components and budget are in `results/_paper/supplementary_tables/Supplementary_Table_S15c.csv`; unrecorded seeds or dependencies are not inferred. No all-model training command or universal VRAM minimum is claimed.
