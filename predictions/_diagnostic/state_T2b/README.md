# STATE T2 (Soskic donor LODO) — DIAGNOSTIC ONLY, NOT CENSUS EVIDENCE

39 bundles deposited 2026-09-15 02:10–05:37 UTC by `state_soskic_runner.py`
(jobs `state_T2_batch_2/3`, `state_T2b_2`, `state_T2b_5`).

They are withheld from the census because that runner puts the held donor's control
cells into the SAME AnnData as the training donors, and the installed few-shot loader
(`cell_load/data_modules/perturbation_dataloader.py:789-812`) places every control cell of
a lineage into the train subset. With `should_yield_control_cells: true` those held
controls become training targets. That is a fit = train_idx contract violation: the held
donor is not a leak of held TREATED cells, but its controls were used in fitting.
`basal_mapping_strategy=batch` does not fix it — it changes which control is drawn as the
basal, not which cells are in the train subset.

The replacement is `state_soskic_split_runner.py`, which writes a training-only AnnData/TOML
and an inference-only AnnData/TOML and goes through the official `predict --toml` path.
Its output lands in `predictions/v2_native` under job ids `state_T2c_*`.

Keep these for the record. Do not quote their mean as the cell's value, and do not compare
a 13- or 26-donor subset mean against the 106-donor floor.
