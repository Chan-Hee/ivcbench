# STATE prediction-artifact recovery audit (9 September 2026)

The earlier 50-entry census is superseded by a 47-entry census (34 native,
7 adapted, 6 diagnostic). STATE remains evaluated on T1/T2 only. This is an
implementation correction, not a score-based or low-variance exclusion.

Evidence inspected directly:

- Historical `benchmark/model_runners/state_runner.py` and `state_c5_runner.py`
  aggregate the lexicographically last `*.h5ad` below the STATE output directory.
  Their copied `ivcbench/model_runners` versions had the same defect.
- The installed arc-state `_cli/_tx/_predict.py`, lines 498–502, always saves
  both `adata_pred.h5ad` and `adata_real.h5ad` in the same evaluation directory,
  including with `--predict-only`. The latter sorts last.
- Both runners construct their test/query arm from control cells relabelled with
  each held gene/compound. Thus the recovered `adata_real` is not the model's
  predicted output. In a seen-compound split, duplicate training/query labels can
  additionally mix observed training responses into that aggregate. Neither
  result evaluates the declared STATE predictor.
- The generic runner supplies T3 and the previously excluded T4; the compound
  runner supplies T5c and T5u through `STATEc5` in `clusters/spec.py`.
- The separate T1/T2 runners explicitly prefer `adata_pred.h5ad`; they do not
  have the demonstrated normal-case selection defect. Their census entries stay.

Action: exclude STATE T3/T5c/T5u from the final panel, retain historical records
outside the panel, remove their program readouts and chemical-distance row, and
recalculate all downstream counts and the full eligible multiplicity family.
Do not claim these outputs demonstrate STATE failure. No replacement drug fit
is performed. The correct prediction artifacts were in temporary workspaces
removed by the historical runner, so they cannot be recovered from its compact
mean bundles. Fixing a runner does not retroactively repair its saved results.

The release runner now requires exactly one named `adata_pred.h5ad`, with no
generic fallback. Regression tests cover real-file ordering, missing prediction,
ambiguous checkpoints and exclusion from the current census. The unmodified
historical runner copies remain in `benchmark/` as evidence.

The unrelated scFoundation T3/T4 exclusion is supported by its executed source:
condition vocabulary is built from training perturbations only; a held gene is
assigned an all-zero condition vector. It therefore has no held-target-specific
input in that implementation. Small prediction variance is not the criterion,
and the predicted output is not necessarily equal to the control mean.
