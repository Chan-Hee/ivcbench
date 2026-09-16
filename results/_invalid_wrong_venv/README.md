# Quarantined outputs — do not use

Everything in this directory was produced by a run that imported the **wrong `ivcbench`
package**: the historical copy under the authors' `benchmark/` working tree rather than the
released `ivcbench/src/ivcbench`. The two differ in the evaluation code, so these scores are not
comparable with anything else in `results/`.

They are kept only so that the record of what was run is complete. **No number here enters the
census, any table, any figure or any reported statistic**, and nothing in the deposit reads this
directory. `scripts/_env_guard.py` now makes the same mistake impossible: `run_cluster.py`,
`assemble_cross_cluster.py` and `census_units.py` refuse to start under the wrong interpreter, and
any new entry point is expected to call the same guard.

Contents: two scGPT compute-budget sweeps (`scgpt_budget_4x`, `scgpt_budget_12x`, with their
timing JSON) and three STATE runs (`state_C3`, `state_C5`, `state_C5_seed1`). The valid
replacements live under `results/C3/`, `results/C5/` and `results/_paper/`.
