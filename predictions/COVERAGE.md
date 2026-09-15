# Prediction coverage

The revision manifest is `results/_paper/census_bundle_manifest.csv`. It selects **1,362 mean-profile bundles: 1,110 model and 252 simple-reference inputs**, yielding 47 model-by-task evaluations and 1,605 model analysis-unit rows. See `CANONICAL_NUMBERS.json`, `census_unit_scores.csv` and the interface records for exact membership.

The six settings contain 8 / 9 / 7 / 7 / 9 / 7 entries (T1 / T2 / T3 / T4 / T5c / T5u). Status counts are 34 native, seven adapted and six diagnostic. A bundle may contain several held genes/compounds; it is not an independent replicate.

The default `make reproduce` path scores only this manifest before running the independent consistency gate. Historical, diagnostic and non-selected bundles remain as provenance/reference inputs; do not include every NPZ in an inferential panel simply because it exists. In particular, non-native CPA/scGen and wrongly recovered STATE T3/T4/T5 outputs are excluded. Native chemCPA provides the held-compound CPA/chemCPA result. Correctly recovered STATE T1/T2 remain eligible.

These compact inputs retain predicted/observed per-stratum means, controls, gene order and scoring-mask metadata. They reproduce Pearson-Δ. They do **not** retain the per-cell clouds and training-only PCA basis needed to reproduce energy distance. Cell-level rank-program observations are supplied separately as caches; scores of a mean profile are not substituted for mean observed per-cell scores.

For the same task, target vectors and exclusion masks must agree across models. The consistency checks enforce this contract. For Soskic, aligned C2 metric metadata accompany the revision; the preserved numeric mean predictions are unchanged. Seed provenance is task-specific and is recorded in Supplementary Note S2; do not infer that every single saved bundle represents a multi-seed ensemble.

Run `make reproduce` at the package root. A passing gate ends with `DEPOSIT CONSISTENCY: PASS`.
