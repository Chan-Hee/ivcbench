# Prediction coverage

The revision manifest is `results/_paper/census_bundle_manifest.csv`. It selects **1,401 mean-profile bundles: 1,149 model and 252 simple-reference inputs**, yielding 58 model-by-task evaluations and 1,698 model analysis-unit rows. See `CANONICAL_NUMBERS.json`, `census_unit_scores.csv` and the interface records for exact membership.

The six settings contain 9 / 9 / 10 / 10 / 11 / 9 entries (T1 / T2 / T3 / T4 / T5c / T5u). Status counts are 46 native, eight adapted and four diagnostic. A bundle may contain several held genes/compounds; it is not an independent replicate.

The default `make reproduce` path scores only this manifest before running the independent consistency gate. Historical, diagnostic and non-selected bundles remain as provenance/reference inputs; do not include every NPZ in an inferential panel simply because it exists. In particular, non-native CPA and scGen outputs are excluded, and so is the superseded chemCPA bundle: native chemCPA provides the held-compound CPA/chemCPA result from its re-run. The STATE T3/T4/T5 outputs that were wrongly recovered have been re-run through the corrected runner, so STATE is eligible on all six settings; the superseded bundles stay registered in `withdrawn_bundles.csv` by path and sha256.

These compact inputs retain predicted/observed per-stratum means, controls, gene order and scoring-mask metadata. They reproduce Pearson-Δ. They do **not** retain the per-cell clouds and training-only PCA basis needed to reproduce energy distance. Cell-level rank-program observations are supplied separately as caches; scores of a mean profile are not substituted for mean observed per-cell scores.

For the same task, target vectors and exclusion masks must agree across models. The consistency checks enforce this contract. For Soskic, aligned C2 metric metadata accompany the revision; the preserved numeric mean predictions are unchanged. Seed provenance is task-specific and is recorded in Supplementary Note S2; do not infer that every single saved bundle represents a multi-seed ensemble.

Run `make reproduce` at the package root. A passing gate ends with `DEPOSIT CONSISTENCY: PASS`.
