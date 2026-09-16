# Descriptive family-by-task summary

Each row reports the highest-scoring recorded member of that family on that task,
selected retrospectively. The named member's paired margin uses the same units and
split-fixed binding floor as the census and S18. Diagnostic comparators have separate
rows. Intervals (4,000 unit-bootstrap draws; seed 0; at least eight units) are
unadjusted and conditional on member/floor selection, not family-level tests.
Smaller samples receive unit ranges, not significance claims. No a-priori
expectation or general model-selection recommendation is inferred from these results.

| Task | Family / role | Scored member | n | Score | Floor | Margin | Interval or unit range |
|---|---|---|---:|---:|---:|---:|---|
| T1 | Flow / conditioned | CellFlow | 8 | 0.798 | 0.780 | +0.018 | 95% CI [-0.011, +0.051] |
| T1 | Foundation / conditioned | scFoundation | 8 | 0.677 | 0.780 | -0.103 | 95% CI [-0.134, -0.073] |
| T1 | Hybrid / conditioned | STATE | 8 | 0.642 | 0.780 | -0.139 | 95% CI [-0.185, -0.095] |
| T1 | Latent / conditioned | scGen | 8 | 0.750 | 0.780 | -0.031 | 95% CI [-0.060, +0.000] |
| T1 | OT / conditioned | CellOT | 8 | 0.787 | 0.780 | +0.006 | 95% CI [-0.024, +0.038] |
| T2 | Flow / conditioned | PerturbNet | 106 | 0.255 | 0.260 | -0.005 | 95% CI [-0.013, +0.002] |
| T2 | Foundation / conditioned | scGPT | 106 | 0.250 | 0.260 | -0.010 | 95% CI [-0.012, -0.008] |
| T2 | Hybrid / conditioned | STATE | 106 | 0.136 | 0.260 | -0.124 | 95% CI [-0.135, -0.112] |
| T2 | Latent / conditioned | CPA | 106 | 0.208 | 0.260 | -0.052 | 95% CI [-0.059, -0.043] |
| T2 | OT / conditioned | CellOT | 106 | 0.367 | 0.260 | +0.107 | 95% CI [+0.086, +0.128] |
| T3 | Deterministic shift / diagnostic: linear-shift-KOemb | linear-shift-KOemb | 5 | 0.494 | 0.502 | -0.008 | range [-0.027, +0.002] |
| T3 | Flow / conditioned | PerturbNet | 5 | 0.371 | 0.502 | -0.131 | range [-0.195, +0.007] |
| T3 | Foundation / conditioned | scGPT | 5 | 0.169 | 0.502 | -0.333 | range [-0.625, -0.129] |
| T3 | Graph / conditioned | AttentionPert | 5 | 0.236 | 0.502 | -0.266 | range [-0.328, -0.115] |
| T3 | Hybrid / conditioned | PertAdapt | 5 | 0.240 | 0.502 | -0.262 | range [-0.348, -0.195] |
| T3 | Latent / conditioned | Biolord | 5 | 0.402 | 0.502 | -0.100 | range [-0.142, +0.044] |
| T4 | Deterministic shift / diagnostic: linear-shift-KOemb | linear-shift-KOemb | 2 | 0.614 | 0.660 | -0.045 | range [-0.057, -0.034] |
| T4 | Flow / conditioned | PerturbNet | 2 | 0.596 | 0.660 | -0.064 | range [-0.066, -0.062] |
| T4 | Foundation / conditioned | scGPT | 2 | 0.568 | 0.660 | -0.092 | range [-0.103, -0.080] |
| T4 | Graph / conditioned | AttentionPert | 2 | 0.503 | 0.660 | -0.157 | range [-0.158, -0.155] |
| T4 | Hybrid / conditioned | PertAdapt | 2 | 0.515 | 0.660 | -0.144 | range [-0.168, -0.121] |
| T4 | Latent / conditioned | Biolord | 2 | 0.612 | 0.660 | -0.048 | range [-0.051, -0.045] |
| T5u | Chemistry / conditioned | PRnet | 28 | 0.134 | 0.172 | -0.038 | 95% CI [-0.052, -0.023] |
| T5u | Chemistry / diagnostic: FP-ridge | FP-ridge | 28 | 0.164 | 0.172 | -0.008 | 95% CI [-0.031, +0.019] |
| T5u | Flow / conditioned | CellFlow | 28 | 0.166 | 0.172 | -0.006 | 95% CI [-0.019, +0.008] |
| T5u | Foundation / conditioned | scFoundation | 28 | 0.177 | 0.172 | +0.005 | 95% CI [-0.017, +0.029] |
| T5u | Hybrid / conditioned | STATE | 28 | 0.115 | 0.172 | -0.057 | 95% CI [-0.069, -0.045] |
| T5u | Latent / conditioned | Biolord | 28 | 0.127 | 0.172 | -0.046 | 95% CI [-0.077, -0.005] |
| T5c | Chemistry / conditioned | PRnet | 4 | 0.094 | 0.269 | -0.175 | range [-0.209, -0.143] |
| T5c | Chemistry / diagnostic: FP-ridge | FP-ridge | 4 | 0.387 | 0.269 | +0.118 | range [+0.105, +0.130] |
| T5c | Flow / conditioned | PerturbNet | 4 | 0.149 | 0.269 | -0.120 | range [-0.187, -0.037] |
| T5c | Foundation / conditioned | scGPT | 4 | 0.347 | 0.269 | +0.077 | range [+0.048, +0.093] |
| T5c | Hybrid / conditioned | STATE | 4 | 0.042 | 0.269 | -0.228 | range [-0.280, -0.189] |
| T5c | Latent / conditioned | scGen | 4 | 0.180 | 0.269 | -0.089 | range [-0.172, -0.018] |
| T5c | OT / conditioned | CellOT | 4 | 0.194 | 0.269 | -0.076 | range [-0.152, -0.004] |
