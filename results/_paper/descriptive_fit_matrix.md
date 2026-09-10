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
| T1 | Hybrid / conditioned | STATE | 8 | 0.708 | 0.780 | -0.072 | 95% CI [-0.102, -0.044] |
| T1 | Latent / conditioned | scGen | 8 | 0.750 | 0.780 | -0.031 | 95% CI [-0.060, +0.000] |
| T1 | OT / conditioned | CellOT | 8 | 0.787 | 0.780 | +0.006 | 95% CI [-0.024, +0.038] |
| T2 | Flow / conditioned | CellFlow | 106 | 0.237 | 0.260 | -0.022 | 95% CI [-0.033, -0.011] |
| T2 | Foundation / conditioned | scGPT | 106 | 0.250 | 0.260 | -0.010 | 95% CI [-0.012, -0.008] |
| T2 | Hybrid / conditioned | PertAdapt | 106 | 0.220 | 0.260 | -0.040 | 95% CI [-0.053, -0.028] |
| T2 | Latent / conditioned | scGen | 106 | 0.147 | 0.260 | -0.113 | 95% CI [-0.128, -0.099] |
| T2 | OT / conditioned | CellOT | 106 | 0.367 | 0.260 | +0.107 | 95% CI [+0.086, +0.128] |
| T3 | Flow / conditioned | PerturbNet | 5 | 0.362 | 0.494 | -0.132 | range [-0.196, +0.005] |
| T3 | Foundation / conditioned | scGPT | 5 | 0.165 | 0.494 | -0.328 | range [-0.622, -0.130] |
| T3 | Graph / conditioned | AttentionPert | 5 | 0.227 | 0.494 | -0.267 | range [-0.327, -0.125] |
| T3 | Latent / conditioned | Biolord | 5 | 0.394 | 0.494 | -0.100 | range [-0.138, +0.042] |
| T3 | OT / diagnostic: CINEMA-OT | CINEMA-OT | 5 | 0.458 | 0.494 | -0.036 | range [-0.061, -0.003] |
| T4 | Deterministic shift / diagnostic: linear-shift-KOemb | linear-shift-KOemb | 2 | 0.618 | 0.661 | -0.044 | range [-0.055, -0.032] |
| T4 | Flow / conditioned | PerturbNet | 2 | 0.598 | 0.661 | -0.063 | range [-0.065, -0.061] |
| T4 | Foundation / conditioned | scGPT | 2 | 0.557 | 0.661 | -0.104 | range [-0.109, -0.100] |
| T4 | Graph / conditioned | AttentionPert | 2 | 0.508 | 0.661 | -0.153 | range [-0.158, -0.149] |
| T4 | Latent / conditioned | Biolord | 2 | 0.613 | 0.661 | -0.048 | range [-0.051, -0.045] |
| T5u | Chemistry / conditioned | PRnet | 28 | 0.134 | 0.172 | -0.038 | 95% CI [-0.052, -0.023] |
| T5u | Chemistry / diagnostic: FP-ridge | FP-ridge | 28 | 0.164 | 0.172 | -0.008 | 95% CI [-0.031, +0.019] |
| T5u | Flow / conditioned | CellFlow | 28 | 0.166 | 0.172 | -0.006 | 95% CI [-0.019, +0.008] |
| T5u | Latent / conditioned | Biolord | 28 | 0.127 | 0.172 | -0.046 | 95% CI [-0.077, -0.005] |
| T5u | OT / diagnostic: CINEMA-OT | CINEMA-OT | 28 | 0.172 | 0.172 | -0.000 | 95% CI [-0.002, +0.002] |
| T5c | Chemistry / conditioned | PRnet | 4 | 0.094 | 0.269 | -0.175 | range [-0.209, -0.143] |
| T5c | Chemistry / diagnostic: FP-ridge | FP-ridge | 4 | 0.387 | 0.269 | +0.118 | range [+0.105, +0.130] |
| T5c | Flow / conditioned | CellFlow | 4 | 0.104 | 0.269 | -0.166 | range [-0.199, -0.116] |
| T5c | Foundation / conditioned | scFoundation | 4 | 0.055 | 0.269 | -0.214 | range [-0.257, -0.126] |
| T5c | Latent / conditioned | Biolord | 4 | 0.079 | 0.269 | -0.191 | range [-0.252, -0.122] |
| T5c | OT / conditioned | CellOT | 4 | 0.101 | 0.269 | -0.168 | range [-0.240, -0.095] |
| T5c | OT / diagnostic: CINEMA-OT | CINEMA-OT | 4 | 0.253 | 0.269 | -0.016 | range [-0.035, +0.005] |
