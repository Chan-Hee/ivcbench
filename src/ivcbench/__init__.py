"""ivcbench — Section 3 immune-perturbation benchmark framework.

Layers (see PLAN.md §4):
  data/      unified obs schema, dataset loaders, side-info, synthetic fixtures
  splits/    declarative leak-proof split specs + the leak auditor (the v5 guardrail)
  baselines/ BaselineAdapter ABC, simple baselines, 13x9 applicability registry
  metrics/   4-axis metrics (response / distribution / immune-program / robustness) + stats
  runner/    job = (cluster, split, baseline, seed); gating; 2-GPU scheduler
  report/    Supp tables + per-cluster figures

Core logic (splits + metrics) is dependency-light (numpy/pandas/scipy/sklearn) so the whole
pipeline is validated by a GPU-free synthetic smoke test before any real data or model touches it.
"""

__version__ = "0.1.0"
