"""Descriptive score difference between two prediction settings.

Examples: random↔LODO (C2), within↔across functional class (C1), in-vitro↔in-vivo (C4),
Tanimoto-near↔far (C5). gap = score(easy) − score(hard) on the same baseline/metric.
This arithmetic does not establish comparable target quality or independent
preprocessing across the settings.
"""

from __future__ import annotations


def split_gap(score_easy: float, score_hard: float) -> float:
    return float(score_easy - score_hard)
