"""Axis 4 — Generalization-robustness: the gap between an easy and a leak-proof split. Lower=better.

Examples: random↔LODO (C2), within↔across functional class (C1), in-vitro↔in-vivo (C4),
Tanimoto-near↔far (C5). gap = score(easy) − score(hard) on the same baseline/metric.
"""
from __future__ import annotations


def split_gap(score_easy: float, score_hard: float) -> float:
    return float(score_easy - score_hard)
