#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail the build when a release document states a census fact the census contradicts.

The deposit gates check the deposit. Nothing checked the prose around it, and every release
document had drifted: README, ANALYSIS_SCOPE, EXECUTION_AUDIT, REPRODUCE, RELEASE_FOR_REVISION,
predictions/COVERAGE.md and .zenodo.json all still described a 56- or 47-cell census, REPRODUCE
contradicted itself in two places, RELEASE_FOR_REVISION's own pre-tag assertion failed, and
.zenodo.json is the text that lands on the archive record the paper's DOI resolves to.

Each check names a phrase and the number the census says belongs in it, so the failure tells you
what to edit rather than that something is wrong somewhere.

    python scripts/check_release_docs.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results/_paper"

WORDS = {4: "four", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven"}


def facts() -> dict:
    canon = json.loads((PAPER / "CANONICAL_NUMBERS.json").read_text())
    head = list(csv.DictReader((PAPER / "cross_cluster_headline.csv").open()))
    man = list(csv.DictReader((PAPER / "census_bundle_manifest.csv").open()))
    fam = [r["family"] for r in csv.DictReader((PAPER / "headline_multiplicity_adjusted.csv").open())]
    roles = [r["role"] for r in man]
    per = canon["per_task"]
    return {
        "cells": canon["census_cells"],
        "native": canon["status"]["native"],
        "adapted": canon["status"]["adapted"],
        "diagnostic": canon["status"]["diagnostic"],
        "bundles": len(man),
        "model_bundles": roles.count("method"),
        "floor_bundles": roles.count("floor"),
        "unit_rows": sum(1 for _ in (PAPER / "census_unit_scores.csv").open()) - 1,
        "panel_rows": sum(1 for _ in (PAPER / "panel_precision_attenuation.csv").open()) - 1,
        "family": fam.count("current_census_n_ge_8"),
        "descriptive": fam.count("descriptive_low_n"),
        "clears": sum(1 for r in head if r["beats_both_floor_members"] in ("True", "true", "1")),
        "per_task": " / ".join(str(per[t]) for t in ("T1", "T2", "T3", "T4", "T5c", "T5u")),
        "version": re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(),
                             re.M).group(1),
    }


def _n(value) -> set[str]:
    """How a number may legitimately be written: digits, with thousands separator, or as a word."""
    out = {str(value), f"{value:,}"}
    if value in WORDS:
        out.add(WORDS[value])
        out.add(WORDS[value].capitalize())   # sentence-initial: "Eight point estimates exceed..."
    return out


# (file, regex with ONE capture group, fact key). The capture is compared to the fact.
CHECKS = [
    ("README.md", r"\*\*(\d+) model-by-task evaluations", "cells"),
    ("README.md", r"evaluations: (\d+) native", "native"),
    ("README.md", r"native, (\w+) adapted", "adapted"),
    ("README.md", r"adapted and (\w+) diagnostic", "diagnostic"),
    ("README.md", r"have ([\d /]+?) entries for T1", "per_task"),
    ("README.md", r"\*\*([\d,]+) selected mean-profile bundles\*\*", "bundles"),
    ("README.md", r"bundles\*\* \(([\d,]+) model", "model_bundles"),
    ("README.md", r"model and ([\d,]+) simple-reference", "floor_bundles"),
    ("README.md", r"checks the (\d+)-entry panel", "cells"),
    ("README.md", r"panel, ([\d,]+) analysis-unit rows", "unit_rows"),
    ("README.md", r"the (\d+)-comparison BH/Holm family", "family"),
    ("README.md", r"revised (\d+)-entry panel", "cells"),
    ("ANALYSIS_SCOPE.md", r"panel contains (\d+) evaluations", "cells"),
    ("ANALYSIS_SCOPE.md", r"evaluations: (\d+) native", "native"),
    ("ANALYSIS_SCOPE.md", r"All (\d+) entries with at least eight units", "family"),
    ("ANALYSIS_SCOPE.md", r"The other (\d+) entries receive descriptive", "descriptive"),
    ("ANALYSIS_SCOPE.md", r"MDE values for the (\d+) eligible", "family"),
    ("EXECUTION_AUDIT.md", r"the panel holds (\d+) entries", "cells"),
    ("REPRODUCE.md", r"writes ([\d,]+) rows to", "bundles"),
    ("REPRODUCE.md", r"includes ([\d,]+) selected model inputs", "model_bundles"),
    ("REPRODUCE.md", r"reconstructs the selected (\d+) evaluations", "cells"),
    ("REPRODUCE.md", r"hashes, ([\d,]+) analysis-unit rows", "unit_rows"),
    ("REPRODUCE.md", r"The panel has (\d+) native", "native"),
    ("REPRODUCE.md", r"inference family has (\d+) entries", "family"),
    ("REPRODUCE.md", r"the remaining (\d+) receive descriptive", "descriptive"),
    ("REPRODUCE.md", r"statements for all (\d+) contrasts", "panel_rows"),
    ("RELEASE_FOR_REVISION.md", r"revision reports \*\*(\d+)\*\* evaluations", "cells"),
    ("RELEASE_FOR_REVISION.md", r"assert len\(h\) == (\d+)", "cells"),
    ("RELEASE_FOR_REVISION.md", r"([\d,]+) bundles covering all", "bundles"),
    ("RELEASE_FOR_REVISION.md", r"bundles covering all (\d+) cells", "cells"),
    ("RELEASE_FOR_REVISION.md", r"git tag -a v(\S+) -m", "version"),
    ("predictions/COVERAGE.md", r"\*\*([\d,]+) mean-profile bundles", "bundles"),
    ("predictions/COVERAGE.md", r"yielding (\d+) model-by-task", "cells"),
    ("predictions/COVERAGE.md", r"and ([\d,]+) model analysis-unit rows", "unit_rows"),
    ("predictions/COVERAGE.md", r"settings contain ([\d /]+?) entries", "per_task"),
    ("predictions/COVERAGE.md", r"Status counts are (\d+) native", "native"),
    ("results/_paper/DEPOSIT_NOTES.md", r"one row per census evaluation \((\d+)\)", "panel_rows"),
    (".zenodo.json", r'"version": "([^"]+)"', "version"),
    (".zenodo.json", r"contains (\d+) model-by-task evaluations", "cells"),
    (".zenodo.json", r"evaluations: (\d+) native", "native"),
    (".zenodo.json", r"([\d,]+) deposited mean-profile bundles", "bundles"),
    # "clears" was derived in facts() and consumed by nothing, so the one README sentence that
    # states it was unchecked; and EXECUTION_AUDIT's status triple was unpinned, so 999/777/555
    # passed. Both are pinned now.
    ("README.md", r"(\w+) point estimates exceed the task-fixed", "clears"),
    ("EXECUTION_AUDIT.md", r"the panel holds (\d+) entries", "cells"),
    ("EXECUTION_AUDIT.md", r"entries \((\d+) native", "native"),
    ("EXECUTION_AUDIT.md", r"native, (\d+) adapted", "adapted"),
    ("EXECUTION_AUDIT.md", r"adapted, (\d+) diagnostic", "diagnostic"),
]


def main() -> int:
    f = facts()
    bad = 0
    for name, pattern, key in CHECKS:
        path = ROOT / name
        if not path.is_file():
            print(f"  ✗ {name}: missing")
            bad += 1
            continue
        m = re.search(pattern, path.read_text(encoding="utf-8"))
        if m is None:
            print(f"  ✗ {name}: no text matches {pattern!r}; the sentence it checks was reworded")
            bad += 1
            continue
        want = f[key]
        got = m.group(1)
        ok = got in _n(want) if isinstance(want, int) else got == want
        if not ok:
            print(f"  ✗ {name}: {key} reads {got!r}, the census says {want!r}")
            bad += 1
    print(f"  릴리스 문서 {len(CHECKS)}개 검사 · 불일치 {bad}건")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
