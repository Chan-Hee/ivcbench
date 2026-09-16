#!/usr/bin/env python3
"""Recompute the PRE-SPECIFIED headline multiplicity family from the current census.

This family is fixed in its CONTRASTS, not in its numbers: H1 to H5 were named in advance, and the
three confirmatory tests were named with them. What each one evaluates has to be recomputed
whenever the census changes, and until now nothing did that. The deposited file was written by hand
on 2026-09-09 and carried H2 at a raw p of 0.0625, the smallest a signed-rank test over five
dataset arms can return, because the best conditioned entry was then below the floor in every arm.
It is not: biolord clears the cell-mean floor on Schmidt by +0.044, so the statistic is 0.125.

Table S11 is printed from this file (revision_claude/02_build/tools/fix_table_s11.py) and the
machine-readable deposit under supplementary_tables/ is a copy of it, so a stale value here reaches
the paper twice.

The two families are adjusted separately, as they were pre-specified: the five floor tests
(H3 contributes no p, so m = 4) and the three confirmatory tests (m = 3).

    python scripts/headline_family.py            # report only
    python scripts/headline_family.py --apply    # rewrite the deposit
"""

from __future__ import annotations

import collections
import csv
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results" / "_paper"
OUT = PAPER / "historical_headline_multiplicity_adjusted.csv"
DEPOSIT = PAPER / "supplementary_tables" / "Supplementary_Table_S11.csv"

DIAGNOSTIC = {"FP-ridge", "linear-shift-KOemb", "CINEMA-OT"}
FLOOR_MEMBER = {"cell-mean", "linear-PCA", "ctrl-pred", "donor-shift"}


def unit_scores():
    by = collections.defaultdict(dict)
    for r in csv.DictReader(open(PAPER / "census_unit_scores.csv")):
        by[(r["task_key"], r["model"])][r["unit"]] = float(r["pearson_delta"])
    return by


def binding(by, task):
    """The floor member selected once per task on its macro-average, as the census selects it."""
    cm, lp = by[(task, "cell-mean")], by[(task, "linear-PCA")]
    return (cm, "cell-mean") if statistics.mean(cm.values()) >= statistics.mean(lp.values()) \
        else (lp, "linear-PCA")


def gaps(by, task, model):
    fl, _ = binding(by, task)
    return [by[(task, model)][u] - fl[u] for u in sorted(fl) if u in by[(task, model)]]


def adjust(pvalues):
    """BH and Holm with monotonicity, over one family."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    bh = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    holm = np.maximum.accumulate(ranked * np.arange(m, 0, -1))
    out_bh, out_holm = np.empty(m), np.empty(m)
    out_bh[order], out_holm[order] = np.minimum(bh, 1.0), np.minimum(holm, 1.0)
    return out_bh, out_holm


def main() -> None:
    by = unit_scores()
    rows = []

    # ---- H1: cytokine cell-context, scGen against the binding floor, over lineages -------------
    d = gaps(by, "T1", "scGen")
    rows.append(dict(
        contrast="H1_C1_cellcontext_scGen_vs_floor", p=wilcoxon(d).pvalue, family="headline_floor",
        source=f"recomputed: Wilcoxon over {len(d)} lineages from census_unit_scores.csv"))

    # ---- H2: unseen gene, the best conditioned entry against the binding floor, over arms ------
    cond = [m for (t, m) in by if t == "T3" and m not in DIAGNOSTIC | FLOOR_MEMBER]
    best = max(cond, key=lambda m: statistics.mean(by[("T3", m)].values()))
    d = gaps(by, "T3", best)
    up = sum(1 for x in d if x > 0)
    rows.append(dict(
        contrast="H2_C3_unseenPerturbation_bestcond_vs_floor", p=wilcoxon(d).pvalue,
        family="headline_floor",
        source=(f"recomputed: Wilcoxon over {len(d)} datasets (pre-specified unit) from "
                f"census_unit_scores.csv; best conditioned entry {best}, above the floor in "
                f"{up} of {len(d)} arms")))

    # ---- H3: modality split, scGen. Its execution is excluded from the census ------------------
    rows.append(dict(
        contrast="H3_C4_modality_scGen_vs_floor", p=None, family="headline_floor",
        source=("[MISSING] the scGen modality execution substitutes an author-written latent shift "
                "for the model's published conditioning and is not a census cell (Table S15b); "
                "two holdout fractions supply no biological-unit replicate in any case")))

    # ---- H4: OP3 cell-context, FP-ridge against the binding floor, over the six fine lineages --
    fine = list(csv.DictReader(open(PAPER / "Supplementary_Table_S5_op3_fine_lineage.csv")))
    d = [float(r["gap_FPridge_minus_binding_floor"]) for r in fine]
    rows.append(dict(
        contrast="H4_C5_cellcontext_FPridge_vs_floor", p=wilcoxon(d).pvalue,
        family="headline_floor",
        source=(f"recomputed: Wilcoxon over {len(d)} fine lineages from "
                "Supplementary_Table_S5_op3_fine_lineage.csv")))

    # ---- H5: unseen compound, Tanimoto distance-slope equivalence for FP-ridge -----------------
    tan = {r["model"]: r for r in csv.DictReader(open(PAPER / "op3_tanimoto_sensitivity.csv"))}
    rows.append(dict(
        contrast="H5_C5_unseenCompound_TanimotoTOST_equiv0", p=float(tan["FP-ridge"]["tost_p"]),
        family="headline_floor",
        source="recomputed: TOST equivalence (+/-0.05/SD) from op3_tanimoto_sensitivity.csv"))

    # ---- the three confirmatory tests, adjusted as their own family ---------------------------
    sp = next(iter(csv.DictReader(open(PAPER / "scpram_vs_cellot_donor_paired.csv"))))
    rows.append(dict(
        contrast="C2_donor_scPRAM_vs_CellOT_paired", p=float(sp["wilcoxon_p"]),
        family="confirmatory",
        source="deposited: scpram_vs_cellot_donor_paired.csv (wilcoxon_p)"))

    mult = {r["contrast"]: r for r in
            csv.DictReader(open(PAPER / "headline_multiplicity_adjusted.csv"))}
    rows.append(dict(
        contrast="C2_donor_CellOT_vs_floor",
        p=float(mult["C2_donor_CellOT_vs_floor"]["raw_p"]), family="confirmatory",
        source="headline_multiplicity_adjusted.csv (paired two-sided Wilcoxon over 106 donors)"))

    null = list(csv.DictReader(open(ROOT / "results" / "C5" / "ifn_shuffle_null.csv")))
    # the column is the one-sided permutation p, floored at the resolution 2,000 permutations can
    # resolve (1/(n+1) = 5e-4); every lineage sits at that floor
    per = [float(r["perm_p_onesided"]) for r in null if r.get("perm_p_onesided")]
    stat = -2.0 * float(np.sum(np.log(per)))
    rows.append(dict(
        contrast="C5_FPridge_compound_matching_null_Fisher",
        p=float(chi2.sf(stat, 2 * len(per))), family="confirmatory",
        source=f"recomputed: Fisher-combine {len(per)} per-lineage perm_p from C5/ifn_shuffle_null.csv"))

    # ---- adjust within family -----------------------------------------------------------------
    for fam in ("headline_floor", "confirmatory"):
        members = [r for r in rows if r["family"] == fam and r["p"] is not None
                   and np.isfinite(r["p"])]
        bh, holm = adjust([r["p"] for r in members])
        for r, b, h in zip(members, bh, holm):
            r["bh"], r["holm"] = float(b), float(h)
            r["survives"] = bool(b < 0.05 and h < 0.05)

    def fmt(x):
        return "[MISSING]" if x is None else f"{x:.6g}"

    print(f"{'contrast':46s} {'raw_p':>12s} {'BH_p':>12s} {'Holm_p':>12s}  survives")
    out = []
    for r in rows:
        print(f"{r['contrast']:46s} {fmt(r.get('p')):>12s} {fmt(r.get('bh')):>12s} "
              f"{fmt(r.get('holm')):>12s}  {r.get('survives', '[MISSING]')}")
        out.append({"contrast": r["contrast"], "raw_p": fmt(r.get("p")), "BH_p": fmt(r.get("bh")),
                    "Holm_p": fmt(r.get("holm")),
                    "survives_FDR05": str(r["survives"]) if "survives" in r else "[MISSING]",
                    "family": r["family"], "source": r["source"]})

    if "--apply" not in sys.argv:
        print("\n(report only; pass --apply to write)")
        return
    fields = ["contrast", "raw_p", "BH_p", "Holm_p", "survives_FDR05", "family", "source"]
    for path in (OUT, DEPOSIT):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader(); w.writerows(out)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
