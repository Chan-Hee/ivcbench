#!/usr/bin/env python
"""Per-census-cell coverage: how many held units the model actually predicted.

Why this exists. A held unit a model cannot express keeps the control mean, and a control mean is
shape-compatible with a real prediction -- so downstream it is scored as a confident "no response"
unless something records that it was never predicted. The audit (F-11, F-14) found four places that
substitute the control silently. This reads the deposited bundles and states, per cell, which held
units are model predictions and which are substitutions, with the reason each one is checkable
against the bundle itself.

Reason classes, decided from the bundle and nothing else:
  panel      the held target is not one of the modelled genes, so there is no feature to condition
             on. Shared across every model that reads the same panel.
  model      the target IS in the panel but this model produced no prediction for it -- its own
             vocabulary, embedding table or input contract, not the benchmark's.
  unit       a non-gene held unit (a compound) the model declined.

The score convention is NOT changed here: a substituted unit stays in the macro-average with zero
response, which costs the model (measured: +0.0003 to +0.0188 if they were dropped instead). This
table is what lets a reader see that cost rather than infer it.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble_cross_cluster import eligible_bundle  # noqa: E402  the census's own bundle policy

# Clusters whose held unit IS a gene, so "is it one of the modelled genes?" is the right question.
# C1/C2/C5 hold out a lineage, a donor or a compound; those can never be looked up in the panel.
GENE_HELD = {"C3_LO_gene", "C4_Axis2"}


def label_of(stratum: str) -> str:
    """'perturbation=IRF4' -> 'IRF4'; compound|lineage keys keep the compound."""
    s = str(stratum).split("|")[0]
    return s.split("=", 1)[-1] if "=" in s else s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=str(ROOT / "predictions"))
    ap.add_argument("--csv", default=str(ROOT / "results/_paper/cell_coverage.csv"))
    a = ap.parse_args()

    # Read the bundle store through the CENSUS's own eligibility policy, not the raw glob. The
    # store holds withdrawn runs, the quarantined STATE T2 diagnostics, single-unit validation
    # deposits and superseded re-runs alongside the reported ones; counting those inflates
    # held_units and mixes a fixed bundle's units with the broken bundle it replaced.
    per_cell: dict[tuple[str, str, str], dict] = {}
    paths = [p for p in glob.glob(os.path.join(a.dump, "**", "*.npz"), recursive=True)
             if eligible_bundle(p)]
    skipped = len(glob.glob(os.path.join(a.dump, "**", "*.npz"), recursive=True)) - len(paths)
    # Oldest first, so when two eligible bundles cover the same held unit of the same cell the
    # LATER deposit is the one that decides whether that unit was predicted.
    for p in sorted(paths, key=lambda f: (os.path.getmtime(f), f)):
        try:
            d = np.load(p, allow_pickle=True)
        except Exception:
            continue
        if not {"pred_means", "control_mean"} <= set(d.files):
            continue
        P = np.atleast_2d(np.asarray(d["pred_means"], np.float64))
        C = np.asarray(d["control_mean"], np.float64).ravel()
        if P.shape[1] != C.shape[0]:
            continue
        genes = {str(x) for x in np.atleast_1d(d["genes"])} if "genes" in d.files else set()
        strata = (
            [str(x) for x in np.atleast_1d(d["strata"])]
            if "strata" in d.files
            else [str(i) for i in range(len(P))]
        )
        cl = str(d["cluster"]) if "cluster" in d.files else "?"
        # One cell can span two readouts with different feature panels: C4 scores RNA (2,000 genes)
        # and surface protein (20 features) separately, and a held KO can be in one panel and not
        # the other. Collapsing them would report "outside panel" without saying outside which.
        ds = str(d["dataset"]) if "dataset" in d.files else ""
        key = (cl, str(d["model"]) if "model" in d.files else "?", ds)
        # Keyed by held unit, not accumulated as a count: one cell is often deposited by several
        # bundles (eight STATE donor shards, four CellOT lineages), and a re-run deposits the same
        # units again. A counter double-counts both; a per-unit verdict does not.
        rec = per_cell.setdefault(key, {"seen": {}, "panel": set(), "model": set(), "unit": set()})
        # A bundle may also carry an explicit decline mask written by the runner.
        declared = set()
        if "declined_strata" in d.files:
            declared = {str(x) for x in np.atleast_1d(d["declined_strata"])}
        for i in range(len(P)):
            sub = bool(np.array_equal(P[i], C) or strata[i] in declared)
            rec["seen"][strata[i]] = sub      # later deposit wins
            if not sub:
                continue
            lab = label_of(strata[i])
            if cl not in GENE_HELD:
                rec["unit"].add(lab)
            elif not genes:
                rec["unit"].add(lab)      # no panel recorded -> cannot attribute; say so rather than guess
            elif lab in genes:
                rec["model"].add(lab)
            else:
                rec["panel"].add(lab)

    rows = []
    for (cl, model, ds), r in sorted(per_cell.items()):
        # A unit counts once. Its verdict comes from the latest eligible bundle that covered it, so
        # a re-run that fixed a substitution is not still reported as substituted, and the reasons
        # below are pruned to the units whose final verdict is still a substitution.
        n_units = len(r["seen"])
        still = {label_of(k) for k, v in r["seen"].items() if v}
        n_sub = sum(1 for v in r["seen"].values() if v)
        for cls in ("panel", "model", "unit"):
            r[cls] &= still
        rows.append(dict(
            cluster=cl, model=model, readout=ds or "(default)",
            held_units=n_units,
            predicted=n_units - n_sub,
            substituted=n_sub,
            outside_panel=";".join(sorted(r["panel"])),
            in_panel_not_predicted=";".join(sorted(r["model"])),
            other_declined=";".join(sorted(r["unit"])),
        ))
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"coverage -> {a.csv}   ({len(rows)} cells; "
          f"{skipped} bundle(s) skipped as not census-eligible)")
    bad = [r for r in rows if r["substituted"]]
    print(f"cells with at least one substituted held unit: {len(bad)}")
    for r in bad:
        why = " / ".join(
            x for x in (
                f"outside panel: {r['outside_panel']}" if r["outside_panel"] else "",
                f"in panel, not predicted: {r['in_panel_not_predicted']}" if r["in_panel_not_predicted"] else "",
                f"other: {r['other_declined']}" if r["other_declined"] else "",
            ) if x
        )
        print(f"  {r['cluster']:14s} {r['model']:20s} {r['readout']:22s} "
              f"{r['predicted']}/{r['held_units']}  {why}")


if __name__ == "__main__":
    main()
