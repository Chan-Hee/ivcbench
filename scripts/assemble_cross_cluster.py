#!/usr/bin/env python3
"""Assemble the CROSS-CLUSTER HEADLINE + WITHIN-FAMILY CONSISTENCY tables.

SINGLE SOURCE OF TRUTH = the deposited prediction bundles. This script re-scores every
bundle under predictions/ (the same GPU-free path a reviewer runs via `make reproduce-eval`)
and aggregates those scores into the current headline. Nothing here reads a results_raw.csv
or any hand-maintained summary: the census is a pure function of the bundles, so it cannot
drift away from what a reviewer can reproduce. `scripts/check_consistency.py` (run by
`make test`) re-derives this file and fails the build if the committed copy disagrees.

The primary margin is Pearson-delta minus the stronger task-average cell-mean /
linear-PCA reference. Analysis units are lineage, donor, dataset-arm, overlapping
KO holdout fraction or compound, as defined in ANALYSIS_SCOPE.md. The legacy
floor_mean fields are the arithmetic mean of the two references, not the binding
floor used for claims or inference. Explicit binding_floor, floor_score and margin
columns are authoritative; census_uncertainty.csv retains full precision.

C2 carries two split-construction schemes among the deposited bundles (bespoke
`C2_soskic_LODO_*` for CellOT/CPA/STATE/scPRAM, framework `C2_lodo_*` for the floors / scGen);
the model->scheme assignment is disjoint, so unioning the C2 clusters recovers each model from
its own bundle. The donor index is normalised across the two schemes for the within-family rho.
"""
from __future__ import annotations
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

from ivcbench.eval.bundle import score_bundle

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])
OUT = os.path.join(ROOT, "results", "_paper")

FAMILY = {
    "cell-mean": "Simple",
    "linear-PCA": "Simple",
    "ctrl-pred": "Simple",
    "donor-shift": "Simple",
    "scGen": "Latent",
    "CPA": "Latent",
    "Biolord": "Latent",
    "chemCPA": "Chemistry",
    "FP-ridge": "Chemistry",
    "scGPT": "Foundation",
    "scFoundation": "Foundation",
    "GEARS": "Graph",
    "AttentionPert": "Graph",
    "STATE": "Hybrid",
    "PertAdapt": "Hybrid",
    "CellOT": "OT",
    "scPRAM": "OT",
    "CINEMA-OT": "OT",
    "CellFlow": "Flow",
    "PerturbNet": "Flow",
    "PRnet": "Chemistry",
    "linear-shift-KOemb": (
        "Deterministic shift"
    ),  # category matches main Table 2 / Supplementary Table S2
}

# ---------------- re-score every deposited bundle (GPU-free), excluding the format-demo toys --------
# Bundles that exist in the deposit but are NOT census members. Each is excluded for a stated reason,
# and each exclusion was found by an audit that caught the census averaging over things that are not
# the same measurement.
#   _withdrawn/  — SCREEN, withdrawn in full (its VAE diverges to NaN on 45 of 53 donors on T2)
#   op3_fcond    — the conditioned-head analysis on the compound splits. It is a SEPARATE, disclosed
#                  adaptation reported in its own section; averaging it into the cell-context cell
#                  mixed two different execution interfaces and produced a number
#                  (0.1857 = mean of 0.0247 and 0.3467) that describes neither.
#   frangieh_protein — the surface-protein readout. The T4 census cell is the RNA readout; folding the
#                  protein bundles in moved CellFlow from 0.4924 to 0.3067.
#   the three untagged C3 bundles — written before run_obligation.py passed a `dataset` key on the T3
#                  branch, so all five datasets collided on one filename. Their per-dataset
#                  replacements are deposited; keeping the stale file made the macro-average count six
#                  units and double-count one dataset.
_NON_CENSUS_BUNDLE = (
    os.sep + "example" + os.sep,
    os.sep + "_withdrawn" + os.sep,
    "__op3_fcond.npz",
    "__frangieh_protein.npz",
    # These historical CPA/scGen drug runs use an author-written fingerprint-to-latent
    # regression. They are preserved on disk, but excluded under the author's native-only
    # rule for task-specific methods. The unseen-compound CPA entry uses native chemCPA.
    "__CPA__C5_loct_",
    "__scGen__C5_loct_",
    "__CPA__C5_global_compound_holdout.npz",
    # CPAC1 adds a training-population latent_after mean difference instead of
    # using CPA's learned perturbation embedding. This is an author-written
    # inference adaptation on both context tasks and is excluded by the same rule.
    "__CPA__C1_loct_",
    "__CPA__C2_soskic_LODO_",
    "__CPA__C2_lodo_",
)
_STALE_UNTAGGED_C3 = {
    f"C3_LO_gene__{m}__C3_true_lo_gene_10.npz"
    for m in ("Biolord", "CellFlow", "PerturbNet")
}


EXECUTION_MODEL = {("C5", "unseen-compound", "CPA"): "chemCPA"}


def canonicalise_bundle_models(df):
    """Use the native chemCPA execution for the CPA/chemCPA method-group row.

    Keep the original execution name in a separate column. Refuse to average the
    native execution with the historical author-written CPA fingerprint adaptation.
    This function is also used by the figure builder.
    """
    df = df.copy()
    native = (df["model"] == "chemCPA") & (df["split"] == "C5_global_compound_holdout")
    old = (df["model"] == "CPA") & (df["split"] == "C5_global_compound_holdout")
    if old.any():
        raise ValueError(
            "The non-native CPA compound bundle must be excluded before"
            " canonicalisation"
        )
    df["execution_model"] = df["model"]
    df.loc[native, "model"] = "CPA"
    return df


def eligible_bundle(path):
    """Shared bundle-store policy, including superseded metric metadata."""
    from pathlib import Path

    path = Path(path)
    if (
        any(token in str(path) for token in _NON_CENSUS_BUNDLE)
        or path.name in _STALE_UNTAGGED_C3
    ):
        return False
    if path.parent == Path(ROOT) / "predictions" and path.name.startswith(
        ("C2_LODO__Biolord__C2_lodo_", "C2_LODO__CellFlow__C2_lodo_")
    ):
        replacement = Path(ROOT) / "predictions/C2_metric_aligned" / path.name
        if not replacement.exists():
            raise ValueError(f"Missing T2 metric-metadata correction: {replacement}")
        return False
    return True


def score_all():
    files = sorted(
        f
        for f in glob.glob(
            os.path.join(ROOT, "predictions", "**", "*.npz"), recursive=True
        )
        if eligible_bundle(f)
    )
    df = canonicalise_bundle_models(pd.DataFrame(score_bundle(f) for f in files))
    df["bundle_path"] = [os.path.relpath(f, ROOT) for f in files]
    df["dataset"] = df["dataset"].fillna("")
    return df


# ---------------- task-cell definitions (which bundles -> which census cell) ------------------------
# each cell: the bundle clusters it draws from, the split it is defined on, how to read the biological
# unit (for the macro-average and the within-family rho), and the PUBLISHED roster of conditioned
# models reported for that cell. The roster fixes the final 47-entry panel;
# values are sourced from the bundles. Stored but out-of-panel executions do not
# become model evidence merely because their numeric bundle can be read.
CELLS = [
    dict(
        cl="C1",
        task_id="T1",
        task="cytokine/Kang",
        split="cell-context (LOCT)",
        unit="lineage",
        n_unit=8,
        clusters=["C1_LOCT"],
        match=lambda s: s.startswith("C1_loct"),
        unit_of=lambda r: r["split"].replace("C1_loct_", ""),
        roster=[
            "Biolord",
            "CellFlow",
            "CellOT",
            "STATE",
            "scFoundation",
            "scGPT",
            "scGen",
            "scPRAM",
        ],
    ),
    dict(
        cl="C2",
        task_id="T2",
        task="donor/Soskic",
        split="donor (LODO)",
        unit="donor",
        n_unit=106,
        clusters=["C2", "C2_LODO"],
        match=lambda s: s.startswith(("C2_soskic_LODO", "C2_lodo")),
        unit_of=lambda r: r["split"]
        .replace("C2_soskic_LODO_", "")
        .replace("C2_lodo_", ""),
        roster=[
            "Biolord",
            "CellFlow",
            "CellOT",
            "PertAdapt",
            "STATE",
            "scFoundation",
            "scGPT",
            "scGen",
            "scPRAM",
        ],
    ),
    dict(
        cl="C3",
        task_id="T3",
        task="gene/CRISPR",
        split="unseen-perturbation (LO-gene 10%)",
        unit="dataset",
        n_unit=5,
        clusters=["C3_LO_gene"],
        match=lambda s: s == "C3_true_lo_gene_10",
        unit_of=lambda r: r["dataset"],
        roster=[
            "AttentionPert",
            "Biolord",
            "CINEMA-OT",
            "CellFlow",
            "GEARS",
            "PerturbNet",
            "scGPT",
        ],
    ),
    dict(
        cl="C4",
        task_id="T4",
        task="complex/Frangieh",
        split="unseen-KO (modality, RNA)",
        unit="modality-fold",
        n_unit=2,
        clusters=["C4", "C4_Axis2"],
        match=lambda s: s.startswith("C4_modality_lo_ko"),
        unit_of=lambda r: r["split"],
        roster=[
            "AttentionPert",
            "Biolord",
            "CellFlow",
            "GEARS",
            "PerturbNet",
            "linear-shift-KOemb",
            "scGPT",
        ],
    ),
    dict(
        cl="C5",
        task_id="T5",
        task="small-mol/OP3",
        split="unseen-compound",
        unit="compound",
        n_unit=28,
        clusters=["C5", "C5_unseen_cpd"],
        match=lambda s: s == "C5_global_compound_holdout",
        unit_of=lambda r: r["split"],
        roster=[
            "Biolord",
            "CINEMA-OT",
            "CPA",
            "CellFlow",
            "FP-ridge",
            "PRnet",
            "PerturbNet",
        ],
    ),
    dict(
        cl="C5",
        task_id="T5",
        task="small-mol/OP3",
        split="cell-context (LOCT)",
        unit="lineage",
        n_unit=4,
        # the OP3 cell-axis bundles carry the legacy scheme tag "C1_LOCT" in their `cluster` field;
        # the SPLIT name is what identifies the cell, so both tags are accepted and the split match
        # does the work. Keeping the cluster tag out of the decision is what stops the two
        # cell-context tasks from ever being conflated again (their registry columns are distinct).
        clusters=["C1_LOCT", "C5_LOCT"],
        match=lambda s: s.startswith("C5_loct"),
        unit_of=lambda r: r["split"].replace("C5_loct_", ""),
        roster=[
            "Biolord",
            "CINEMA-OT",
            "CellFlow",
            "CellOT",
            "FP-ridge",
            "PRnet",
            "scFoundation",
            "scGPT",
            "scPRAM",
        ],
    ),
]


def cell_long(df, cell):
    """Long table of per-(model, unit) pearson_delta for one census cell."""
    d = df[df["cluster"].isin(cell["clusters"]) & df["split"].map(cell["match"])].copy()
    d["unit"] = d.apply(cell["unit_of"], axis=1)
    # collapse any cluster/dataset duplication onto (model, unit) -> the model's own bundle value
    return d.groupby(["model", "unit"], as_index=False)["pearson_delta"].mean()


# ---------------------------------------------------------------------------
# PER-CELL STATUS (Table 2 / Suppl. Table S4). Derived here so it is a pure
# function of the code and cannot be lost on a census rebuild.
# Status is resolved on one operational criterion: whether the census execution
# needed a model-specific task interface authored for this study.  Ordinary
# benchmark I/O plumbing is not an adaptation.
#   native     - no author-written model-specific task interface was needed
#   adapted    - an author-written model-specific task interface was needed;
#                a shortfall bounds that interface as well as the architecture
#   diagnostic - comparator run to characterise the task, not a competing method
# Resolved from ivcbench.baselines.registry.APPLICABILITY, with explicit
# overrides for the cells whose status the registry cannot express.
DIAGNOSTIC_MODELS = {"CINEMA-OT", "FP-ridge", "linear-shift-KOemb"}
APPLICABILITY_KEY = {
    ("C1", "cell-context (LOCT)"): "C1_LOCT",
    ("C2", "donor (LODO)"): "C2_LODO",
    ("C3", "unseen-perturbation (LO-gene 10%)"): "C3_LO_gene",
    ("C4", "unseen-KO (modality, RNA)"): "C4_Axis2",
    ("C5", "cell-context (LOCT)"): "C5_LOCT",
    ("C5", "unseen-compound"): "C5_unseen_cpd",
}
STATUS_OVERRIDE = {
    ("C1", "cell-context (LOCT)", "scGPT"): "adapted",
    ("C1", "cell-context (LOCT)", "scFoundation"): "adapted",
    ("C2", "donor (LODO)", "scGPT"): "adapted",
    ("C2", "donor (LODO)", "scFoundation"): "adapted",
    ("C2", "donor (LODO)", "PertAdapt"): "adapted",
    ("C5", "cell-context (LOCT)", "scGPT"): "adapted",
    ("C5", "cell-context (LOCT)", "scFoundation"): "adapted",
}

STATUS_DEFINITION = {
    "native": (
        "No model-specific task interface authored for this study was needed; the"
        " released/published model interface was used, apart from common benchmark I/O"
        " plumbing."
    ),
    "adapted": (
        "A model-specific task interface authored for this study was required to make"
        " the census execution defined on this split."
    ),
    "diagnostic": (
        "Task-specific comparator used to characterise the evaluation; it is not"
        " counted as a conditioned-method claim."
    ),
}

# Exact disclosure text for every adapted census cell.  Keeping this beside the
# canonical roster/status resolver prevents Table S2, Table S4 and Table S15
# from independently inventing (and drifting on) the meaning of "adapted".
AUTHOR_WRITTEN_INTERFACE = {
    ("C2", "donor (LODO)", "PertAdapt"): (
        "Yes — a study-written PertAdapt-inspired T2 head uses pooled frozen "
        "scFoundation cell embeddings, a learned stimulation token and lineage "
        "embedding, a binary GO co-membership mask and a shared decoder anchored "
        "to the training control mean. It reconstructs stimulated training cells "
        "from their own embeddings, then receives held-donor control embeddings "
        "at inference. This is not the published control-to-perturbed training "
        "operation or GEARS gene-condition encoder; its score does not establish "
        "native PertAdapt performance."
    ),
    ("C1", "cell-context (LOCT)", "scFoundation"): (
        "Yes — the author-written T1 interface estimates a "
        "training-lineage latent shift and trains an MLP "
        "decoder for 512 response genes; other genes retain "
        "the held-control mean."
    ),
    ("C1", "cell-context (LOCT)", "scGPT"): (
        "Yes — the author-written T1 interface replaces the "
        "gene-specific perturbation token with one global flag for "
        "the seen stimulus."
    ),
    ("C2", "donor (LODO)", "scGPT"): (
        "Yes — the author-written T1 cell-context interface reused here, "
        "with the donor as the held group: the gene-specific perturbation "
        "token is replaced by one global flag for the seen stimulation."
    ),
    ("C2", "donor (LODO)", "scFoundation"): (
        "Yes — the author-written T1 cell-context interface reused "
        "here, with the donor as the held group: a training-group "
        "latent shift with an MLP decoder for 512 response genes; "
        "other genes retain the held-control mean."
    ),
    ("C5", "cell-context (LOCT)", "scGPT"): (
        "Yes — the author-written T1 cell-context interface reused "
        "here for the seen-compound cell-axis split: the "
        "gene-specific perturbation token is replaced by one global "
        "flag for the seen exposure."
    ),
    ("C5", "cell-context (LOCT)", "scFoundation"): (
        "Yes — the author-written T1 cell-context interface "
        "reused here for the seen-compound cell-axis split: a "
        "training-lineage latent shift with an MLP decoder for "
        "512 response genes; other genes retain the "
        "held-control mean."
    ),
}

EXPECTED_CENSUS_CELLS = 47
EXPECTED_STATUS_COUNTS = {"native": 34, "adapted": 7, "diagnostic": 6}


def cell_status(cluster, split, model):
    """native / adapted / diagnostic for one census cell.

    Status is the INTERFACE-PROVENANCE axis and nothing else: a cell is adapted exactly when an
    interface authored for this study was needed to make it defined, which is recorded once in
    AUTHOR_WRITTEN_INTERFACE. It is deliberately not read from the applicability registry, whose job
    is to decide what may be RUN; conflating the two is what let a hand-edited registry relabel a
    published interface. The second axis -- whether the entry conditions on the held entity -- is
    derived separately (see revision_BIB-26-1553/07_admission/)."""
    if model in DIAGNOSTIC_MODELS:
        return "diagnostic"
    key = (cluster, split, model)
    if key in STATUS_OVERRIDE:
        return STATUS_OVERRIDE[key]
    return "adapted" if key in AUTHOR_WRITTEN_INTERFACE else "native"


def census_metadata_rows():
    """Authoritative model x census-cell metadata used by S2/S4/S15 builders."""
    rows = []
    for cell in CELLS:
        for model in cell["roster"]:
            key = (cell["cl"], cell["split"], model)
            status = cell_status(*key)
            if status == "adapted":
                if key not in AUTHOR_WRITTEN_INTERFACE:
                    raise SystemExit(
                        f"[assemble] adapted cell has no interface disclosure: {key}"
                    )
                interface = AUTHOR_WRITTEN_INTERFACE[key]
            elif status == "native":
                interface = (
                    "No — no model-specific task interface authored for this study was"
                    " used."
                )
            else:
                interface = (
                    "n.a. — diagnostic comparator, not a conditioned-method claim."
                )
            rows.append(
                {
                    "task_id": cell["task_id"],
                    "cluster": cell["cl"],
                    "task": cell["task"],
                    "split": cell["split"],
                    "unit": cell["unit"],
                    "n_unit": cell["n_unit"],
                    "model": model,
                    "status": status,
                    "execution_model": EXECUTION_MODEL.get(key, model),
                    "author_written_interface": interface,
                }
            )
    counts = {s: sum(r["status"] == s for r in rows) for s in EXPECTED_STATUS_COUNTS}
    if len(rows) != EXPECTED_CENSUS_CELLS or counts != EXPECTED_STATUS_COUNTS:
        raise SystemExit(
            f"[assemble] canonical census metadata drift: n={len(rows)},"
            f" status={counts}"
        )
    adapted = {
        (r["cluster"], r["split"], r["model"]) for r in rows if r["status"] == "adapted"
    }
    if adapted != set(AUTHOR_WRITTEN_INTERFACE):
        raise SystemExit(
            "[assemble] adapted-interface map drift:"
            f" {adapted ^ set(AUTHOR_WRITTEN_INTERFACE)}"
        )
    # Reviewer-facing sentinels for the two OT methods that previously drifted.
    sentinel = {
        (r["task_id"], r["model"]): r["status"]
        for r in rows
        if r["model"] in {"CellOT", "scPRAM"}
    }
    expected = {
        ("T1", "CellOT"): "native",
        ("T2", "CellOT"): "native",
        ("T5", "CellOT"): "native",
        ("T1", "scPRAM"): "native",
        ("T2", "scPRAM"): "native",
        ("T5", "scPRAM"): "native",
    }
    if any(sentinel.get(k) != v for k, v in expected.items()):
        raise SystemExit(f"[assemble] OT status sentinel drift: {sentinel}")
    return rows


def build(scored=None):
    """Re-score the bundles and assemble (headline, within-family) DataFrames in memory."""
    census_metadata_rows()  # fail fast if roster/status/interface metadata has drifted
    df = score_all() if scored is None else scored

    # ---------------- HEADLINE TABLE: family delta vs universal floor ----------------
    long_by_cell = {}
    rows = []
    for cell in CELLS:
        lng = cell_long(df, cell)
        long_by_cell[(cell["cl"], cell["split"])] = lng
        g = lng.groupby("model")["pearson_delta"].mean()  # macro over biological unit
        cm = g.get("cell-mean", np.nan)
        lp = g.get("linear-PCA", np.nan)
        floor_mean = np.nanmean([cm, lp])
        floor_score = max(cm, lp)
        for model in cell["roster"]:  # published order; values from the bundles
            if model not in g.index:
                raise SystemExit(
                    f"[assemble] {cell['cl']} {cell['split']}: no bundle for rostered"
                    f" model {model!r}"
                )
            val = g[model]
            fam = FAMILY.get(model, "?")
            rows.append(
                {
                    "cluster": cell["cl"],
                    "task": cell["task"],
                    "split": cell["split"],
                    "unit": cell["unit"],
                    "n_unit": cell["n_unit"],
                    "family": fam,
                    "model": model,
                    "pearson_delta": round(float(val), 4),
                    "floor_cell_mean": round(float(cm), 4),
                    "floor_linear_PCA": round(float(lp), 4),
                    "binding_floor": "cell-mean" if cm >= lp else "linear-PCA",
                    "floor_score": round(float(floor_score), 4),
                    "margin": round(float(val - floor_score), 4),
                    "floor_mean": round(float(floor_mean), 4),
                    "delta_vs_floor_mean": round(float(val - floor_mean), 4),
                    "delta_vs_cell_mean": round(float(val - cm), 4),
                    "delta_vs_linear_PCA": round(float(val - lp), 4),
                    "beats_both_floor_members": bool(val > cm and val > lp),
                    "beats_floor_mean": bool(val > floor_mean),
                    "status": cell_status(cell["cl"], cell["split"], model),
                    "execution_model": EXECUTION_MODEL.get(
                        (cell["cl"], cell["split"], model), model
                    ),
                }
            )
    head = pd.DataFrame(rows)

    # ---------------- WITHIN-FAMILY CONSISTENCY: families with >=2 models on a task ----------------
    con = []
    for (cl, task, split), grp in head.groupby(["cluster", "task", "split"]):
        for fam, fg in grp.groupby("family"):
            if len(fg) < 2:
                continue
            beats = fg["beats_both_floor_members"].tolist()
            con.append(
                {
                    "cluster": cl,
                    "task": task,
                    "split": split,
                    "family": fam,
                    "models": "+".join(fg["model"].tolist()),
                    "deltas_vs_floor_mean": fg["delta_vs_floor_mean"].tolist(),
                    "n_beat_both_floor": int(sum(beats)),
                    "n_models": len(fg),
                    "verdict_agreement": (
                        "agree" if (all(beats) or not any(beats)) else "split"
                    ),
                }
            )
    cons = pd.DataFrame(con)

    def family_rho(cl, split, models):
        """Spearman rho between two models' per-unit pearson_delta vectors (bundle-sourced)."""
        if len(models) != 2:
            return np.nan
        lng = long_by_cell.get((cl, split))
        if lng is None:
            return np.nan
        vecs = {
            m: lng[lng.model == m].set_index("unit")["pearson_delta"] for m in models
        }
        a, b = vecs[models[0]], vecs[models[1]]
        common = a.index.intersection(b.index)
        if len(common) < 3:
            return np.nan
        return float(a.loc[common].corr(b.loc[common], method="spearman"))

    if not cons.empty:
        cons["spearman_rho_pair"] = [
            family_rho(r.cluster, r.split, r.models.split("+"))
            for r in cons.itertuples()
        ]
        cons["flag"] = np.where(
            cons["cluster"] == "C4",
            "C4: 2 modality folds only (rho undefined, <3 biological-unit replicates);"
            " descriptive only, no inferential CI by design",
            "",
        )
    return head, cons, len(df)


def census_bundle_manifest(scored):
    """Identify the exact method and floor bundles used by the headline.

    The repository also preserves sensitivity and superseded outputs. Their
    presence in the store does not make them inputs to a reported census cell.
    """
    rows = []
    for cell in CELLS:
        task = cell["task_id"]
        if task == "T5":
            task = "T5u" if cell["split"] == "unseen-compound" else "T5c"
        selected = scored[
            scored["cluster"].isin(cell["clusters"])
            & scored["split"].map(cell["match"])
            & scored["model"].isin(cell["roster"] + ["cell-mean", "linear-PCA"])
        ].copy()
        selected["unit"] = selected.apply(cell["unit_of"], axis=1)
        duplicate = selected.duplicated(["model", "unit"], keep=False)
        if duplicate.any():
            raise ValueError(
                f"Ambiguous census input for {task}: "
                f"{selected.loc[duplicate, ['model', 'unit', 'bundle_path']].to_dict('records')}"
            )
        expected_units = 1 if task == "T5u" else cell["n_unit"]
        for model in cell["roster"] + ["cell-mean", "linear-PCA"]:
            found = selected[selected["model"] == model]
            if len(found) != expected_units:
                raise ValueError(
                    f"Incomplete {task}/{model}: {len(found)} bundle units, expected"
                    f" {expected_units}"
                )
        for _, row in selected.iterrows():
            path = os.path.join(ROOT, row.bundle_path)
            with open(path, "rb") as handle:
                sha = hashlib.sha256(handle.read()).hexdigest()
            rows.append(
                {
                    "task": task,
                    "model": row.model,
                    "execution_model": row.execution_model,
                    "role": (
                        "floor"
                        if row.model in {"cell-mean", "linear-PCA"}
                        else "method"
                    ),
                    "unit": row.unit,
                    "split": row.split,
                    "dataset": row.dataset,
                    "bundle_path": row.bundle_path,
                    "sha256": sha,
                }
            )
    return pd.DataFrame(rows)


def canonical_numbers(head, scored, manifest):
    """Machine-readable counts with bundle-store and actual census-input counts distinguished."""
    task_of = lambda row: (
        ("T5u" if row.split == "unseen-compound" else "T5c")
        if row.cluster == "C5"
        else "T" + row.cluster[1:]
    )
    tasks = head.apply(task_of, axis=1)
    conditioned = head["status"] != "diagnostic"
    result = {
        "census_cells": len(head),
        "status": head["status"].value_counts().to_dict(),
        "per_task": tasks.value_counts(sort=False).to_dict(),
        "per_task_models": tasks[conditioned].value_counts(sort=False).to_dict(),
        "models": sorted(head.model.unique()),
        "n_models": int(head.model.nunique()),
        "n_conditioned": int(conditioned.sum()),
        "adapted_cells": [
            f"{row.model} @ {task_of(row)}"
            for row in head.itertuples()
            if row.status == "adapted"
        ],
        "clears_both": [
            {
                "task": task_of(row),
                "model": row.model,
                "status": row.status,
                "pd": row.pearson_delta,
            }
            for row in head.itertuples()
            if row.beats_both_floor_members
        ],
        "floors": {
            task: {
                "cell_mean": float(group.floor_cell_mean.iloc[0]),
                "linear_PCA": float(group.floor_linear_PCA.iloc[0]),
            }
            for task, group in head.groupby(tasks, sort=False)
        },
        "submitted_cells": 35,
        "new_models": ["Biolord", "CellFlow", "PRnet", "PerturbNet"],
        "census_bundles": len(manifest),
        "method_prediction_bundles": int((manifest.role == "method").sum()),
        "floor_prediction_bundles": int((manifest.role == "floor").sum()),
        "rescored_bundles": len(scored),
        "bundle_count_definition": (
            "census_bundles counts only method and two-member-floor inputs "
            "listed in census_bundle_manifest.csv; rescored_bundles also "
            "includes preserved non-census references and sensitivity splits"
        ),
        "execution_variants": [
            {
                "cluster": cluster,
                "split": split,
                "model": model,
                "execution_model": source,
            }
            for (cluster, split, model), source in EXECUTION_MODEL.items()
        ],
        "withdrawn_from_census": [
            "SCREEN (all cells)",
            "PertAdapt @ T3",
            "scPRAM @ T4",
            "scFoundation @ T3",
            "scFoundation @ T4",
            "STATE @ T4",
            "STATE @ T3 (adata_real recovery error)",
            "STATE @ T5c (adata_real recovery error)",
            "STATE @ T5u (adata_real recovery error)",
            "PerturbNet @ T5c",
            "CPA @ T1 (latent-shift adaptation)",
            "CPA @ T2 (latent-shift adaptation)",
            "CPA @ T5c (fingerprint adaptation)",
            "scGen @ T5c (fingerprint adaptation)",
            "CellOT @ T4 (pooled-KO map without target-KO input)",
        ],
    }
    return result


def main():
    scored = score_all()
    head, cons, n = build(scored)
    manifest = census_bundle_manifest(scored)
    canon = canonical_numbers(head, scored, manifest)
    head.to_csv(os.path.join(OUT, "cross_cluster_headline.csv"), index=False)
    cons.to_csv(os.path.join(OUT, "within_family_consistency.csv"), index=False)
    manifest.to_csv(os.path.join(OUT, "census_bundle_manifest.csv"), index=False)
    with open(os.path.join(OUT, "CANONICAL_NUMBERS.json"), "w") as handle:
        json.dump(canon, handle, indent=2)
        handle.write("\n")
    print(
        "=== HEADLINE (response-direction delta vs universal floor {cell-mean,"
        " linear-PCA}) ==="
    )
    print(
        head[
            [
                "cluster",
                "task",
                "split",
                "family",
                "model",
                "pearson_delta",
                "floor_score",
                "margin",
                "beats_both_floor_members",
            ]
        ].to_string(index=False)
    )
    print(f"\n[bundle-sourced] {n} bundles re-scored -> {len(head)} headline cells")
    print("\n=== WITHIN-FAMILY CONSISTENCY ===")
    if not cons.empty:
        print(
            cons[
                [
                    "cluster",
                    "task",
                    "family",
                    "models",
                    "n_beat_both_floor",
                    "n_models",
                    "verdict_agreement",
                    "spearman_rho_pair",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
