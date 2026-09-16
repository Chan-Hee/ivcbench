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

import _env_guard  # noqa: F401  (fails fast on the stale package copy)
import glob
import hashlib
import json
import os
import re

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
# Artifact-KIND rules: a different readout or conditioning variant, not a withdrawn run. These
# stay patterns because any future run producing that artifact kind is excluded for the same reason.
# Artifact-KIND rules only. `__op3_fcond.npz` used to sit here, when the compound-conditioned
# foundation head was a side evaluation. It is now the T5c foundation cell itself -- both scGPT and
# scFoundation, uniformly -- and the pooled C1-adapter bundles it replaces are withdrawn by exact
# path+sha256 in results/_paper/withdrawn_bundles.csv, which is where a decision about a SPECIFIC
# execution belongs.
_NON_CENSUS_BUNDLE = (
    os.sep + "example" + os.sep,
    os.sep + "_withdrawn" + os.sep,
    # Runs kept as evidence but not as results. _diagnostic holds the STATE T2 bundles produced
    # before the training/inference split (the held donor's controls entered the train subset);
    # _validate holds the short single-unit checks a runner must pass before its full run is
    # queued. Both live under predictions/, so without this they would be scored as census cells.
    os.sep + "_diagnostic" + os.sep,
    os.sep + "_validate" + os.sep,
    "__frangieh_protein.npz",
)


def _withdrawn_bundles():
    """Runs withdrawn under the interface rule, keyed by EXACT path (and recorded sha256).

    This replaced a model-specific filename denylist ("__CPA__C1_loct_", "__scGen__C5_loct_", ...).
    That denylist excluded by model and split, so a NEW native CPA or scGen run on the same split
    matched the same pattern and would have been dropped silently — which is exactly the coverage
    the native-coverage plan orders us to restore. Keying on the path admits a new run automatically
    while the historical bundles stay excluded by name, and results/_paper/withdrawn_bundles.csv
    records why each one was withdrawn.

    A withdrawal names a SPECIFIC EXECUTION, not a filename. A re-run that writes to the same path
    is a different artifact -- different bytes, different sha256 -- and must be admitted, or fixing
    a cell in place would silently delete it from the census. So the sha is checked too: the entry
    excludes the file only while its content is still the withdrawn one.
    """
    import csv as _csv
    path = os.path.join(ROOT, "results", "_paper", "withdrawn_bundles.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "withdrawn_bundles.csv is missing; refusing to run with no withdrawal registry"
        )
    with open(path, encoding="utf-8") as fh:
        return {r["bundle_path"]: (r.get("sha256") or "").strip() for r in _csv.DictReader(fh)}


def _is_withdrawn(path):
    """True only if this file is still the exact execution that was withdrawn."""
    import hashlib

    rel = os.path.relpath(str(path), ROOT)
    want = _WITHDRAWN.get(rel)
    if want is None:
        return False
    if not want:
        return True          # legacy entry with no recorded sha: keep excluding by path
    try:
        with open(path, "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return True
    return got == want


def _withdrawal_reasons():
    """The reason recorded against each withdrawal, for the lapsed-entry check."""
    import csv as _csv

    path = os.path.join(ROOT, "results", "_paper", "withdrawn_bundles.csv")
    with open(path, encoding="utf-8") as fh:
        return {r["bundle_path"]: (r.get("reason") or "") for r in _csv.DictReader(fh)}


_WITHDRAWN = _withdrawn_bundles()
_WITHDRAWN_REASON = _withdrawal_reasons()


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
        or _is_withdrawn(path)
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


def assert_uniform_metric_mask(df):
    """Every model on a unit must be scored on the same genes, or the cell is not a comparison.

    NATIVE_102_FINAL_REVIEW.md section 247 replaces control-padding on the unseen-gene and
    unseen-knockout cells with a common evaluation mask, and the word that carries the weight is
    common: a model scored on 1,974 genes against floors scored on 2,000 is worse than no mask at
    all. The mask reaches a bundle two ways -- run_job writes it for runs started after it landed,
    and scripts/apply_panel_mask.py adds it to everything deposited earlier. If that migration has
    not been run, a cell arrives here half masked, and nothing downstream would notice: the scores
    are all finite and all plausible.

    This is the check that notices. It compares exclude_gene_idx across the bundles of one unit
    and refuses to assemble when they disagree.
    """
    import numpy as np

    bad = []
    for (cluster, split, dataset), grp in df.groupby(["cluster", "split", "dataset"], dropna=False):
        if not (str(split).startswith("C3_true_lo_gene") or str(split).startswith("C4_modality_lo_ko")):
            continue
        if str(dataset) == "frangieh_protein":
            continue
        seen = {}
        for _, row in grp.iterrows():
            path = os.path.join(ROOT, row["bundle_path"])
            try:
                d = np.load(path, allow_pickle=True)
            except Exception:
                continue
            excl = (
                tuple(sorted(np.asarray(d["exclude_gene_idx"], int).tolist()))
                if "exclude_gene_idx" in d.files
                else ()
            )
            seen.setdefault(excl, []).append(row["model"])
        if len(seen) > 1:
            groups = " | ".join(
                f"{len(k)} excluded: {', '.join(sorted(v))}" for k, v in sorted(seen.items(), key=lambda x: -len(x[0]))
            )
            bad.append(f"  {cluster} {split} {dataset or ''}: {groups}")
    if bad:
        raise SystemExit(
            "Refusing to assemble: the models on a unit are not scored on the same genes.\n"
            + "\n".join(bad)
            + "\n\nRun scripts/apply_panel_mask.py --apply first (cascade step 4b). It is the "
            "migration that brings bundles deposited before the mask existed up to the same "
            "exclusion set."
        )


def assert_withdrawals_still_bind():
    """A withdrawal is a decision. A later job writing over the file must not undo it silently.

    _is_withdrawn pins each entry to a sha256 -- "still the exact execution that was withdrawn" --
    so an entry stops applying the moment its file changes. That is right for the withdrawal that
    names its own remedy: the CINEMA-OT bundles were withdrawn because the reference arm collapsed
    the prediction to the training treated mean, and the reason ends "superseded by the re-run
    whose reference also carries the training controls". When that re-run wrote over them, the
    entry lapsing IS the supersession landing.

    It is wrong for a withdrawal that names no replacement. There the file changing means some job
    has written over a bundle a person refused, and the refusal has quietly stopped applying --
    which looks exactly like a bundle that was never withdrawn. apply_panel_mask.py, the one tool
    that rewrites bundles in place, skips withdrawn ones, so nothing benign does this.

    So: flag a lapsed entry only when its reason does not say the bundle was replaced.
    """
    import hashlib

    replaced = re.compile(r"supersede|superseded by|re-run", re.I)
    lapsed, expected = [], 0
    for rel, want in _WITHDRAWN.items():
        if not want:
            continue
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() == want:
            continue
        if replaced.search(_WITHDRAWN_REASON.get(rel, "")):
            expected += 1
            continue
        lapsed.append(rel)
    if expected:
        print(
            f"[withdrawals] {expected} entr(ies) superseded by the re-run they name; "
            "the file on disk is the replacement"
        )
    if lapsed:
        raise SystemExit(
            "withdrawn bundles have been overwritten by something their reason does not name, so "
            f"the withdrawal no longer binds and they are being scored again ({len(lapsed)}):\n  "
            + "\n  ".join(lapsed[:12])
            + ("\n  ..." if len(lapsed) > 12 else "")
            + "\n\nRe-record the decision against the execution now on disk, or restore the file."
        )


def score_all():
    assert_withdrawals_still_bind()
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
# models reported for that cell. The roster fixes the final 56-entry panel (revision_claude/NATIVE_102_FINAL_REVIEW.md;
# verify_census_closure.py fails the build if it drifts from cell_ledger.csv);
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
            "CPA",
            "CellFlow",
            "CellOT",
            "PerturbNet",
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
            "CPA",
            "CellFlow",
            "CellOT",
            "PerturbNet",
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
            "CellFlow",
            "GEARS",
            "PertAdapt",
            "PerturbNet",
            "STATE",
            "linear-shift-KOemb",
            "scFoundation",
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
            "PertAdapt",
            "PerturbNet",
            "STATE",
            "linear-shift-KOemb",
            "scFoundation",
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
        # The two foundation models sit here through the same compound-conditioned head they use on
        # the cell-context split. Reviewer 2 comment 1 named T1, T2 and T5, and the author's rule
        # for this revision is that a foundation model may be adapted because adapting one for a
        # downstream task is ordinary practice. The head is the construction chemCPA already uses
        # and which is reported native on this very split -- a frozen molecular vector feeding a
        # trainable map whose output is decoded through the model's own cell representation -- so
        # admitting it for chemCPA and refusing it here would not be a position we could defend.
        roster=[
            "Biolord",
            "CPA",
            "CellFlow",
            "FP-ridge",
            "PRnet",
            "PerturbNet",
            "STATE",
            "scFoundation",
            "scGPT",
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
            "CPA",
            "CellFlow",
            "CellOT",
            "FP-ridge",
            "PRnet",
            "PerturbNet",
            "STATE",
            "scFoundation",
            "scGPT",
            "scGen",
            "scPRAM",
        ],
    ),
]


def cell_long(df, cell):
    """Long table of per-(model, unit) pearson_delta for one census cell."""
    d = df[df["cluster"].isin(cell["clusters"]) & df["split"].map(cell["match"])].copy()
    d["unit"] = d.apply(cell["unit_of"], axis=1)
    # One bundle per (model, unit). This used to take the MEAN, which silently averages a model
    # with itself when a unit is scored twice -- and the average is a number that describes
    # neither run. Measured: duplicating one real C1/scGen/CD4T bundle moves scGen @ T1 from
    # 0.7497 (margin -0.0307) to 0.7809 (margin +0.0006), flipping the cell above the floor.
    # The refusal did exist, but in census_bundle_manifest(), which both writers happen to call
    # afterwards -- so purity was a call-order convention rather than a property of this
    # function, and any new caller inherited the average. It refuses here now.
    dup = d.groupby(["model", "unit"]).size()
    dup = dup[dup > 1]
    if len(dup):
        raise SystemExit(
            f"{cell['task_id']} {cell['split']}: {len(dup)} (model, unit) pair(s) scored more than once; "
            "one of them must be withdrawn or superseded before the census can be built.\n  "
            + "\n  ".join(f"{m} @ {u}: {n} bundles" for (m, u), n in dup.items())
        )
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
    ("C5", "unseen-compound", "scGPT"): "adapted",
    ("C5", "unseen-compound", "scFoundation"): "adapted",
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
    # PertAdapt x T2 was an adapted cell in the submitted panel. The final ruling excludes it --
    # anti-CD3/CD28 stimulation has no node in the perturbation graph the published adapter
    # conditions on, and the interface carries no donor slot -- so it is no longer a census cell and
    # its disclosure text belongs to Supplementary Table S15b, not here. Leaving it made the
    # adapted-set equality check below fail.
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
    # These two described the T1 latent-shift/global-flag adapter, which was WITHDRAWN on this
    # split: it pooled all 141 OP3 compounds into one exposure, so every compound received an
    # identical profile. The deposited bundles come from the compound-conditioned head, and the
    # disclosure has to describe the interface that produced them.
    ("C5", "cell-context (LOCT)", "scGPT"): (
        "Yes — a compound-conditioned head: the released scGPT_human encoder is held fixed and a "
        "trainable MLP maps its cell embedding concatenated with the compound's Morgan "
        "fingerprint onto the response, fitted on the training fold against observed "
        "lineage-by-compound means."
    ),
    ("C5", "cell-context (LOCT)", "scFoundation"): (
        "Yes — a compound-conditioned head: the released scFoundation encoder is held fixed and a "
        "trainable MLP maps its cell embedding concatenated with the compound's Morgan "
        "fingerprint onto the response, fitted on the training fold against observed "
        "lineage-by-compound means."
    ),
    # Same head, unseen-compound regime: the held compounds are absent from every training cell and
    # reach the model only as a fingerprint, which is what this cell is reported to measure.
    ("C5", "unseen-compound", "scGPT"): (
        "Yes — the same compound-conditioned head as on the cell-context split, with the held "
        "compounds absent from training: the frozen scGPT_human encoder embeds the control pool "
        "and the held compound enters only as its Morgan fingerprint."
    ),
    ("C5", "unseen-compound", "scFoundation"): (
        "Yes — the same compound-conditioned head as on the cell-context split, with the held "
        "compounds absent from training: the frozen scFoundation encoder embeds the control pool "
        "and the held compound enters only as its Morgan fingerprint."
    ),
}

# These guard the assembled census against a silent change of shape. They were left at the
# submitted 47-cell panel when the CELLS rosters were moved to 56, which aborts the assembler --
# retyping a roster in one place and a count in another is exactly how they drift, so
# verify_census_closure.py now checks these two constants against cell_ledger.csv as well.
EXPECTED_CENSUS_CELLS = 58
EXPECTED_STATUS_COUNTS = {"native": 46, "adapted": 8, "diagnostic": 4}


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
    assert_uniform_metric_mask(df)

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
            dup = selected.loc[duplicate, ["model", "unit", "bundle_path"]]
            rerun = dup[dup["bundle_path"].str.contains("v2_native")]
            hint = ""
            if len(rerun):
                hint = (
                    "\n\nA native re-run deposits under the same filename as the bundle it "
                    "replaces, so both are visible here. Resolve it explicitly, never by picking "
                    "one silently:\n"
                    "    python scripts/supersede_reruns.py            # report\n"
                    "    python scripts/supersede_reruns.py --apply    # withdraw the superseded "
                    "bundles by path+sha256\n"
                    "(--apply refuses while any job is still RUNNING.)"
                )
            raise ValueError(
                f"Ambiguous census input for {task}: "
                f"{dup.to_dict('records')}{hint}"
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
        # Derived, not listed. The literal this replaces was a snapshot of an earlier state:
        # 12 of its 15 entries named cells the census REPORTS, because the model was re-run and
        # re-admitted at a new path while the old bundle stayed withdrawn. check_consistency
        # compares stored against rebuilt, so a literal is always equal to itself and the gate was
        # structurally blind to it -- and this is the machine-readable file the availability
        # statement points a reviewer at. A cell belongs here only if a withdrawal still binds
        # against its sha256 AND the census does not report it. The full not-reported record,
        # with a reason for every cell, is Supplementary Table S15b.
        "withdrawn_from_census": withdrawn_cells(head, task_of),
    }
    return result


_TASK_OF_SPLIT_PREFIX = {
    "C1_LOCT": "T1", "C2": "T2", "C2_LODO": "T2", "C3_LO_gene": "T3", "C4_Axis2": "T4",
    "C5_LOCT": "T5c", "C5": "T5u", "C5_unseen_cpd": "T5u",
}


def withdrawn_cells(head, task_of):
    """Cells whose withdrawal still binds and which the census does not report.

    Most withdrawn bundles belong to a cell the census DOES report, at a different path: the
    execution was superseded by a re-run, and it is the execution that was withdrawn, not the
    cell. Those must not appear here. A withdrawal whose sha256 no longer matches has lapsed
    (a same-path re-run), and is not a withdrawal at all.
    """
    reported = {(str(row.model), task_of(row)) for row in head.itertuples()}
    cells = set()
    for path in sorted(_WITHDRAWN):
        if not _is_withdrawn(path):
            continue                      # lapsed: the file on disk is the replacement
        name = os.path.basename(path)[:-4].split("__")
        prefix, model = name[0], name[1]
        task = _TASK_OF_SPLIT_PREFIX.get(prefix)
        if task is None:
            raise SystemExit(f"withdrawn bundle with an unknown split prefix: {path}")
        # The census names this cell by its method group; chemCPA is the CPA group's execution.
        model = {v: k[2] for k, v in EXECUTION_MODEL.items()}.get(model, model)
        if (model, task) not in reported:
            cells.add(f"{model} @ {task}")
    return sorted(cells)



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
