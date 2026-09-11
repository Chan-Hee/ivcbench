"""Runtime applicability matrix for the generic fitting driver.

The runner consults status_for() to decide what to do with each (baseline, split):
  applicable   -> run, eligible for headline ranking
  adapted      -> run only if the conditioning extension is implemented; report separately
  not_defined  -> floor reference only, EXCLUDED from headline ranking
  inapplicable -> skip entirely (family mismatch)
These flags select execution paths, not the final native/adapted/diagnostic
panel. Historical runnable configurations may be excluded after interface or
artifact validation. The curated census and model_task_interfaces.csv determine
which results are reported in the submission.
"""

from __future__ import annotations

from enum import Enum


class Status(str, Enum):
    APPLICABLE = "applicable"  # ✓
    ADAPTED = "adapted"  # ⚙
    NOT_DEFINED = "not_defined"  # —
    INAPPLICABLE = "inapplicable"  # ×


A, G, N, X = Status.APPLICABLE, Status.ADAPTED, Status.NOT_DEFINED, Status.INAPPLICABLE

SPLIT_TASKS = [
    "C1_LOCT",
    "C5_LOCT",
    "C1_state",
    "C1_LOcyt",
    "C2_LODO",
    "C2_temporal",
    "C3_LO_gene",
    "C4_Axis1",
    "C4_Axis2",
    "C5_unseen_cpd",
]

# Column order is SPLIT_TASKS; preserve it when changing a runtime entry.
APPLICABILITY: dict[str, dict[str, Status]] = {
    "ctrl-pred": dict(zip(SPLIT_TASKS, [A, A, A, N, A, N, A, A, A, N])),
    "cell-mean": dict(zip(SPLIT_TASKS, [A, A, A, N, A, N, A, A, A, N])),
    "donor-shift": dict(zip(SPLIT_TASKS, [A, A, A, N, A, N, A, A, A, N])),
    "linear-PCA": dict(zip(SPLIT_TASKS, [A, A, A, N, A, N, A, A, A, N])),
    "scGen": dict(zip(SPLIT_TASKS, [A, A, A, N, A, G, N, N, N, N])),
    "CPA": dict(zip(SPLIT_TASKS, [A, A, A, G, A, G, N, N, N, N])),  # chemCPA on C5
    # PRnet (Qi et al., Nat Commun 2024): the Perturb-adaptor consumes the compound fingerprint, so
    # both OP3 splits run through the released interface; it has no gene- or stimulus-side input.
    "PRnet": dict(zip(SPLIT_TASKS, [N, A, N, N, N, N, N, N, N, A])),
    "GEARS": dict(zip(SPLIT_TASKS, [X, X, X, X, X, X, A, A, A, X])),
    "AttentionPert": dict(zip(SPLIT_TASKS, [X, X, X, X, X, X, A, A, A, X])),
    "scGPT": dict(zip(SPLIT_TASKS, [N, N, A, G, N, G, A, A, A, N])),
    # Runtime genetic support is not census admission: the historical unseen-gene
    # head lacks a target-specific condition and is excluded from the final panel.
    "scFoundation": dict(zip(SPLIT_TASKS, [N, N, A, G, N, G, A, A, A, N])),
    # No expression-prediction decoder is implemented for this encoder interface.
    "UCE": dict(zip(SPLIT_TASKS, [N, N, N, N, N, N, N, N, N, N])),
    # SCREEN (2024): published cell-context interface (held group predicted from its own
    # controls); applicable wherever a seen perturbation transfers to a held group.
    "SCREEN": dict(zip(SPLIT_TASKS, [A, A, A, N, A, N, N, N, N, N])),
    "CellOT": dict(zip(SPLIT_TASKS, [A, A, A, N, A, G, N, N, N, N])),
    "CINEMA-OT": dict(zip(SPLIT_TASKS, [A, A, A, G, A, G, N, N, N, N])),
    # Paired response transfer is supported; the pooled genetic map has no held-KO input.
    "scPRAM": dict(zip(SPLIT_TASKS, [A, A, N, N, A, N, N, N, N, N])),
    # Ordered attributes supply gene/chemical representations for unseen interventions.
    "Biolord": dict(zip(SPLIT_TASKS, [A, A, N, N, A, N, A, N, A, A])),
    # ChemicalVAE and GenotypeVAE supply compound and gene conditions. The
    # historical cross-lineage extension is runtime-adapted and not in the panel.
    "PerturbNet": dict(zip(SPLIT_TASKS, [N, G, N, N, N, N, A, N, A, A])),
    "STATE": dict(zip(SPLIT_TASKS, [A, A, A, A, A, G, A, A, A, N])),
    # Frozen foundation encoder with a trained adaptation component; consult the
    # task-specific interface record for the operation actually retained.
    "PertAdapt": dict(zip(SPLIT_TASKS, [A, A, N, N, A, N, A, A, A, N])),
    # FP-ridge: chemistry-aware reference (Morgan fingerprint). Applicable wherever a compound-side
    # representation is defined — the unseen-compound split and the C5 cross-cell-type (seen-compound,
    # cell-axis) split, which reuses the C1_LOCT applicability pattern. C5-only (no fingerprints else).
    "FP-ridge": dict(zip(SPLIT_TASKS, [X, A, X, X, X, X, X, X, X, A])),
    # CPU ridge diagnostic from control-PCA gene embeddings to response shifts.
    "linear-shift-KOemb": dict(zip(SPLIT_TASKS, [X, X, N, N, X, N, A, A, A, X])),
    # Held contexts use split_covariate; held interventions use covariate representations.
    "CellFlow": dict(zip(SPLIT_TASKS, [A, A, N, N, A, N, A, A, A, A])),
    # Unseen-compound interface only; the local execution did not pass the
    # held-response validation gate and is excluded (see map_c5_runner.py).
    "MAP": dict(zip(SPLIT_TASKS, [N, N, N, N, N, N, N, N, N, A])),
}


def status_for(baseline: str, split_task: str) -> Status:
    return APPLICABILITY[baseline][split_task]


def headline_eligible(baseline: str, split_task: str) -> bool:
    """Return the legacy runtime ranking flag, not final-panel eligibility."""
    return status_for(baseline, split_task) is Status.APPLICABLE
