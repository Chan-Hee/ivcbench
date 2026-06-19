"""13 baselines x 9 split tasks applicability matrix (Master Overview, page 2).

The runner consults status_for() to decide what to do with each (baseline, split):
  applicable   -> run, eligible for headline ranking
  adapted      -> run only if the conditioning extension is implemented; report separately
  not_defined  -> floor reference only, EXCLUDED from headline ranking
  inapplicable -> skip entirely (family mismatch)
This is what closes the "vanilla label-conditioned model faked onto unseen-perturbation" over-claim.
"""
from __future__ import annotations

from enum import Enum


class Status(str, Enum):
    APPLICABLE = "applicable"      # ✓
    ADAPTED = "adapted"            # ⚙
    NOT_DEFINED = "not_defined"    # —
    INAPPLICABLE = "inapplicable"  # ×


A, G, N, X = Status.APPLICABLE, Status.ADAPTED, Status.NOT_DEFINED, Status.INAPPLICABLE

SPLIT_TASKS = [
    "C1_LOCT", "C1_state", "C1_LOcyt",
    "C2_LODO", "C2_temporal",
    "C3_LO_gene",
    "C4_Axis1", "C4_Axis2",
    "C5_unseen_cpd",
]

# rows transcribed from the applicability matrix (page 2)
APPLICABILITY: dict[str, dict[str, Status]] = {
    "ctrl-pred":     dict(zip(SPLIT_TASKS, [A, A, N, A, N, A, A, A, N])),
    "cell-mean":     dict(zip(SPLIT_TASKS, [A, A, N, A, N, A, A, A, N])),
    "donor-shift":   dict(zip(SPLIT_TASKS, [A, A, N, A, N, A, A, A, N])),
    "linear-PCA":    dict(zip(SPLIT_TASKS, [A, A, N, A, N, A, A, A, N])),
    "scGen":         dict(zip(SPLIT_TASKS, [A, A, N, A, G, G, A, A, G])),
    "CPA":           dict(zip(SPLIT_TASKS, [A, A, G, A, G, G, A, A, A])),  # chemCPA on C5
    "GEARS":         dict(zip(SPLIT_TASKS, [X, X, X, X, X, A, A, N, X])),
    "AttentionPert": dict(zip(SPLIT_TASKS, [X, X, X, X, X, A, A, N, X])),
    "scGPT":         dict(zip(SPLIT_TASKS, [A, A, G, A, G, A, A, A, G])),
    # scFoundation: 2nd foundation model. Native unseen-gene prediction on C3_LO_gene (index 5) via a
    # frozen-embedding + one-hot-conditioned fine-tune head → applicable there (same pattern as scGPT
    # on the gene-intervention task). Only C3 is exercised in this roster; other tasks mirror scGPT.
    "scFoundation":  dict(zip(SPLIT_TASKS, [A, A, G, A, G, A, A, A, G])),
    # UCE is encoder-only (no decoder) → cannot emit a predicted expression profile for ANY gene-
    # intervention task; CZI docs confirm "no in silico perturbation". So it is not_defined for the
    # C3_LO_gene prediction task (index 5), not merely unrun. (Other tasks left as transcribed.)
    "UCE":           dict(zip(SPLIT_TASKS, [A, A, N, A, G, N, A, A, G])),
    "CellOT":        dict(zip(SPLIT_TASKS, [A, A, N, A, G, N, A, A, G])),
    "CINEMA-OT":     dict(zip(SPLIT_TASKS, [A, A, G, A, G, N, A, A, G])),
    # scPRAM (Jiang et al. 2024): 2nd conditioned Optimal-Transport model (VAE + OT cell-matching +
    # per-cell attention). Needs PAIRED ctrl/stimulation conditioned on (cell_type, condition) →
    # applicable on the Fig1 OT-STRONG paired-stimulation cells: Cytokine (Kang C1_LOCT, index 0) and
    # Donor (Soskic C2_LODO, index 3), alongside CellOT. Not defined on the unseen-gene CRISPR-KO split
    # C3_LO_gene (Frangieh KO is not paired stimulation) or the other tasks in this roster.
    "scPRAM":        dict(zip(SPLIT_TASKS, [A, N, N, A, N, N, N, N, N])),
    "STATE":         dict(zip(SPLIT_TASKS, [A, A, A, A, G, A, A, A, G])),
    # PertAdapt (Bai et al. 2025): 2nd Hybrid-family model. Frozen scFoundation + GO-masked perturbation
    # adapter + adaptive DE loss → native unseen-gene prediction → applicable on the CRISPR gene-
    # intervention split C3_LO_gene (index 5) and the Soskic donor-transfer split C2_LODO (index 3): the
    # Fig1 Hybrid-STRONG cells, alongside STATE. Not exercised on the other tasks in this roster (left
    # not_defined; a compound/cytokine adaptation is out of scope here, distinct from STATE's C5 variant).
    "PertAdapt":     dict(zip(SPLIT_TASKS, [N, N, N, A, N, A, N, N, N])),
    # FP-ridge: chemistry-aware reference (Morgan fingerprint). Applicable wherever a compound-side
    # representation is defined — the unseen-compound split and the C5 cross-cell-type (seen-compound,
    # cell-axis) split, which reuses the C1_LOCT applicability pattern. C5-only (no fingerprints else).
    "FP-ridge":      dict(zip(SPLIT_TASKS, [A, X, X, X, X, X, X, X, A])),
    # linear-shift-KOemb: a fully-reproducible CPU conditioned baseline (per-KO gene-space shift
    # ridge-regressed on a leak-safe control-only-PCA gene embedding). It is the second conditioned
    # family on the C4 modality axis (distinct from scGen's latent-VAE δ). Applicable on the unseen-KO
    # modality tasks (C4_Axis1/Axis2) where a gene-side embedding for the held KO is defined; not
    # defined elsewhere. Registering the name closes the prior KeyError so run_job emits a first-class row.
    "linear-shift-KOemb": dict(zip(SPLIT_TASKS, [N, N, N, N, N, N, A, A, N])),
}


def status_for(baseline: str, split_task: str) -> Status:
    return APPLICABILITY[baseline][split_task]


def headline_eligible(baseline: str, split_task: str) -> bool:
    """Only ✓ applicable cells enter the leaderboard ranking."""
    return status_for(baseline, split_task) is Status.APPLICABLE
