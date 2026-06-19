"""ClusterSpec registry — the single, uniform way every cluster (C1–C5) is declared.

This is what makes the benchmark extensible and multi-agent-ready: an agent owning a cluster fills in
one ClusterSpec (loaders + splits + program + baselines) against the frozen shared contracts
(data.preprocess, BaselineAdapter, metrics, gating, report). The driver is fully generic over the
registry; adding a cluster never touches the runner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..baselines.chemistry import FPRidge
from ..baselines.heavy import CPA as _CPA
from ..baselines.heavy import CPAchem as _CPAchem
from ..baselines.heavy import CINEMAOT as _CINEMAOT
from ..baselines.heavy import STATE as _STATE
from ..baselines.heavy import PertAdapt as _PertAdapt
from ..baselines.heavy import STATEc5 as _STATEc5
from ..baselines.heavy import ScGenC5 as _ScGenC5
from ..baselines.heavy import ScGenC1 as _ScGenC1
from ..baselines.heavy import CPAC1 as _CPAC1
from ..baselines.heavy import AttentionPert as _AttnPert
from ..baselines.heavy import GEARS as _GEARS
from ..baselines.heavy import ScGen as _ScGen
from ..baselines.heavy import ScGPT as _ScGPT
from ..baselines.heavy import ScFoundation as _ScFoundation
from ..baselines.heavy import CellOT as _CellOT
from ..baselines.heavy import STATE as _STATEcls
from ..baselines.heavy import PertAdapt as _PertAdaptcls
from ..baselines.simple import SIMPLE_BASELINES
from ..data.schema import CellSet
from ..splits.spec import SplitSpec
from . import c1, c2, c3, c4, c5


@dataclass
class ClusterSpec:
    name: str
    title: str
    fig_no: str
    program_key: str                                   # uns["immune_program"][key] for fixtures
    program_genes: list[str]                           # real-data fallback gene set (symbols)
    splits: Callable[[CellSet], list[SplitSpec]]
    baselines: list                                    # adapter classes (implemented ones only)
    load_synth: Callable[[], CellSet] | None = None
    load_real: Callable[[], CellSet] | None = None
    downstream_only: bool = False                      # exclude perturbed target gene from Pearson-Δ
    datasets: dict | None = None                       # multi-dataset clusters: {name: {loader, modality}}
    programs: dict | None = None                       # dataset-aware multi-program sets {name: [genes]}
    extra: dict = field(default_factory=dict)

    def load(self, real: bool) -> CellSet:
        use_real = (real or self.load_synth is None) and self.load_real is not None
        if use_real:
            return self.load_real()
        if self.load_synth is None:
            raise RuntimeError(f"{self.name}: no loader available (need real data)")
        return self.load_synth()

    def program(self, cs: CellSet) -> list[str]:
        return cs.uns.get("immune_program", {}).get(self.program_key) or self.program_genes

    def program_sets(self, cs: CellSet) -> dict[str, list[str]]:
        """Dataset-aware program sets {name: [genes]} for the immune-program axis. Falls back to a
        single set keyed by program_key when no multi-program dict is declared (C1/C5)."""
        if self.programs:
            return self.programs
        g = self.program(cs)
        return {self.program_key: g} if g else {}


REGISTRY: dict[str, ClusterSpec] = {}


def register(spec: ClusterSpec) -> None:
    REGISTRY[spec.name] = spec


def _c5_held_compounds(cs: CellSet) -> list[str]:
    # hold a substantial, seed-reproducible ~20% of fingerprinted compounds (was 5/141 — too thin for
    # a real unseen-compound test or a Tanimoto near/far split). Random over the fingerprinted set
    # spans chemical space, enabling the post-hoc Tanimoto stratification.
    import numpy as _np
    fps = cs.side_info.get("fingerprint", {})
    cpds = sorted(c for c in cs.uns.get("compounds", []) if c in fps) or list(cs.uns.get("compounds", []))
    n_hold = max(5, int(round(0.20 * len(cpds))))
    if len(cpds) <= n_hold:
        return cpds[:max(2, len(cpds) // 2)]
    idx = _np.random.default_rng(0).choice(len(cpds), n_hold, replace=False)
    return [cpds[i] for i in sorted(idx)]


def _c5_splits(cs: CellSet) -> list:
    """Unseen-compound holdout + a leave-one-cell-type-out split for EVERY lineage that has treated
    cells (all-lineage LOCT: T/B/NK/myeloid on real OP3), not NK-only."""
    splits = [c5.global_compound_holdout(_c5_held_compounds(cs))]
    if "cell_type_coarse" in cs.obs:
        ct = cs.obs["cell_type_coarse"].astype(str)
        is_ctrl = cs.obs["is_control"].astype(bool) if "is_control" in cs.obs else None
        lineages = sorted(ct.unique())
        if is_ctrl is not None:                       # keep only lineages with enough treated cells
            treated = ct[~is_ctrl]
            lineages = [l for l in lineages if int((treated == l).sum()) >= 50]
        splits += [c5.cross_celltype_loct(l) for l in lineages]
    return splits


def _lazy(modpath: str, fn: str, **kw):
    import importlib
    return lambda: getattr(importlib.import_module(modpath), fn)(**kw)


def _c1_splits(cs: CellSet) -> list:
    """Real Kang: (cell-context axis) all-coarse-lineage LOCT + (donor axis) leave-one-donor-out with a
    matched random-cell-split control quantifying random-split inflation. Synthetic: original design."""
    if not cs.uns.get("dataset", "").startswith("kang"):
        return [c1.resolution_fine_loct(), c1.cd4_state_transfer(), c1.locyt()]
    import numpy as _np
    ct = cs.obs["cell_type_coarse"].astype(str)
    is_ctrl = cs.obs["is_control"].astype(bool)
    treated = ct[~is_ctrl]
    lineages = [l for l in sorted(ct.unique()) if int((treated == l).sum()) >= 50]
    splits = [c1.coarse_loct(l) for l in lineages]                      # cell-context axis

    # DONOR axis: one LODO split per donor + a matched random-cell-split control. The random folds
    # ignore donor_id (assigned here into a transient `_rand_fold` obs column) so each random fold holds
    # the SAME #treated cells as the corresponding donor — isolating random-split optimism from #held.
    donors = sorted(cs.obs["donor_id"].astype(str).unique())
    donors = [d for d in donors if int(((cs.obs["donor_id"].astype(str) == d) & ~is_ctrl).sum()) >= 50]
    rng = _np.random.default_rng(0)
    rand_fold = _np.array(["__none__"] * cs.n_cells, dtype=object)
    treated_idx = _np.where(~is_ctrl.to_numpy())[0]
    ctrl_idx = _np.where(is_ctrl.to_numpy())[0]
    # match each random fold's TREATED count to that donor's treated count; controls split evenly
    perm_t = rng.permutation(treated_idx); perm_c = rng.permutation(ctrl_idx)
    ti = ci = 0
    for d in donors:
        nt = int(((cs.obs["donor_id"].astype(str) == d) & ~is_ctrl).sum())
        nc = max(1, len(ctrl_idx) // len(donors))
        fold = f"f{d}"
        rand_fold[perm_t[ti:ti + nt]] = fold; ti += nt
        rand_fold[perm_c[ci:ci + nc]] = fold; ci += nc
    cs.obs["_rand_fold"] = rand_fold                                    # transient; used by random_cell_split
    splits += [c1.donor_lodo(d) for d in donors]
    splits += [c1.random_cell_split(f"f{d}") for d in donors]
    return splits


# ---- C1 cytokine-response ----
from ..data.synth_c1 import make_c1_like  # noqa: E402
register(ClusterSpec(
    name="C1", title="Cytokine-response prediction", fig_no="Figure 3",
    program_key="type_i_ifn",
    program_genes=["ISG15", "IFI6", "MX1", "MX2", "OAS1", "OAS2", "IFIT1", "IFIT3", "ISG20",
                   "STAT1", "IRF7", "IFI44", "IFI44L", "RSAD2", "USP18"],
    splits=_c1_splits,
    # latent cross-cell-type models join the simple floors: scGen (Kang's original benchmark model) +
    # CPA, both classic δ-arithmetic on the SEEN cytokine. scGPT-C1/STATE-C1 = deferred (foundation
    # conditioning port / single-perturbation unsuited); CINEMA-OT undefined for a hidden held lineage.
    baselines=list(SIMPLE_BASELINES) + [_ScGenC1, _CPAC1],
    load_synth=lambda: make_c1_like(seed=0),
    load_real=_lazy("ivcbench.data.loaders.kang", "load"),
))

# ---- C2 donor-generalization — Soskic CD4 activation leave-one-donor-out (Fig-1 OT-STRONG donor axis)
def _c2_splits(cs: CellSet) -> list:
    """One leave-one-donor-out split per QC'd donor that has ≥1 stimulated (16h) cell. The held donor's
    16h cells are predicted from its OWN 0h controls; leak-safe (audit_split-enforced). A donor cap can
    be set via $IVCBENCH_C2_MAX_DONORS for a CPU smoke (sorted-first K donors)."""
    import os as _os
    is_ctrl = cs.obs["is_control"].astype(bool)
    donors = sorted(cs.obs["donor_id"].astype(str).unique())
    donors = [d for d in donors
              if int(((cs.obs["donor_id"].astype(str) == d) & ~is_ctrl).sum()) >= 1]
    cap = _os.environ.get("IVCBENCH_C2_MAX_DONORS")
    if cap:
        donors = donors[: int(cap)]
    return [c2.donor_lodo(d) for d in donors]


register(ClusterSpec(
    name="C2", title="Donor-generalization (CD4 activation transfer)", fig_no="Figure 4",
    program_key="T_cell_activation", program_genes=c2.SOSKIC_PROGRAMS["T_cell_activation"],
    programs=c2.SOSKIC_PROGRAMS,                        # dataset-aware multi-program AUCell (CD4 activation)
    splits=_c2_splits,
    # OT-STRONG paired-stimulation donor axis: every cell-axis family is applicable on C2_LODO.
    # simple×4 (floors) + scGen (latent δ-arithmetic, the donor-transfer headline) + CellOT (conditioned
    # OT, the +0.102 headline) + STATE + PertAdapt (hybrid). chemistry/graph families are inapplicable
    # (no compound/gene-side rep on a seen-perturbation donor split).
    baselines=list(SIMPLE_BASELINES) + [_ScGenC1, _CellOT, _STATEcls, _PertAdaptcls],
    load_real=_lazy("ivcbench.data.loaders.soskic", "load"),
    # leak-safe TRAINING-only response-gene panel for the Pearson-Δ metric (the bespoke Soskic rule),
    # consumed by run_cluster -> run_job(response_gene_fn=...). Never seen by any model.
    extra={"response_gene_fn": c2.response_gene_idx},
))

# ---- C3 gene-intervention — Q1 leaderboard across focused-hit human-T datasets (modality-stratified)
#   Chen 2025 (real, human FOXP3 Perturb-icCITE-seq) = DDBJ PRJDB16517 / GEA E-GEAD-648, now an active
#   dataset below. The accession GSE255832 we first mis-mapped to "Chen" is actually Pretto 2025
#   (mouse) and belongs to C4 — see scripts/data_provenance_verification.md.
def _c3_splits(cs):
    g = cs.uns["genes_perturbed"]
    return [c3.true_lo_gene(c3.held_gene_fraction(g, f, seed=0), lbl)
            for f, lbl in [(0.10, "10"), (0.25, "25"), (0.50, "50")]]

register(ClusterSpec(
    name="C3", title="Gene-intervention prediction", fig_no="Figure 5",
    program_key="tcell_activation", program_genes=c3.TCELL_ACTIVATION,
    programs=c3.C3_PROGRAMS,                            # dataset-aware multi-program AUCell (Supp Table S3)
    splits=_c3_splits,
    # Family coverage (every family has ≥1 method): simple×4, latent={scGen,CPA}, graph={GEARS,
    # AttentionPert}, foundation={scGPT} (UCE=not-defined, no decoder), hybrid={STATE, PertAdapt} (GPU,
    # frozen scFoundation + GO-masked adapter), ot={CINEMA-OT} run as a perturbation-agnostic not_defined†
    # floor (CellOT software unavailable + undefined for unseen-gene → documented, not run).
    baselines=list(SIMPLE_BASELINES) + [_GEARS, _AttnPert, _ScGPT, _ScFoundation, _ScGen, _CPA, _STATE,
                                        _PertAdapt, _CINEMAOT],
    downstream_only=True,
    datasets={
        "shifrut":            {"loader": _lazy("ivcbench.data.loaders.shifrut", "load"), "modality": "KO"},
        "schmidt":            {"loader": _lazy("ivcbench.data.loaders.schmidt", "load"), "modality": "CRISPRa"},
        "mccutcheon_CRISPRi": {"loader": _lazy("ivcbench.data.loaders.mccutcheon", "load", modality="CRISPRi"), "modality": "CRISPRi"},
        "mccutcheon_CRISPRa": {"loader": _lazy("ivcbench.data.loaders.mccutcheon", "load", modality="CRISPRa"), "modality": "CRISPRa"},
        "chen":               {"loader": _lazy("ivcbench.data.loaders.chen", "load"), "modality": "KO"},
    },
))

# ---- C4 complex-context — Axis 2: RNA → protein modality generalization (Frangieh) ----
def _c4_splits(cs: CellSet) -> list:
    g = cs.uns["genes_perturbed"]
    return [c4.modality_lo_ko(c4.held_ko_fraction(g, f, seed=0), lbl)
            for f, lbl in [(0.25, "25"), (0.50, "50")]]

register(ClusterSpec(
    name="C4", title="Complex-context perturbation prediction (RNA→protein modality)", fig_no="Figure 6",
    program_key="exhaustion", program_genes=[],   # melanoma KO; AUCell program not applicable (documented)
    splits=_c4_splits,
    baselines=list(SIMPLE_BASELINES),              # modality recoverability comparison (clean, leak-safe)
    downstream_only=True,                          # exclude the KO'd gene from RNA Pearson-Δ (absent in protein)
    datasets={
        "frangieh_RNA":     {"loader": _lazy("ivcbench.data.loaders.frangieh", "load", modality="rna"),     "modality": "RNA"},
        "frangieh_protein": {"loader": _lazy("ivcbench.data.loaders.frangieh", "load", modality="protein"), "modality": "protein-CITE"},
    },
))

# ---- C5 small-molecule ----
from ..data.synth import make_op3_like  # noqa: E402
register(ClusterSpec(
    name="C5", title="Small-molecule perturbation prediction", fig_no="Figure 7",
    program_key="immunomod_moa", program_genes=[],
    programs=c5.C5_PROGRAMS,                            # dataset-aware immunomod-MoA AUCell programs (Axis 3)
    splits=_c5_splits,                                  # unseen-compound holdout + all-lineage LOCT
    # full roster: simple-4 + FP-ridge (chemistry) + chemCPA (latent, applicable) + scGen-C5 (latent
    # adapted*) + STATE-C5 (hybrid adapted*) + CINEMA-OT (ot, not-defined† floor). CellOT not installed
    # → documented not-run; UCE encoder-only → not-defined; scGPT-C5 (foundation) is the noted next port.
    baselines=list(SIMPLE_BASELINES) + [FPRidge, _CPAchem, _ScGenC5, _STATEc5, _CINEMAOT],
    load_synth=lambda: make_op3_like(seed=0),
    load_real=_lazy("ivcbench.data.loaders.op3", "load"),
))
