#!/usr/bin/env python
"""Generic driver for the admission obligations: every (model, task) the derived admission matrix
marks native must be reported. One entry point so each cell is produced by the same split builder,
floors, metric and seed policy as the deposited census.

    <python> scripts/run_obligation.py --model SCREEN --task T1 [--chunk i n] [--gpu g] --out <csv>
    ... --smoke   # one unit only, to prove the path before spending the sweep
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ivcbench.clusters import c1, c2, c4, c5
from ivcbench.baselines.heavy import (
    ScreenC1,
    ScPRAM,
    StateC1,
    PertAdaptC1,
    PertAdapt,
    BiolordC1,
    BiolordC5,
    CellOTC1,
    StateC1Split,
    CINEMAOTC1,
    CPAC1,
    PertAdaptC3,
    PerturbNetC1,
    ScFoundationGene,
    CellOTC5,
    CPAchem,
    CINEMAOT,
    PerturbNetC5,
    CellFlowC5,
    PRnetC5,
    StateC5cSplit,
)
from ivcbench.baselines.heavy import CellFlowC1, CellFlowGene
from ivcbench.baselines.heavy import PerturbNetC3
from ivcbench.baselines.heavy import BiolordC3
from run_c4_conditioned import LinearShiftKOEmb
from ivcbench.baselines.fp_wrapper import FPUnseenCompound
from ivcbench.baselines.heavy import MAPC5

# compound-conditioned foundation heads (R2-1): frozen released encoder + a trainable head
# over [cell embedding || Morgan fingerprint], trained against the OBSERVED response.
from ivcbench.baselines.heavy import ScGPTC5Cond
from ivcbench.baselines.heavy import ScFoundationC5Cond
from ivcbench.baselines.heavy import (
    ScGPTC1,
    ScFoundationC1,
    ScGPT,
    ScFoundation,
    GEARS,
    AttentionPert,
    STATE,
    ScGen,
    CPA,
    CellOT,
)


def _fp(base):
    """unseen-compound extension over a base adapter (Morgan fingerprint -> model response)"""
    return lambda: FPUnseenCompound(base)


from ivcbench.runner.run import run_job

ADAPTER = {
    ("SCREEN", "T1"): ScreenC1,
    ("SCREEN", "T2"): ScreenC1,
    ("SCREEN", "T5c"): ScreenC1,
    ("scPRAM", "T1"): ScPRAM,
    ("scPRAM", "T5c"): ScPRAM,
    # F-06a: the split-dataset path. StateC1 let the held unit's controls reach the fit.
    ("STATE", "T1"): StateC1Split,
    ("STATE", "T5c"): StateC5cSplit,
    ("PertAdapt", "T1"): PertAdaptC1,
    ("PertAdapt", "T5c"): PertAdaptC1,
    # F-03: the public graph adapter, not the historical local head (which returned the control
    # for every unseen target and produced action=declined on both T4 folds).
    ("PertAdapt", "T4"): PertAdaptC3,
    ("Biolord", "T2"): BiolordC1,
    # C5 (OP3 compounds) goes through the NATIVE compound runner: the RDKit chemistry vector is
    # biolord's own ordered attribute (published sci-Plex 3 setting), so one adapter covers both
    # the held-lineage and the unseen-compound split and emits ONE PROFILE PER COMPOUND.
    ("Biolord", "T5c"): BiolordC5,
    ("Biolord", "T5u"): BiolordC5,
    # Biolord on the GENETIC axis: the SAME released entry point, with the published
    # `perturbation_neighbors` gene-side vector (GO-Jaccard neighbours) as the ordered
    # attribute in place of the RDKit one -- biolord_reproducibility scripts/biolord/adamson
    # scores exactly this "unseen_single" regime. One profile per held gene.
    ("Biolord", "T3"): BiolordC3,
    ("Biolord", "T4"): BiolordC3,
    # T4p mirrors the RNA roster of the unseen-KO split (see the T4p block below): the gene-side
    # attribute is a property of the KO gene, not of the readout, so the same runner serves the
    # CITE modality unchanged. Drop this line if the protein roster is not being extended.
    ("Biolord", "T4p"): BiolordC3,
    # CellFlow and PRnet both condition on the compound's own structure, so one adapter covers the
    # held-lineage and the unseen-compound split for each.
    ("CellFlow", "T5c"): CellFlowC5,
    ("CellFlow", "T5u"): CellFlowC5,
    ("PRnet", "T5c"): PRnetC5,
    ("PRnet", "T5u"): PRnetC5,
    # T5c goes to the per-compound runner: the C1 adapter's pooled map was broadcast to every
    # compound, which is not CellOT's published operation. T1/T2 keep the pooled entry point,
    # where the treatment really is a single seen stimulus.
    ("CellOT", "T5c"): CellOTC5,
    # PerturbNet: native on the unseen-compound split (its own published experiment),
    # adapted on the held-lineage split via the model's counterfactual generate_zprime.
    # cells opened by the delegated runners; each is an N cell in the plan's table
    ("PerturbNet", "T1"): PerturbNetC1,
    ("PerturbNet", "T2"): PerturbNetC1,
    ("PertAdapt", "T3"): PertAdaptC3,
    ("scFoundation", "T3"): ScFoundationGene,
    ("scFoundation", "T4"): ScFoundationGene,
    ("CPA", "T1"): CPAC1,
    ("CPA", "T2"): CPAC1,
    ("PerturbNet", "T5u"): PerturbNetC5,
    ("PerturbNet", "T5c"): PerturbNetC5,
    # PerturbNet on the GENETIC axis: the held gene is represented by the frozen pretrained
    # GenotypeVAE applied to its released GO-annotation vector, so an unseen gene has a
    # representation without any harness-supplied gene embedding -- the authors' own
    # unseen-perturbation protocol. One profile per held gene (pred_key_is_group stays False).
    ("PerturbNet", "T3"): PerturbNetC3,
    ("PerturbNet", "T4"): PerturbNetC3,
    ("chemCPA", "T5c"): CPAchem,   # historical fingerprint->latent Ridge; NOT the native T5c cell
    # F-04: the native T5c cell. Seen compounds + held lineage -> the public categorical
    # CPA.predict path, one query per compound against the held lineage's own controls.
    ("CPA", "T5c"): CPAC1,
    # MAP (Nat Mach Intell 2026): SMILES + a frozen MAP-KG knowledge encoder. T5u is its
    # published unprofiled-drug regime; T5c has no published analogue (its OP3 cell-context
    # experiment holds drug x cell-type PAIRS). The runner's leak gate refuses T5u while the
    # released knowledge encoder was pre-trained on our held compounds -- see map_c5_runner.py.
    ("MAP", "T5c"): MAPC5,
    ("MAP", "T5u"): MAPC5,
    # perturbation-agnostic distributional comparator: defined wherever control and treated clouds exist
    # T1/T2 use the published per-condition call; the pooled diagnostic stays on the
    # unseen-entity tasks, where a perturbation-agnostic reduction is what is intended.
    ("CINEMA-OT", "T1"): CINEMAOTC1,
    ("CINEMA-OT", "T2"): CINEMAOTC1,
    ("CINEMA-OT", "T5c"): CINEMAOTC1,
    ("CINEMA-OT", "T4"): CINEMAOT,
    # in-process, CPU only: regresses the per-perturbation shift on a control-only-PCA gene embedding
    ("linear-shift-KOemb", "T3"): LinearShiftKOEmb,
    # T4p: the same roster as the RNA modality of the unseen-KO split
    ("PertAdapt", "T4p"): PertAdapt,
    ("CINEMA-OT", "T4p"): CINEMAOT,
    ("scGPT", "T4p"): ScGPT,
    ("scFoundation", "T4p"): ScFoundation,
    ("GEARS", "T4p"): GEARS,
    ("AttentionPert", "T4p"): AttentionPert,
    ("STATE", "T4p"): STATE,
    ("scGen", "T4p"): ScGen,
    ("CPA", "T4p"): CPA,
    ("CellOT", "T4p"): CellOTC1,
    ("scPRAM", "T4p"): ScPRAM,
    ("linear-shift-KOemb", "T4p"): LinearShiftKOEmb,
    # T5u: the disclosed fingerprint extension (already used for scGen and STATE) applied to the
    # remaining models, so the unseen-compound column is not confined to the chemistry family
    ("scGPT", "T5u"): _fp(ScGPTC1),
    ("scFoundation", "T5u"): _fp(ScFoundationC1),
    ("CellOT", "T5u"): _fp(
        CellOTC1
    ),  # Biolord T5u is native (see above), not _fp-wrapped
    ("scPRAM", "T5u"): _fp(ScPRAM),
    ("SCREEN", "T5u"): _fp(ScreenC1),
    ("PertAdapt", "T5u"): _fp(PertAdaptC1),
    # CellFlow on the non-compound splits. Held GROUP (T1/T2) -> the stimulus is a single
    # categorical and the held unit is a split_covariate; held PERTURBATION (T3/T4/T4p) -> the
    # gene enters as a perturbation_covariate_rep, the same construct as the unseen drug.
    ("CellFlow", "T1"): CellFlowC1,
    ("CellFlow", "T2"): CellFlowC1,
    ("CellFlow", "T3"): CellFlowGene,
    ("CellFlow", "T4"): CellFlowGene,
    # The remaining unseen-KO cells of the reported panel. Their deposited bundles exist but carry
    # the B2/B3 control-fallback rows, so they are re-run under the corrected code through the same
    # gene-side runners that produced them.
    ("scGPT", "T4"): ScGPT,
    ("GEARS", "T4"): GEARS,
    ("AttentionPert", "T4"): AttentionPert,
    ("CellFlow", "T4p"): CellFlowGene,
    # R2-1 compound-conditioned foundation heads. Registered under their OWN model key so
    # nothing that already resolves ("scGPT","T5u") / ("scGPT","T5c") changes; the adapter's
    # .name stays "scGPT", so gating, the registry and the census see the same model.
    ("scGPT-fcond", "T5u"): ScGPTC5Cond,
    ("scGPT-fcond", "T5c"): ScGPTC5Cond,
    # Same construction for scFoundation (scfoundation_c5cond_runner.py). T5c_cond is the
    # SAME split as T5c; it carries a `dataset` key so its deposited bundle sits BESIDE --
    # never on top of -- the C1-adapter T5c bundles, which are a different evaluation.
    ("scFoundation-fcond", "T5u"): ScFoundationC5Cond,
    # same key as scGPT-fcond: the two foundation models are treated identically on this cell.
    ("scFoundation-fcond", "T5c"): ScFoundationC5Cond,
}


def units_and_spec(task):
    """-> (loader'd CellSet, [(unit_label, SplitSpec)], programs, run_job kwargs)"""
    if task == "T1":
        from ivcbench.data.loaders import kang

        cs = kang.load()
        lins = sorted(set(cs.obs["cell_type_coarse"]))
        return cs, [(l, c1.coarse_loct(l)) for l in lins], None, {}
    if task == "T2":
        from ivcbench.data.loaders import soskic

        cs = soskic.load()
        ds = sorted(set(cs.obs["donor_id"]))
        return (
            cs,
            [(d, c2.donor_lodo(d)) for d in ds],
            None,
            {"response_gene_fn": c2.response_gene_idx},
        )
    if task == "T4p":
        # Frangieh's protein (CITE) modality of the unseen-KO split. It was run at submission for six
        # entries but never deposited: the bundle name carried no modality key, so a protein dump
        # would have overwritten the RNA bundle the census reads. Passing the modality as the dataset
        # key keeps both, and leaves every existing RNA bundle untouched.
        from ivcbench.data.loaders.frangieh import load

        cs = load(modality="protein")
        g = cs.uns["genes_perturbed"]
        out = []
        for frac, lbl in [(0.25, "25"), (0.50, "50")]:
            held = c4.held_ko_fraction(g, frac, seed=0)
            out.append((lbl, c4.modality_lo_ko(held, lbl)))
        return cs, out, None, {"exclude_from_spec": True, "dataset": "frangieh_protein"}
    if task == "T4":
        from ivcbench.data.loaders.frangieh import load

        cs = load(modality="rna")
        g = cs.uns["genes_perturbed"]
        out = []
        for frac, lbl in [(0.25, "25"), (0.50, "50")]:
            held = c4.held_ko_fraction(g, frac, seed=0)
            out.append((lbl, c4.modality_lo_ko(held, lbl)))
        return cs, out, None, {"exclude_from_spec": True}
    if task == "T3":
        # the unseen-gene cell spans five CRISPR datasets, each its own CellSet; the held genes are
        # the same 10% fraction the census uses, drawn per dataset with seed 0
        from ivcbench.data.loaders import chen, mccutcheon, schmidt, shifrut
        from ivcbench.clusters import c3

        srcs = [
            ("chen", lambda: chen.load()),
            ("mccutcheon_CRISPRa", lambda: mccutcheon.load(modality="CRISPRa")),
            ("mccutcheon_CRISPRi", lambda: mccutcheon.load(modality="CRISPRi")),
            ("schmidt", lambda: schmidt.load()),
            ("shifrut", lambda: shifrut.load()),
        ]
        return "MULTI", [(n, f) for n, f in srcs], None, {"c3": True}
    if task == "T5u":
        from ivcbench.data.loaders import op3 as op3mod
        from ivcbench.clusters.c5 import C5_PROGRAMS
        from ivcbench.clusters.spec import (
            _c5_held_compounds,
        )  # the deposited seed-0 20% holdout
        from ivcbench.clusters import c5 as _c5

        cs = op3mod.load()
        held = _c5_held_compounds(cs)
        return (
            cs,
            [("compounds", _c5.global_compound_holdout(held))],
            dict(C5_PROGRAMS),
            {},
        )
    if task == "T5c_cond":
        # The held-lineage compound split run through a compound-conditioned head. Identical
        # CellSet, units and SplitSpec as T5c -- only the deposited bundle filename differs,
        # via the `dataset` key, so the C1-adapter T5c deposit is never overwritten.
        cs, units, programs, kw = units_and_spec("T5c")
        return cs, units, programs, dict(kw, dataset="op3_compound_conditioned")
    if task == "T5c":
        from ivcbench.data.loaders import op3 as op3mod
        from ivcbench.clusters.c5 import C5_PROGRAMS

        cs = op3mod.load()
        lins = sorted(set(cs.obs["cell_type_coarse"]))
        return cs, [(l, c5.cross_celltype_loct(l)) for l in lins], dict(C5_PROGRAMS), {}
    raise SystemExit(f"unknown task {task}")


def _declined_fraction(result: dict):
    """Share of held strata whose predicted profile equals the control mean (to 1e-5), or None.

    Reads the bundle the run just deposited rather than the model object, so it applies uniformly to
    every adapter regardless of how its runner declines a label.
    """
    import numpy as np

    path = result.get("pred_bundle") or result.get("bundle_path")
    if not path or not Path(path).exists():
        return None
    try:
        z = np.load(path, allow_pickle=True)
        if "pred_means" not in z.files:
            return None
        pred = np.atleast_2d(np.asarray(z["pred_means"], dtype=float))
        ctrl = np.asarray(z["control_mean"], dtype=float).ravel()
        if pred.shape[0] < 2:
            return None
        return float(np.mean(np.abs(pred - ctrl[None, :]).max(axis=1) <= 1e-5))
    except Exception:  # a guard must never break the sweep
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", nargs=2, type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cls = ADAPTER.get((a.model, a.task))
    if cls is None:
        raise SystemExit(f"no adapter registered for ({a.model}, {a.task})")

    cs, units, programs, kw = units_and_spec(a.task)
    if a.model.endswith("-fcond") and not kw.get("dataset"):
        # the compound-conditioned head is a SECOND entry for the same model on the same
        # split; key its bundle so it cannot overwrite the deposited C1-adapter bundle.
        # `not kw.get('dataset')` so a task branch that already supplies its own key wins.
        kw = dict(kw, dataset="op3_fcond")
    if a.chunk:
        i, n = a.chunk
        units = units[i::n]
    if a.smoke:
        units = units[:1]
    print(
        f"[{a.model} @ {a.task}] {len(units)} units; adapter={cls.__name__}", flush=True
    )

    rows, out = [], Path(a.out)
    for lbl, spec in units:
        t0 = time.time()
        if kw.get("c3"):
            # one CellSet per dataset; the held genes are drawn inside the loop
            from ivcbench.clusters import c3 as _c3

            cs = spec()  # `spec` is the loader thunk here
            held = _c3.held_gene_fraction(cs.uns["genes_perturbed"], 0.10, seed=0)
            spec = _c3.true_lo_gene(held, "10")
        ad = cls()
        if a.gpu is not None:
            ad.cuda_device = str(a.gpu)
        rk = {"immune_programs": programs} if programs else {}
        if kw.get("response_gene_fn") is not None:
            rk["response_gene_fn"] = kw["response_gene_fn"]
        if kw.get("exclude_from_spec") or kw.get("c3"):
            rk["exclude_genes"] = list(spec.held_values)
        if kw.get("dataset"):
            rk["dataset"] = kw[
                "dataset"
            ]  # keys the bundle filename, so RNA and protein coexist
        elif kw.get("c3"):
            # T3 runs one CellSet PER DATASET but every unit carries the same split name, so without a
            # dataset key all five bundles collide on one filename and four are silently overwritten.
            # The unit label IS the dataset, and this is the same mechanism the T4p branch already uses
            # to keep the RNA and protein readouts apart.
            rk["dataset"] = str(lbl)
        try:
            # A model reaches ADAPTER only once the runner for that entry point exists, so an
            # ADAPTED registry cell here IS implemented; without this flag gating.decide()
            # would SKIP it. Adapted cells still fail headline_eligible(), so they are
            # reported separately and never enter the ranking.
            rk["adapted_implemented"] = True
            # NOTE (gap-fill): run_job() has no `side_info` parameter -- it reads cs.side_info
            # itself (run.py:51 `adapter.fit(cs, split, side_info=cs.side_info)`), and CellSet
            # always carries the field (schema.py:44, default_factory=dict). Passing it here
            # raised TypeError for EVERY T3 unit of EVERY model, so the whole T3 column of this
            # driver returned action=failed. Dropping the kwarg restores the identical call.
            r = run_job(cs, spec, ad, seed=0, **rk)
        except Exception as e:  # noqa: BLE001
            # a unit that dies (diverged model -> NaN, OOM, timeout) must be recorded and skipped,
            # never allowed to abort the remaining units of the chunk
            r = {
                "baseline": a.model,
                "split": getattr(spec, "name", str(lbl)),
                "action": "failed",
                "ran": False,
                "error": f"{type(e).__name__}: {e}",
            }
        # A cell whose predictions are mostly the control mean scores as the ctrl-pred floor while
        # looking like a real result. This has now happened three times in this revision (scGPT/
        # scFoundation on the compound cell-context split, PertAdapt on unseen-gene, CellFlow on the
        # protein readout), so refuse the score instead of recording it. `declined_fraction` is the
        # share of held strata whose predicted profile is the control mean to within 1e-5.
        if r.get("ran"):
            _bad = _declined_fraction(r)
            if _bad is not None and _bad > 0.5:
                r = dict(
                    r,
                    action="declined",
                    ran=False,
                    pearson_delta=None,
                    reason=(
                        f"{_bad:.0%} of held strata were returned as the control mean;"
                        " the score would be the ctrl-pred floor, not a prediction"
                    ),
                )
        rows.append(
            {
                k: r.get(k)
                for k in [
                    "baseline",
                    "family",
                    "split",
                    "action",
                    "ran",
                    "leak_free",
                    "n_train",
                    "n_test",
                    "n_test_strata",
                    "pearson_delta",
                    "pearson_delta_lo",
                    "pearson_delta_hi",
                    "e_distance",
                    "error",
                    "reason",
                ]
            }
            | {"unit": str(lbl)}
        )
        pd.DataFrame(rows).to_csv(out, index=False)
        print(
            f"  {str(lbl)[:16]:16s} {time.time()-t0:6.0f}s ran={r.get('ran')} "
            f"leak_free={r.get('leak_free')} pearson_delta={r.get('pearson_delta')}",
            flush=True,
        )
    print(f"[done] {len(rows)} units -> {out}", flush=True)


if __name__ == "__main__":
    main()
