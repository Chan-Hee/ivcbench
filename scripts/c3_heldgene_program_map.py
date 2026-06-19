#!/usr/bin/env python
"""Resolve the C3 held-gene -> immune-program mapping (reviewer req 7).

For each of the five CRISPR datasets, reconstruct the held target genes (exactly as the benchmark did:
cs.uns['genes_perturbed'] -> c3.held_gene_fraction(g, frac, seed=0)) at each holdout fraction, and tag
which held genes are MEMBERS of each AUCell program (TCR_activation / IL2_STAT5 / effector_cytokine /
Treg_exhaustion / proliferation). Writes results/C3/heldgene_program_map.json with the gene-resolved
membership so we can test whether the 24/435 program recovery concentrates on on-program vs off-program
held genes.

Pure reconstruction of the existing split logic — no new model runs. CPU; loads each dataset once.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ivcbench.clusters import c3  # noqa: E402

LOADERS = {
    "shifrut":            ("ivcbench.data.loaders.shifrut", "load", {}),
    "schmidt":            ("ivcbench.data.loaders.schmidt", "load", {}),
    "mccutcheon_CRISPRi": ("ivcbench.data.loaders.mccutcheon", "load", {"modality": "CRISPRi"}),
    "mccutcheon_CRISPRa": ("ivcbench.data.loaders.mccutcheon", "load", {"modality": "CRISPRa"}),
    "chen":               ("ivcbench.data.loaders.chen", "load", {}),
}
FRACS = [(0.10, "10"), (0.25, "25"), (0.50, "50")]
PROGRAMS = c3.C3_PROGRAMS
PROG_SETS = {k: set(v) for k, v in PROGRAMS.items()}


def main():
    import importlib
    out = {}
    for ds, (mod, fn, kw) in LOADERS.items():
        try:
            loader = getattr(importlib.import_module(mod), fn)
            cs = loader(**kw)
            genes = sorted(set(cs.uns["genes_perturbed"]))
        except Exception as e:  # noqa: BLE001
            print(f"!! {ds} load failed: {type(e).__name__}: {e}", flush=True)
            out[ds] = {"error": f"{type(e).__name__}: {e}"}
            continue
        rec = {"n_perturbed_genes": len(genes), "perturbed_genes": genes, "holdouts": {}}
        # which perturbed genes are program members at all (regardless of holdout)
        rec["program_members_in_screen"] = {
            p: sorted(set(genes) & PROG_SETS[p]) for p in PROGRAMS}
        for frac, lbl in FRACS:
            held = c3.held_gene_fraction(genes, frac, seed=0)
            held_set = set(held)
            on = {p: sorted(held_set & PROG_SETS[p]) for p in PROGRAMS}
            n_on = len(set().union(*on.values())) if on else 0
            rec["holdouts"][lbl] = {
                "n_held": len(held), "held_genes": sorted(held),
                "on_program_held": on,
                "n_held_on_any_program": n_on,
                "n_held_off_all_programs": len(held) - n_on,
            }
        out[ds] = rec
        print(f"{ds}: {len(genes)} perturbed genes; program members in screen: "
              f"{ {p: len(v) for p, v in rec['program_members_in_screen'].items()} }", flush=True)
        for lbl in ("10", "25", "50"):
            h = rec["holdouts"][lbl]
            print(f"   held@{lbl}%: n={h['n_held']}, on-program={h['n_held_on_any_program']}, "
                  f"off-program={h['n_held_off_all_programs']}", flush=True)
    (ROOT / "results/C3/heldgene_program_map.json").write_text(json.dumps(out, indent=2))
    print("wrote results/C3/heldgene_program_map.json", flush=True)


if __name__ == "__main__":
    main()
