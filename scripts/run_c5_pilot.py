#!/usr/bin/env python
"""C5 pilot driver — the "1패스" on synthetic OP3-shaped data (GPU-free).

Runs the Simple-4 baselines through both C5 splits, end to end:
  synth -> build split -> LEAK AUDIT -> applicability gating -> fit -> predict -> 4-axis metrics.
Heavy baselines (chemCPA/scGPT/CINEMA-OT/STATE) are added in Phase 1 behind the same BaselineAdapter
interface; this driver proves the surrounding machinery first.
"""
from __future__ import annotations

import pandas as pd

from ivcbench.baselines.simple import SIMPLE_BASELINES
from ivcbench.clusters import c5
from ivcbench.data.synth import make_op3_like
from ivcbench.runner.run import run_job


def main() -> pd.DataFrame:
    cs = make_op3_like(seed=0)
    program = cs.uns["immune_program"]["immunomod_moa"]
    held_compounds = cs.uns["compounds"][:3]  # Tanimoto bins computed post-hoc in Phase 1

    splits = [
        c5.cross_celltype_loct(held_lineage="NK"),
        c5.global_compound_holdout(held_compounds=held_compounds),
    ]

    rows = []
    for spec in splits:
        for B in SIMPLE_BASELINES:
            rows.append(run_job(cs, spec, B(), seed=0, immune_program_genes=program))

    df = pd.DataFrame(rows)
    cols = ["split", "baseline", "action", "headline_eligible", "leak_free",
            "n_train", "n_test", "pearson_delta", "e_distance", "aucell_program_corr"]
    out = df[cols].round(3)
    pd.set_option("display.width", 140)
    print("\n=== C5 pilot (synthetic OP3-shaped) — all splits leak-audited ===\n")
    print(out.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
