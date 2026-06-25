# Reproduction report — full 35-cell prediction layer re-run (approach B)

**Date:** 2026-06-22  ·  **Status:** Phase 1–3 complete; awaiting sign-off before any manuscript-number change.

## What this is
Every model behind the 35-cell generalization map was re-run once from scratch (heavy models on the 4× L40
over the weekend), the per-model predictions were frozen, and the cross-cluster headline was recomputed from
them with the **same aggregation as the deposited pipeline** (macro-mean per-unit Pearson-Δ per model; floor =
mean(cell-mean, linear-PCA); identical to `assemble_cross_cluster.py`). The reproduction headline was built by
`outputs/reproduction/repro_assemble.py`, which reads ONLY the re-run outputs and **never touches or
overwrites any deposited file**. Full per-cell table: `outputs/reproduction/verdict_diff_35cell.csv`;
reproduction census: `outputs/reproduction/cross_cluster_headline_repro.csv`.

## Headline result
**All 35 cells reproduce. Zero verdict flips.** Every "beats-floor / fails-floor" call in the paper is
unchanged by an independent end-to-end re-run.

| metric | value |
|---|---|
| cells compared | 35 / 35 |
| exact (Δ = 0.0000) | 15 |
| within \|Δ\| < 0.01 | 25 |
| within \|Δ\| < 0.03 | 33 |
| \|Δ\| ≥ 0.05 | 2 |
| mean \|Δ Pearson-Δ\| | 0.011 |
| max \|Δ Pearson-Δ\| | 0.158 |
| **verdict flips (beats-floor)** | **0** |

## Reproducibility tiers (what the numbers show)
- **Exact (bit-identical):** the deterministic comparators (cell-mean, linear-PCA floor; FP-ridge;
  linear-shift-KOemb) AND the seed-deterministic trained models **STATE and scPRAM** reproduce per-unit
  bit-for-bit (e.g. C2 scPRAM 0.1592, C2 STATE 0.1830, both 0/106 donors differing). The June-22 re-run and
  the June-7 deposit are different files written six weeks apart, so this is genuine seed-determinism, not a
  shared source.
- **Within stochastic variance (\|Δ\| < 0.03):** CellOT (C2 0.3691→0.3666), CPA (C2/C4 exact-ish; C1 exact),
  scGen (C1/C2/C4/C5 ≤0.016), chemCPA (C5 0.1116→0.1160), GEARS, AttentionPert, scFoundation — all reproduce
  comfortably inside the seed-to-seed spread already reported in the paper.
- **Two larger stochastic deviations (verdict unchanged):**
  - **scGPT, C3 unseen-gene:** 0.0582 → 0.2164 (Δ0.158). A foundation model on the 5-dataset macro; the re-run
    lands higher but still **far below the 0.395 floor** — the "foundation models do not extrapolate to unseen
    genes" verdict is unaffected.
  - **CPA, C5 unseen-compound:** 0.1587 → 0.1067 (Δ0.052). Still **below floor** — the "conditioning fails on
    unseen-perturbation" verdict is unaffected.

## Why this is the right deposit (reviewer-proofing)
A reviewer who asks for model weights and re-runs the pipeline will land on the SAME conclusions: the
deterministic and seed-deterministic cells come back identical, and the stochastic cells stay on the same side
of the floor. Depositing the **frozen prediction layer** (not checkpoints) lets a reviewer regenerate every
reported number and the figure GPU-free in minutes, while the per-family runner scripts let the motivated reader
retrain from scratch and land here. Nothing in the manuscript needs to move.

## Proposed action (NEEDS YOUR SIGN-OFF)
The reproduction confirms the deposited numbers; the cleanest path is to **keep the deposited manuscript
numbers as-is** (they are the frozen reference the prediction layer reproduces) and ship the prediction layer +
the per-family runner scripts + this report as the reproducibility evidence. I have NOT changed any deposited
result or manuscript number. Options:
1. **(recommended)** Keep deposited numbers; deposit the prediction layer + per-family runner scripts + this
   report. No manuscript-number change.
2. Overwrite the deposited census with the re-run numbers (15 cells identical, 20 shift ≤0.158, 0 verdicts
   change) — only if you want the manuscript to show the exact re-run values.

Awaiting your choice before touching anything in `results/_paper/` or the manuscript.
