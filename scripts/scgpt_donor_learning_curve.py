#!/usr/bin/env python
"""scGPT donor-transfer LEARNING CURVE on Soskic CD4 activation (C2).

Answers Reviewer 2 comment 3. The deposited donor-axis learning curve is CellOT's
(`scripts/cellot_donor_learning_curve.py`): as training donors grow 8 -> 96 the simple floor rises
0.214 -> 0.273 while CellOT DECLINES 0.430 -> 0.359. The reviewer objects that a model that gets
WORSE with more data looks under-trained, so the conclusion may be an artefact of the training
regime rather than a property of the donor axis. The decisive control is the same curve for a
PRETRAINED model FINE-TUNED END-TO-END. scGPT is that model on this axis: it enters from the
scGPT_human checkpoint and the TransformerGenerator is fine-tuned end to end (Supplementary
Table S15), and it is already deposited at the FULL donor count (mean Pearson-delta 0.250 against a
binding cell-mean floor of 0.260). This script produces the missing curve.

MIRRORS `scripts/cellot_donor_learning_curve.py` EXACTLY where comparability depends on it:
  * data                load_soskic_donor(cap) from scripts/c2_soskic_donor.py, --cap 300
  * eval donors         np.random.default_rng(20240607).choice(all_donors, size=10, replace=False)
  * train pool          all donors MINUS those 10 (96 donors) -- see the note on --n-eval below
  * k-subset draw       np.random.default_rng(7919 * seed + 31 * k).choice(train_pool, size=k, ...)
  * split               build_subset_split(), copied verbatim from the reference
  * metric panel        response_gene_idx(cs, sp.train_idx) -- refit inside each subset fold
  * scoring             pearson_delta(...)['macro'], e_distance on the train-fold PCA basis,
                        AUCell program-delta MAE
  * baselines           CellMean and DonorShift fit on the SAME subset split; primary = the better
  * schema              one row per (k, seed, eval_donor, metric); `cellot_score` -> `scgpt_score`
  * resume              --skip-existing

WHAT NECESSARILY DIFFERS. CellOT trains ONE transport per (k, seed) and reuses it for every eval
donor because the transport is donor-agnostic. scGPT runs through the ScGPTC1 SubprocessAdapter
(`ivcbench.baselines.heavy.ScGPTC1` -> `model_runners/scgpt_c1_runner.py`), which fits inside a
subprocess per split, so the fine-tune is repeated for every (k, seed, eval_donor). The training
payload the runner receives is in fact IDENTICAL across the eval donors of one (k, seed) cell -- only
the inference input (the held donor's own 0h cells) changes -- so the repetition is pure overhead,
but removing it would mean editing the deposited runner and the curve would no longer be produced by
the same code path as the deposited scGPT T2 cell. The published harness path is kept.

--eval-chunk I N shards the eval donors of a grid point across GPUs; it changes nothing about the
splits, only which of them one process scores. Each shard writes its own CSV, concatenated afterwards.

--n-eval < 10 scores only a subset of the fixed eval donors. The TRAIN POOL is still all donors minus
the full 10, so the k-subsets, the training payloads and hence the fine-tuned models are bit-for-bit
what a 10-donor run would have produced; only the number of donors scored against them falls. The
subset is the first --n-eval elements OF THE DRAW (pre-sort), i.e. a seed-fixed random subset of the
CellOT eval set, never a hand-picked one.

The fine-tune configuration is pinned to the values the deposited scGPT T2 run used (epochs 10,
seqlen 2048, max 8000 stimulated training cells) and is set explicitly on the child environment so a
stray shell variable cannot silently shrink it -- an under-trained curve is the very artefact the
reviewer suspects.

MEASURED COST (idle NVIDIA L40, deposited hyperparameters, one (k, seed, eval_donor) unit):
  k=8   269 s (mean of 10 units, 256-287 s) -- 4,800 stimulated training cells -> 6,000 steps
  k=96  425 s -- the runner's 8,000-stimulated-cell cap binds from k=14 up, so k=16/32/64/96 all
        cost the same 425 s; only k=8 is cheaper
  peak GPU memory 3,675 MiB (16 MiB idle baseline); one-time CellSet load ~145 s per process.
  Whole curve at n_eval=10, grid [8,16,32,64,96]: 50 units = 5.5 GPU-hours per seed.

ENVIRONMENT CAVEAT. The deposited CellOT curve ran under the `cellot` conda env (py3.9 / sklearn
1.1.1 / numpy 1.19.5); this script runs under ivcbench/.venv (py3.10 / sklearn 1.7.2 / numpy 2.2.6)
because ScGPTC1 shells out to the `scgpt` env for the model itself. pearson_delta -- the metric the
paper's claim and the deposited figure use -- and aucell_delta_score are bit-identical across the two
(verified: all ten CellMean/DonorShift baseline scores at k=8, seed=0 match the CellOT curve to four
decimals, donor-shift-primary D424 included). e_distance is NOT bit-comparable, because sklearn's PCA
auto-solver changed between those versions (cell-mean e_distance for D483 at k=8: 5.2014 here vs
5.1560 in the CellOT file, ~0.9%). Compare the two curves on pearson_delta.

REQUIRES $IVCBENCH_SCGPT_MODEL_DIR -> the scGPT_human checkpoint directory (best_model.pt,
config.json, vocab.json), the same one the deposited scGPT runs used.

DEVICE: --gpu N pins the child subprocess (adapter.cuda_device -> CUDA_VISIBLE_DEVICES).
"""
from __future__ import annotations
import sys, os, time, argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ivcbench.splits.builder import Split
from ivcbench.metrics.response import pearson_delta
from ivcbench.metrics.distribution import e_distance
from ivcbench.baselines.simple import CellMean, DonorShift
from ivcbench.baselines.heavy import ScGPTC1

from c2_soskic_donor import (
    load_soskic_donor,
    lodo_spec,
    e_distance_basis,
    response_gene_idx,
    program_delta_mae,
)

N_EVAL_CANONICAL = (
    10  # the CellOT curve's eval-set size; the draw is always made at this size
)


def build_subset_split(cs, eval_donor, train_donors):
    """Leave-one-donor-out split for `eval_donor`, but with train_idx RESTRICTED to `train_donors`.

    Copied verbatim from scripts/cellot_donor_learning_curve.py so both curves see the same splits.

    test = eval_donor 16h cells; inference input = eval_donor 0h cells; train = (0h+16h) cells of the
    given train_donors ONLY. Built directly (not via build_split) so the training pool is the subset.
    The held eval donor is never in train_donors, so the split stays leak-free by construction.
    """
    obs = cs.obs
    donor = obs["donor_id"].to_numpy()
    is_ctrl = obs["is_control"].to_numpy().astype(bool)
    in_eval = donor == eval_donor
    in_train_pool = np.isin(donor, np.asarray(train_donors, dtype=object))
    assert eval_donor not in set(train_donors), "eval donor leaked into train pool"

    spec = lodo_spec(eval_donor)
    train_idx = np.where(in_train_pool)[0]
    test_idx = np.where(in_eval & ~is_ctrl)[0]
    inference_input_idx = np.where(in_eval & is_ctrl)[0]
    test_strata = np.array(
        [spec.stratum_key(obs.iloc[i]) for i in test_idx], dtype=object
    )
    return Split(spec, train_idx, test_idx, inference_input_idx, test_strata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--grid",
        type=int,
        nargs="*",
        default=[8, 16, 32, 64, 96],
        help="numbers of TRAINING donors to sweep",
    )
    ap.add_argument(
        "--n-eval",
        type=int,
        default=10,
        help="how many of the FIXED 10 eval donors to score (train pool is unchanged)",
    )
    ap.add_argument(
        "--eval-chunk",
        type=int,
        nargs=2,
        default=None,
        metavar=("I", "N"),
        help=(
            "score only eval_donors[I::N] -- shard one grid point across GPUs. Each "
            "shard MUST use its own --out/--timing-out; concatenate the shard CSVs "
            "afterwards (same convention as the CellOT curve's per-seed files)."
        ),
    )
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument(
        "--epochs", type=int, default=10, help="scGPT fine-tune epochs (deposited: 10)"
    )
    ap.add_argument(
        "--seqlen", type=int, default=2048, help="scGPT gene panel (deposited: 2048)"
    )
    ap.add_argument(
        "--max-cells",
        type=int,
        default=8000,
        help="cap on stimulated training cells inside the runner (deposited: 8000)",
    )
    ap.add_argument(
        "--gpu", default=None, help="physical GPU id to pin the fine-tune subprocess to"
    )
    ap.add_argument(
        "--out", default=str(ROOT / "results/newdata/scgpt_donor_learning_curve.csv")
    )
    ap.add_argument(
        "--timing-out",
        default=str(ROOT / "results/newdata/scgpt_donor_learning_curve_timing.json"),
    )
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    # Pin the fine-tune configuration to the deposited scGPT T2 settings (see module docstring).
    os.environ["IVCBENCH_SCGPT_EPOCHS"] = str(args.epochs)
    os.environ["IVCBENCH_SCGPT_SEQLEN"] = str(args.seqlen)
    os.environ["IVCBENCH_SCGPT_MAXCELLS"] = str(args.max_cells)

    cs = load_soskic_donor(args.cap)
    all_donors = sorted(cs.obs.donor_id.unique())
    n_total = len(all_donors)

    # FIXED eval donors: the SAME deterministic, seed-independent draw the CellOT curve makes. The
    # draw is always at the canonical size 10 so the TRAIN POOL is identical no matter how many eval
    # donors are scored; --n-eval only truncates the scored set (first n of the draw, then sorted).
    eval_rng = np.random.default_rng(20240607)
    drawn = eval_rng.choice(
        all_donors, size=max(args.n_eval, N_EVAL_CANONICAL), replace=False
    ).tolist()
    held_out = sorted(drawn)  # never trained on, regardless of --n-eval
    eval_donors = sorted(drawn[: args.n_eval])  # scored this run
    if args.eval_chunk:
        _i, _n = args.eval_chunk
        eval_donors = eval_donors[_i::_n]
    train_pool = [d for d in all_donors if d not in set(held_out)]
    max_k = len(train_pool)
    grid = sorted({min(k, max_k) for k in args.grid})
    print(
        f"[soskic] {cs.X.shape[0]} cells x {cs.X.shape[1]} genes;"
        f" donors_total={n_total}; held_out={len(held_out)};"
        f" n_eval_scored={len(eval_donors)}; train_pool={max_k}; grid={grid};"
        f" seeds={args.seeds}",
        flush=True,
    )
    print(f"[held_out_donors] {held_out}", flush=True)
    print(f"[eval_donors]     {eval_donors}", flush=True)
    print(
        f"[scgpt] epochs={args.epochs} seqlen={args.seqlen} max_cells={args.max_cells} "
        f"gpu={args.gpu}",
        flush=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, timing = [], []
    done = set()  # (grid_size, seed, eval_donor)
    if args.skip_existing and out_path.exists():
        old = pd.read_csv(out_path)
        rows = old.to_dict("records")
        for (gs, sd, ed), sub in old.groupby(["n_train_donors", "seed", "eval_donor"]):
            if len(sub) >= 3:  # all three metrics written for this donor
                done.add((int(gs), int(sd), str(ed)))
        print(
            f"resume: {len(old)} rows; skipping {len(done)} complete"
            " (grid,seed,eval_donor) units",
            flush=True,
        )
    if Path(args.timing_out).exists():
        try:
            timing = json.load(open(args.timing_out))
        except Exception:
            timing = []

    for seed in args.seeds:
        for k in grid:
            # random k-subset of the train pool -- the SAME draw as the CellOT curve makes
            pick_rng = np.random.default_rng(7919 * seed + 31 * k)
            train_donors = sorted(
                pick_rng.choice(train_pool, size=k, replace=False).tolist()
            )
            assert not (
                set(train_donors) & set(held_out)
            ), "eval donor leaked into training subset"
            t_cell = time.time()

            for j, ed in enumerate(eval_donors):
                if (k, seed, str(ed)) in done:
                    print(
                        f"  skip grid={k} seed={seed} eval={ed} (already complete)",
                        flush=True,
                    )
                    continue
                sp = build_subset_split(cs, ed, train_donors)
                # leak-free guarantee: eval donor cells absent from train_idx
                assert not np.isin(
                    sp.train_idx, np.concatenate([sp.test_idx, sp.inference_input_idx])
                ).any()
                test_X = cs.X[sp.test_idx]
                test_strata = sp.test_strata
                ctrl_idx = sp.inference_input_idx
                ctrl_X = cs.X[ctrl_idx]
                ctrl_mean = ctrl_X.mean(0)
                ctrl_strat_str = (
                    cs.obs.iloc[ctrl_idx]["cell_type_coarse"].astype(str).to_numpy()
                )
                rg = response_gene_idx(cs, sp.train_idx)
                ed_basis = e_distance_basis(cs, sp.train_idx)

                # ---- scGPT: end-to-end fine-tune from the pretrained checkpoint, one per split ----
                tA = time.time()
                adapter = ScGPTC1()
                if args.gpu is not None:
                    adapter.cuda_device = str(args.gpu)
                adapter.fit(cs, sp, side_info=cs.side_info)
                pred = adapter.predict(cs, sp, side_info=cs.side_info)
                dt = time.time() - tA
                sg_pe = float(
                    pearson_delta(pred.pred_cells, test_X, ctrl_mean, test_strata, rg)[
                        "macro"
                    ]
                )
                sg_ed = float(
                    e_distance(pred.pred_cells, test_X, test_strata, fit_on=ed_basis)[
                        "macro"
                    ]
                )
                sg_au = program_delta_mae(
                    pred.pred_cells, test_X, ctrl_X, test_strata, ctrl_strat_str, cs
                )["aucell_delta_score"]

                # ---- baselines on the SAME subset split ----
                bp = {}
                for B in (CellMean, DonorShift):
                    b = B()
                    b.fit(cs, sp, side_info=cs.side_info)
                    bp[b.name] = b.predict(cs, sp, side_info=cs.side_info)
                b_pe = {
                    n: float(
                        pearson_delta(
                            bp[n].pred_cells,
                            test_X,
                            bp[n].control_mean,
                            test_strata,
                            rg,
                        )["macro"]
                    )
                    for n in bp
                }
                b_ed = {
                    n: float(
                        e_distance(
                            bp[n].pred_cells, test_X, test_strata, fit_on=ed_basis
                        )["macro"]
                    )
                    for n in bp
                }
                b_au = {
                    n: program_delta_mae(
                        bp[n].pred_cells,
                        test_X,
                        ctrl_X,
                        test_strata,
                        ctrl_strat_str,
                        cs,
                    )["aucell_delta_score"]
                    for n in bp
                }
                prim_pe = max(b_pe, key=b_pe.get)
                prim_ed = min(b_ed, key=b_ed.get)
                prim_au = max(
                    b_au, key=lambda n: (b_au[n] if b_au[n] == b_au[n] else -1e9)
                )

                for metric, cscore, pname, pscore in [
                    ("pearson_delta", sg_pe, prim_pe, b_pe[prim_pe]),
                    ("e_distance", sg_ed, prim_ed, b_ed[prim_ed]),
                    ("aucell_delta_score", sg_au, prim_au, b_au[prim_au]),
                ]:
                    if metric == "e_distance":
                        delta = (
                            pscore - cscore
                        )  # lower-better -> positive favours scGPT
                    else:
                        delta = cscore - pscore
                    ok = (cscore == cscore) and (pscore == pscore)
                    rows.append(
                        dict(
                            n_train_donors=int(k),
                            seed=int(seed),
                            eval_donor=str(ed),
                            metric=metric,
                            scgpt_score=(round(cscore, 4) if cscore == cscore else ""),
                            primary_baseline=pname,
                            baseline_score=(
                                round(pscore, 4) if pscore == pscore else ""
                            ),
                            delta_vs_primary=(round(delta, 4) if ok else ""),
                            best_mmd="",  # CellOT-only transport diagnostic; n.a. for scGPT
                            n_train_cells=int(len(sp.train_idx)),
                            n_test=int(len(sp.test_idx)),
                            n_ctrl=int(len(ctrl_idx)),
                            n_response_genes=int(len(rg)),
                            fit_sec=round(dt, 1),
                        )
                    )
                timing.append(
                    dict(
                        n_train_donors=int(k),
                        seed=int(seed),
                        eval_donor=str(ed),
                        sec=round(dt, 1),
                        n_train_cells=int(len(sp.train_idx)),
                    )
                )
                print(
                    f"  [grid={k} seed={seed} eval={ed} {j+1}/{len(eval_donors)}] "
                    f"{dt:.0f}s"
                    f" pearsonD={sg_pe:.4f} eDist={sg_ed:.4f} aucell={sg_au:.4f} "
                    f"(vs {prim_pe} pe={b_pe[prim_pe]:.4f} margin={sg_pe-b_pe[prim_pe]:+.4f})",
                    flush=True,
                )
                # checkpoint after every eval donor so the job is fully resumable
                pd.DataFrame(rows).to_csv(out_path, index=False)
                json.dump(timing, open(args.timing_out, "w"), indent=2)
            print(
                f"[cell grid={k} seed={seed}] done in {time.time()-t_cell:.0f}s",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nWROTE {out_path} ({len(df)} rows)", flush=True)
    pe = df[(df.metric == "pearson_delta") & (df.delta_vs_primary != "")].copy()
    if len(pe):
        pe["delta_vs_primary"] = pe["delta_vs_primary"].astype(float)
        pe["scgpt_score"] = pe["scgpt_score"].astype(float)
        pe["baseline_score"] = pe["baseline_score"].astype(float)
        g = pe.groupby("n_train_donors").agg(
            scgpt=("scgpt_score", "mean"),
            floor=("baseline_score", "mean"),
            delta=("delta_vs_primary", "mean"),
            n=("delta_vs_primary", "size"),
        )
        print("\nLearning curve (pearson_delta, mean over eval donors x seeds):")
        print(g.to_string())


if __name__ == "__main__":
    main()
