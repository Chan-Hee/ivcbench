#!/usr/bin/env python
"""Chart: does scGPT's REPORTED score improve if it trains longer?

The deposited scGPT runs all use 10 fine-tune epochs -- a number this benchmark chose, not one
scGPT publishes. scGPT's own perturbation tutorial uses 15 epochs with validation-based
best-checkpoint selection and a StepLR decay; the runner cited as the recipe defaults to 3. So a
reviewer can reasonably ask whether the deposited scGPT cells are simply under-trained.

This plots the answer on the metric the paper reports (macro Pearson-delta, `pearson_delta`) against
the same donor-matched floor the census uses, using `scripts/scgpt_donor_learning_curve.py
--eval-epochs`: ONE 20-epoch fine-tune per donor, scored at each epoch checkpoint. Donor, seed,
subset draw and batch order are therefore identical across the x-axis -- the only thing that varies
is how long it trained.

Panel A  reported score vs epochs, one line per donor + the mean, against each donor's own floor.
Panel B  mean training MSE vs epoch (from $IVCBENCH_SCGPT_TRACE), i.e. whether the loss has
         stopped descending by epoch 10 at all.

Usage:
    python scripts/plot_scgpt_epoch_curve.py --csv results/newdata/scgpt_epcurve_s*.csv \
        --trace results/newdata/scgpt_epcurve_trace_s*.jsonl --out results/newdata/scgpt_epoch_curve.png
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEPOSITED_EPOCHS = 10      # what every deposited scGPT cell used
PUBLISHED_EPOCHS = 15      # scGPT's own perturbation tutorial
METRIC = "pearson_delta"   # the metric the census reports

# the census T2 cell, for orientation (Supplementary cross_cluster_headline.csv)
CENSUS_SCGPT_T2, CENSUS_FLOOR_T2 = 0.2501, 0.2598


def _load(patterns):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit(f"no CSV matched {patterns}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.metric == METRIC].copy()
    if "epochs_at_eval" not in df.columns:
        raise SystemExit("CSV has no epochs_at_eval column -- rerun with --eval-epochs")
    for c in ("scgpt_score", "baseline_score", "epochs_at_eval"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["scgpt_score", "epochs_at_eval"]), files


def _load_trace(patterns):
    recs = []
    for p in patterns or []:
        for f in sorted(glob.glob(p)):
            with open(f) as fh:
                for ln in fh:
                    ln = ln.strip()
                    if ln:
                        recs.append(json.loads(ln))
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--trace", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train-donors", type=int, default=None,
                    help="keep only this donor-pool size (default: require the CSVs to hold exactly one)")
    ap.add_argument("--seed", type=int, default=None,
                    help="keep only this seed (default: require the CSVs to hold exactly one)")
    ap.add_argument("--title", default="scGPT donor transfer (Soskic, 96 training donors)")
    a = ap.parse_args()

    df, files = _load(a.csv)
    # Averaging two donor-pool sizes or two seeds into one line would be a silent category error.
    for col, want in (("n_train_donors", a.n_train_donors), ("seed", a.seed)):
        if want is not None:
            df = df[df[col].astype(int) == int(want)]
        elif df[col].nunique() > 1:
            raise SystemExit(
                f"the CSVs hold {df[col].nunique()} distinct {col} values "
                f"({sorted(df[col].unique())}); pass --{col.replace('_', '-')} to pick one"
            )
    if not len(df):
        raise SystemExit("no rows left after filtering")
    donors = sorted(df.eval_donor.unique())
    eps = sorted(df.epochs_at_eval.unique())
    print(f"{len(files)} file(s); donors={donors}; epochs={eps}")

    fig = plt.figure(figsize=(11.0, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.26)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    cmap = plt.get_cmap("tab10")

    # ---- Panel A: reported score vs epochs -------------------------------
    for i, d in enumerate(donors):
        sub = df[df.eval_donor == d].sort_values("epochs_at_eval")
        c = cmap(i % 10)
        axA.plot(sub.epochs_at_eval, sub.scgpt_score, "-o", ms=3.4, lw=1.2, color=c, alpha=0.85, label=d)
        fl = sub.baseline_score.dropna()
        if len(fl):
            axA.axhline(float(fl.iloc[0]), color=c, ls=":", lw=0.8, alpha=0.5)

    piv = df.pivot_table(index="epochs_at_eval", columns="eval_donor", values="scgpt_score")
    piv = piv.dropna(axis=0, how="any")           # mean only over epochs every donor reached
    if len(piv):
        axA.plot(piv.index, piv.mean(1), "-", lw=2.8, color="0.12", zorder=5, label="mean")
        axA.plot(piv.index, piv.mean(1), "o", ms=5.5, color="0.12", zorder=6)
    fl_mean = df.groupby("eval_donor").baseline_score.first().mean()
    axA.axhline(fl_mean, color="0.12", ls="--", lw=1.8, zorder=4,
                label=f"floor, donor-matched mean ({fl_mean:.3f})")

    axA.axvline(DEPOSITED_EPOCHS, color="#b2182b", lw=1.1, alpha=0.55)
    axA.axvline(PUBLISHED_EPOCHS, color="#2166ac", lw=1.1, ls="--", alpha=0.55)
    ymin, ymax = axA.get_ylim()
    axA.text(DEPOSITED_EPOCHS, ymax, " deposited (10)", color="#b2182b", fontsize=8,
             va="top", ha="left", rotation=90)
    axA.text(PUBLISHED_EPOCHS, ymax, " scGPT tutorial (15)", color="#2166ac", fontsize=8,
             va="top", ha="left", rotation=90)
    axA.set_xlabel("fine-tune epochs")
    axA.set_ylabel("macro Pearson-delta  (the reported metric)")
    axA.set_title("A  Reported score vs training length", loc="left", fontsize=10, fontweight="bold")
    axA.set_xticks(eps)
    axA.grid(alpha=0.18, lw=0.6)
    axA.legend(fontsize=7.2, ncol=2, frameon=False, loc="lower right")

    # ---- Panel B: training loss -----------------------------------------
    tr = _load_trace(a.trace)
    if len(tr):
        # The trace is append-only by design ("a killed run still leaves a curve"), so a relaunched
        # unit appends a SECOND set of rows under the same tag. Keep the last write per (tag, epoch),
        # and drop rows with no tag -- the unseen-gene runner writes the same file without one.
        if "tag" not in tr.columns:
            tr["tag"] = "(untagged)"
        tr = tr.dropna(subset=["tag"]).drop_duplicates(subset=["tag", "epoch"], keep="last")
        for i, (tag, sub) in enumerate(tr.groupby("tag")):
            sub = sub.sort_values("epoch")
            axB.plot(sub.epoch, sub.train_mse, "-", lw=1.1, color=cmap(i % 10), alpha=0.85,
                     label=str(tag).split("_")[-1])
        # same guard Panel A uses: average only over epochs EVERY run reached, so the mean never
        # steps because membership changed rather than because training changed
        tpiv = tr.pivot_table(index="epoch", columns="tag", values="train_mse").dropna(axis=0, how="any")
        m = tpiv.mean(1)
        axB.plot(m.index, m.values, "-", lw=2.6, color="0.12", zorder=5, label="mean")
        axB.axvline(DEPOSITED_EPOCHS, color="#b2182b", lw=1.1, alpha=0.55)
        axB.set_xlabel("epoch")
        axB.set_ylabel("mean training MSE")
        axB.legend(fontsize=7.2, frameon=False, ncol=2)
        axB.grid(alpha=0.18, lw=0.6)
    else:
        axB.text(0.5, 0.5, "no --trace given", ha="center", va="center", transform=axB.transAxes,
                 color="0.45")
        axB.set_xticks([]); axB.set_yticks([])
    axB.set_title("B  Is the loss still descending at epoch 10?", loc="left", fontsize=10,
                  fontweight="bold")

    for ax in (axA, axB):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(a.title, y=1.0, fontsize=11)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    fig.savefig(str(Path(a.out).with_suffix(".pdf")), bbox_inches="tight")
    print("wrote", a.out)

    # ---- the numbers the chart is making, printed so they can be quoted ----
    print(f"\ncensus T2 cell: scGPT {CENSUS_SCGPT_T2} vs binding floor {CENSUS_FLOOR_T2} "
          f"(shortfall {CENSUS_SCGPT_T2 - CENSUS_FLOOR_T2:+.4f})")
    if len(piv):
        mean_by_ep = piv.mean(1)
        print(f"\n{'epochs':>7}  {'mean score':>10}  {'vs floor':>9}  {'vs epoch 10':>11}")
        base = mean_by_ep.get(DEPOSITED_EPOCHS, float("nan"))
        for e, v in mean_by_ep.items():
            print(f"{int(e):>7}  {v:>10.4f}  {v - fl_mean:>+9.4f}  {v - base:>+11.4f}")
    print("\nper donor, epoch 10 -> max:")
    for d in donors:
        sub = df[df.eval_donor == d].sort_values("epochs_at_eval")
        at10 = sub[sub.epochs_at_eval == DEPOSITED_EPOCHS].scgpt_score
        best = sub.loc[sub.scgpt_score.idxmax()]
        print(f"  {d}: ep10={float(at10.iloc[0]) if len(at10) else float('nan'):.4f}  "
              f"best={best.scgpt_score:.4f} @ep{int(best.epochs_at_eval)}  "
              f"floor={float(sub.baseline_score.dropna().iloc[0]):.4f}")


if __name__ == "__main__":
    main()
