#!/usr/bin/env python
"""Merge Soskic donor-axis shards and deposit paper-ready source files.

Inputs are the shard/raw CSVs emitted by scripts/c2_soskic_donor.py. The script is intentionally safe on
partial runs: it writes the same files with an INCOMPLETE flag so another agent can resume without guessing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DONORS = 106
PRIMARY_BASELINES = ["cell-mean", "donor-shift"]
ALL_MODELS = ["scGen", "ctrl-pred", "cell-mean", "donor-shift", "linear-PCA"]
B = 10000


def _read_inputs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for pat in paths:
        for p in sorted(ROOT.glob(pat)):
            if "smoke" in p.name:
                continue
            d = pd.read_csv(p)
            if {"donor", "model"}.issubset(d.columns):
                d["source_file"] = str(p.relative_to(ROOT))
                frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    keys = [c for c in ["donor", "model", "timepoint"] if c in df.columns]
    return df.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)


def _bootstrap(vals: np.ndarray, higher_positive: bool = True, seed: int = 0) -> dict:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan, pct_positive=np.nan, boot_pos_frac=np.nan)
    rng = np.random.default_rng(seed)
    boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(B)])
    return dict(n=int(len(vals)), mean=float(vals.mean()), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), pct_positive=float((vals > 0).mean()),
                boot_pos_frac=float((boots > 0).mean()))


def _paired_deltas(df: pd.DataFrame, metric: str, lower_better: bool = False) -> pd.DataFrame:
    rows = []
    for donor, sub in df.groupby("donor"):
        models = {str(r.model): r for r in sub.itertuples(index=False)}
        if "scGen" not in models or not all(b in models for b in PRIMARY_BASELINES):
            continue
        sc = float(getattr(models["scGen"], metric, np.nan))
        prim_vals = [(b, float(getattr(models[b], metric, np.nan))) for b in PRIMARY_BASELINES]
        prim_vals = [(b, v) for b, v in prim_vals if np.isfinite(v)]
        if not prim_vals or not np.isfinite(sc):
            continue
        primary_model, primary = (min if lower_better else max)(prim_vals, key=lambda x: x[1])
        delta = primary - sc if lower_better else sc - primary
        rows.append(dict(donor=donor, metric=metric, lower_better=lower_better,
                         scGen=sc, primary_baseline=primary, primary_model=primary_model,
                         delta_conditioned_minus_primary=delta))
    return pd.DataFrame(rows)


def _verdict(primary: dict) -> str:
    if primary["n"] < EXPECTED_DONORS:
        return "INCOMPLETE"
    if primary["lo"] > 0:
        return "conditioning exceeds baseline"
    if primary["hi"] <= 0:
        return "baseline not exceeded"
    return "competitive / interval crosses zero"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=["results/C2/soskic_donor_raw.csv",
                                                    "results/C2/soskic_donor_shard*.csv"],
                    help="Glob(s), relative to benchmark root. Smoke files are ignored.")
    args = ap.parse_args()

    df = _read_inputs(args.inputs)
    out_root = ROOT / "results"
    c2 = out_root / "C2"
    figsrc = ROOT / "figures" / "source_data"
    supp = ROOT / "supp"
    for p in [c2, figsrc, supp, ROOT / "reports"]:
        p.mkdir(parents=True, exist_ok=True)

    if df.empty:
        raise SystemExit("No Soskic donor CSV rows found. Run scripts/c2_soskic_donor.py first.")

    complete_by_donor = df.groupby("donor")["model"].apply(lambda x: set(map(str, x)))
    complete_donors = sorted(d for d, models in complete_by_donor.items() if set(ALL_MODELS).issubset(models))
    incomplete = EXPECTED_DONORS - len(complete_donors)

    axis_path = out_root / "soskic_donor_axis.csv"
    c2_axis_path = c2 / "soskic_donor_axis.csv"
    df.to_csv(axis_path, index=False)
    df.to_csv(c2_axis_path, index=False)

    forest = _paired_deltas(df[df["donor"].isin(complete_donors)], "pearson_delta", lower_better=False)
    forest.to_csv(figsrc / "soskic_donor_forest.csv", index=False)
    forest.to_csv(figsrc / "soskic_donor_forest.tsv", sep="\t", index=False)

    metric_specs = [("pearson_delta", False), ("e_distance", True)]
    if "aucell_delta_mae" in df.columns:
        metric_specs.append(("aucell_delta_mae", True))
    summaries = []
    for metric, lower in metric_specs:
        deltas = _paired_deltas(df[df["donor"].isin(complete_donors)], metric, lower_better=lower)
        s = _bootstrap(deltas["delta_conditioned_minus_primary"].to_numpy() if len(deltas) else np.array([]))
        summaries.append(dict(metric=metric, lower_better=lower, **s))
    summary = pd.DataFrame(summaries)
    summary.to_csv(c2 / "soskic_donor_bootstrap_summary.csv", index=False)

    pear = summary[summary.metric == "pearson_delta"].iloc[0].to_dict()
    verdict = _verdict(pear)
    sc_mean = forest["scGen"].mean() if len(forest) else np.nan
    prim_mean = forest["primary_baseline"].mean() if len(forest) else np.nan

    def fmt(v: float, nd: int = 3) -> str:
        return "NA" if not np.isfinite(v) else f"{v:.{nd}f}"

    table_row = (
        "| Donor | Soskic CD4 activation, leave-one-donor-out (0h→16h, 106 paired donors) "
        f"| donor (n = {len(complete_donors)}{' / 106 complete' if incomplete else ''}) "
        f"| {fmt(prim_mean)} (best of training-mean/donor shift, per donor) "
        f"| scGen {fmt(sc_mean)} | E-distance and AUCell-Δ in source data "
        f"| {fmt(pear['mean'])} [{fmt(pear['lo'])}, {fmt(pear['hi'])}] "
        f"| cluster bootstrap over donors; {pear['pct_positive'] * 100 if np.isfinite(pear['pct_positive']) else np.nan:.1f}% donors positive "
        f"| {verdict} |"
    )
    if incomplete:
        table_row = "PARTIAL - do not paste into manuscript until full LODO completes.\n\n" + table_row
    (out_root / "soskic_table6_row.md").write_text(table_row + "\n")
    (c2 / "soskic_table6_row.md").write_text(table_row + "\n")

    sign = {
        "conditioning exceeds baseline": "B_positive",
        "baseline not exceeded": "A_negative_or_consistent",
        "competitive / interval crosses zero": "mixed_interval_crosses_zero",
        "INCOMPLETE": "partial_pending",
    }[verdict]
    one_sentence = (
        f"Soskic donor LODO is {verdict}: scGen minus the pre-specified primary baseline on response-gene "
        f"Pearson-Delta is {fmt(pear['mean'])} [{fmt(pear['lo'])}, {fmt(pear['hi'])}] over "
        f"{len(complete_donors)}/106 donors ({sign})."
    )
    result_summary = [
        "# Soskic donor-axis result summary",
        "",
        one_sentence,
        "",
        f"- Complete donor folds: {len(complete_donors)} / {EXPECTED_DONORS}",
        f"- Incomplete donor folds remaining: {max(0, incomplete)}",
        f"- Primary comparator: per-donor best of cell-mean and donor-shift.",
        "- Metric sign convention in bootstrap summary: positive delta means scGen is better than the primary baseline; for E-distance/AUCell-MAE the sign is flipped before comparison.",
        f"- Manuscript branch flag: {sign}",
        "",
        "## Bootstrap summary",
        "",
        "| metric | lower_better | n | mean | lo | hi | pct_positive | boot_pos_frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *[
            "| {metric} | {lower_better} | {n} | {mean:.4f} | {lo:.4f} | {hi:.4f} | "
            "{pct_positive:.4f} | {boot_pos_frac:.4f} |".format(**r)
            for r in summary.to_dict("records")
        ],
    ]
    (ROOT / "reports" / "soskic_result_summary.md").write_text("\n".join(result_summary) + "\n")

    methods = f"""# Supplementary Methods — Soskic Donor-Axis LODO

Soskic 2022 CD4 T-cell activation was evaluated on the openly available Trynka processed HVG h5ad files.
The main donor-axis contrast was 0h resting control to 16h highly active stimulation. The processed portal
files contain 106 paired donors with both 0h and 16h CD4 cells, so the biological replication unit is donor
(n = 106 when complete), not the 119 recruited-donor count in the paper abstract.

For each leave-one-donor-out fold, the held donor's 16h cells were hidden from training and scored from
that donor's own 0h cells. CD4 naive and CD4 memory were retained as lineage strata. To keep every donor
represented under the large atlas size, `scripts/c2_soskic_donor.py` caps cells per donor × condition ×
lineage stratum before joint re-standardization on the shared 0h/16h HVG panel. Response genes are selected
inside each fold from training donors only by control-vs-stimulated Welch tests with BH q < 0.05, falling
back to the largest training-only mean shifts if too few genes pass.

The conditioned model is scGen, matching the Kang donor comparison. The four simple baselines are
control-as-prediction, training-mean (cell-mean) shift, donor shift, and linear-PCA shift. The primary
comparator is the better of training-mean and donor shift within each donor fold. Pearson-Delta is computed
on fold-specific response genes; E-distance is computed in a PCA space fit on training cells only; AUCell
program shift error is reported for T-cell activation, IL2/STAT5, type-I IFN, and type-II IFN programs.
Uncertainty is a cluster bootstrap over donors; seeds, if added later, are technical repeats within donor.

Current completion: {len(complete_donors)} / 106 donor folds.
"""
    (supp / "soskic_methods.md").write_text(methods)

    handoff = {
        "status": verdict,
        "complete_donors": len(complete_donors),
        "expected_donors": EXPECTED_DONORS,
        "remaining_donors": max(0, incomplete),
        "inputs": args.inputs,
        "outputs": {
            "raw_axis_csv": str(axis_path.relative_to(ROOT)),
            "bootstrap_summary": "results/C2/soskic_donor_bootstrap_summary.csv",
            "table6_row": "results/soskic_table6_row.md",
            "forest_source": "figures/source_data/soskic_donor_forest.csv",
            "methods": "supp/soskic_methods.md",
            "summary": "reports/soskic_result_summary.md",
        },
        "resume_command_template": (
            "IVCBENCH_SCPERTURBENCH_EVAL_PYTHON=<conda-env>/"
            "scperturbench_eval_jaxgpu/bin/python .venv/bin/python scripts/c2_soskic_donor.py "
            "--chunk I 4 --cap 300 --epochs 60 --gpu G --scgen-accelerator gpu "
            "--scgen-devices 1 --skip-existing "
            "--out results/C2/soskic_donor_shardIof4.csv"
        ),
        "runner_note": (
            "Use the cloned scperturbench_eval_jaxgpu env for GPU scGen. The original scperturbench_eval "
            "env still has CPU-only JAX."
        ),
    }
    (ROOT / "reports" / "soskic_handoff.json").write_text(json.dumps(handoff, indent=2))
    print(one_sentence)
    print(f"wrote {axis_path}, {figsrc / 'soskic_donor_forest.csv'}, reports/soskic_result_summary.md")


if __name__ == "__main__":
    main()
