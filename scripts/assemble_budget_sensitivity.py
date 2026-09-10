"""Rebuild the six CellOT budget summaries from preserved donor/seed scores."""

from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = ROOT / "results/provenance_inputs/budget"
    rows, inputs = [], []
    for multiplier in (1, 4, 12):
        for donors in (8, 96):
            path = raw / (
                "lc_1x_deposited.csv"
                if multiplier == 1
                else f"lc{donors}_{multiplier}x.csv"
            )
            data = pd.read_csv(path)
            data = data[
                (data.metric == "pearson_delta") & (data.n_train_donors == donors)
            ]
            assert len(data) == 20 and set(data.seed) == {0, 1}
            assert (
                data.eval_donor.nunique() == 10
                and not data.duplicated(["eval_donor", "seed"]).any()
            )
            score, reference = data.cellot_score.mean(), data.baseline_score.mean()
            rows.append(
                dict(
                    multiplier=multiplier,
                    ae_iterations=12000 * multiplier,
                    transport_iterations=8000 * multiplier,
                    training_donors=donors,
                    evaluation_donors=10,
                    seeds="0, 1",
                    model_score=score,
                    reference_score=reference,
                    margin=score - reference,
                    reference=(
                        "per-donor stronger cell-mean/donor-shift; historical auxiliary"
                        " reference"
                    ),
                )
            )
            inputs.append(
                dict(
                    path=str(path.relative_to(ROOT)),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    out = ROOT / "results/_paper"
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "cellot_budget_summary.csv", index=False)
    (out / "cellot_budget_provenance.json").write_text(
        json.dumps(
            dict(
                inputs=inputs,
                aggregation=(
                    "equal-weight donor/seed scalar scores; ten donors and two seeds"
                    " per row"
                ),
                inference="descriptive sensitivity; not an optimization guarantee",
            ),
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
