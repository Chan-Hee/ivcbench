#!/usr/bin/env python3
"""Descriptive family-by-task summary from the exact census biological units.

This is not an a-priori prediction, a family-level hypothesis test, or a model
selection recommendation. The best recorded member is selected retrospectively
within each task/family. Its own paired interval is conditional on that choice;
selection and multiplicity uncertainty are not accounted for. Diagnostic
comparators are reported separately, never substituted for conditioned members.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from assemble_cross_cluster import CELLS, FAMILY, ROOT, cell_status
from census_units import build_unit_scores, uncertainty_table, task_key, validate_units

PAPER = Path(ROOT) / "results/_paper"


def build_fit_matrix(units=None):
    units = build_unit_scores() if units is None else units
    validate_units(units)
    uncertainty = uncertainty_table(units)
    records = []
    for cell in CELLS:
        key = task_key(cell)
        groups = {}
        for model in cell["roster"]:
            status = cell_status(cell["cl"], cell["split"], model)
            family = FAMILY[model]
            # FP-ridge is a diagnostic chemistry comparator, not a stand-in for
            # native chemistry predictors. CINEMA-OT is likewise not conditioned OT.
            group = (
                family,
                "diagnostic: " + model if status == "diagnostic" else "conditioned",
            )
            groups.setdefault(group, []).append(model)
        for (family, role), members in sorted(groups.items()):
            rows = uncertainty[
                (uncertainty.task_key == key) & uncertainty.model.isin(members)
            ]
            scored = rows.sort_values(
                ["model_score", "model"], ascending=[False, True]
            ).iloc[0]
            record = dict(
                task_key=key,
                cluster=cell["cl"],
                task_name=cell["task"],
                task_split=cell["split"],
                family=family,
                role=role,
                family_baselines=";".join(sorted(members)),
                scored_member=scored.model,
                status=scored.status,
                unit=cell["unit"],
                n_unit=int(scored.n_units),
                metric="pearson_delta",
                binding_floor=scored.binding_floor,
                floor=float(scored.binding_floor_score),
                model_score=float(scored.model_score),
                best_model_gap=float(scored.margin),
                ci_low=scored.ci_lo,
                ci_high=scored.ci_hi,
                margin_min=float(scored.margin_min),
                margin_max=float(scored.margin_max),
                units_above_floor=scored.units_above_floor,
                inference=scored.inference,
                evidence=scored.verdict,
                member_selection="retrospective maximum task-macro score",
                interpretation=(
                    "descriptive; member and floor held fixed; no selection or"
                    " multiplicity correction"
                ),
            )
            records.append(record)
    frame = pd.DataFrame(records)
    if frame.duplicated(["task_key", "family", "role"]).any():
        raise ValueError("Duplicate S3 task/family/role")
    return frame


def main():
    units = build_unit_scores()
    frame = build_fit_matrix(units)
    units.to_csv(PAPER / "census_unit_scores.csv", index=False)
    uncertainty_table(units).to_csv(PAPER / "census_uncertainty.csv", index=False)
    frame.to_csv(PAPER / "descriptive_fit_matrix.csv", index=False)
    lines = [
        "# Descriptive family-by-task summary",
        "",
        (
            "Each row reports the highest-scoring recorded member of that family on"
            " that task,"
        ),
        (
            "selected retrospectively. The named member's paired margin uses the same"
            " units and"
        ),
        (
            "split-fixed binding floor as the census and S18. Diagnostic comparators"
            " have separate"
        ),
        (
            "rows. Intervals (4,000 unit-bootstrap draws; seed 0; at least eight"
            " units) are"
        ),
        "unadjusted and conditional on member/floor selection, not family-level tests.",
        "Smaller samples receive unit ranges, not significance claims. No a-priori",
        (
            "expectation or general model-selection recommendation is inferred from"
            " these results."
        ),
        "",
        (
            "| Task | Family / role | Scored member | n | Score | Floor | Margin |"
            " Interval or unit range |"
        ),
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in frame.itertuples():
        interval = (
            f"95% CI [{row.ci_low:+.3f}, {row.ci_high:+.3f}]"
            if pd.notna(row.ci_low)
            else f"range [{row.margin_min:+.3f}, {row.margin_max:+.3f}]"
        )
        lines.append(
            f"| {row.task_key} | {row.family} / {row.role} | {row.scored_member} | "
            f"{row.n_unit} | {row.model_score:.3f} | {row.floor:.3f} | "
            f"{row.best_model_gap:+.3f} | {interval} |"
        )
    (PAPER / "descriptive_fit_matrix.md").write_text("\n".join(lines) + "\n")
    print(
        f"S3: {len(frame)} descriptive rows;"
        f" {int((frame.role == 'conditioned').sum())} conditioned families"
    )
    print(
        frame[
            [
                "task_key",
                "family",
                "role",
                "scored_member",
                "best_model_gap",
                "evidence",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
