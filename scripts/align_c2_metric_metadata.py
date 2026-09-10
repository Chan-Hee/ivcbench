#!/usr/bin/env python3
"""Repair missing T2 evaluation-mask metadata without changing fitted predictions.

The Biolord/CellFlow revision driver omitted response_gene_fn, so those bundles
were initially scored on all genes, unlike the donor census. Copy the exact
training-fold mask from the matching cell-mean reference after verifying common
genes, observed targets, strata and controls. Original bundles remain untouched.
No model is trained and no prediction array is modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / "predictions/C2_metric_aligned"
    output.mkdir(exist_ok=True)
    records = []
    for model in ("Biolord", "CellFlow"):
        sources = sorted(
            (ROOT / "predictions").glob(f"C2_LODO__{model}__C2_lodo_*.npz")
        )
        if len(sources) != 106:
            raise ValueError(
                f"{model}: expected 106 original donor bundles, got {len(sources)}"
            )
        for source in sources:
            reference = (
                ROOT
                / "predictions/C2"
                / source.name.replace(f"__{model}__", "__cell-mean__")
            )
            with (
                np.load(source, allow_pickle=True) as bundle,
                np.load(reference, allow_pickle=True) as floor,
            ):
                values = {key: bundle[key] for key in bundle.files}
                if "exclude_gene_idx" in values:
                    raise ValueError(f"Source already contains a metric mask: {source}")
                for key in ("genes", "strata", "obs_means", "control_mean"):
                    left, right = values[key], floor[key]
                    equal = (
                        np.array_equal(left, right)
                        if key in {"genes", "strata"}
                        else left.shape == right.shape
                        and np.allclose(left, right, atol=1e-4, rtol=1e-5)
                    )
                    if not equal:
                        raise ValueError(
                            f"{source.name}: {key} differs from the donor reference"
                        )
                mask = np.asarray(floor["exclude_gene_idx"], int)
                if (
                    not len(mask)
                    or mask.min() < 0
                    or mask.max() >= len(values["genes"])
                ):
                    raise ValueError("Invalid reference gene mask")
                values["exclude_gene_idx"] = mask
            values["metric_metadata_source"] = str(reference.relative_to(ROOT))
            values["supersedes_bundle"] = str(source.relative_to(ROOT))
            target = output / source.name
            if target.exists():
                with np.load(target, allow_pickle=True) as existing:
                    if set(existing.files) != set(values) or any(
                        not np.array_equal(existing[k], v) for k, v in values.items()
                    ):
                        raise ValueError(
                            "Refusing to overwrite a different corrected bundle:"
                            f" {target}"
                        )
            else:
                np.savez_compressed(target, **values)
            with (
                np.load(source, allow_pickle=True) as original,
                np.load(target, allow_pickle=True) as corrected,
            ):
                for key in original.files:
                    if not np.array_equal(original[key], corrected[key]):
                        raise ValueError(f"Original array changed: {source.name}/{key}")
            records.append(
                dict(
                    model=model,
                    donor=source.stem.split("C2_lodo_")[-1],
                    original=str(source.relative_to(ROOT)),
                    original_sha256=digest(source),
                    reference=str(reference.relative_to(ROOT)),
                    reference_sha256=digest(reference),
                    corrected=str(target.relative_to(ROOT)),
                    corrected_sha256=digest(target),
                    excluded_genes=len(mask),
                    all_original_arrays_unchanged=True,
                )
            )
    manifest = ROOT / "results/_paper/c2_metric_metadata_corrections.json"
    manifest.write_text(
        json.dumps(
            {
                "reason": (
                    "T2 training-fold response-gene mask omitted by the revision driver"
                ),
                "new_training": False,
                "original_bundles_preserved": True,
                "corrections": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Verified {len(records)} mask-corrected copies; originals and prediction"
        " arrays unchanged"
    )


if __name__ == "__main__":
    main()
