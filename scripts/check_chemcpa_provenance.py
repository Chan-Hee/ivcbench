#!/usr/bin/env python3
"""Verify an existing native chemCPA bundle against preserved seed outputs; no training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IVC = ROOT
RAW = ROOT / "provenance/chemcpa"
if not RAW.is_dir():
    RAW = ROOT.parent / "benchmark/outputs/additional_models"
sys.path.insert(0, str(IVC / "src"))
from ivcbench.eval.bundle import score_bundle
from ivcbench.metrics.response import pearson_delta


def main():
    path = IVC / "predictions/C5/C5__chemCPA__C5_global_compound_holdout.npz"
    sources = [RAW / f"chemcpa_native_seed{s}.npz" for s in (0, 1, 2)]
    by_compound = {}
    with np.load(path, allow_pickle=True) as bundle:
        for seed, source in enumerate(sources):
            with np.load(source, allow_pickle=True) as raw:
                assert int(raw["seed"]) == seed
                assert str(raw["cov_mode"]) == "constant"
                assert np.array_equal(raw["genes"], bundle["genes"])
                assert (
                    len(raw["pred_compounds"]) == len(set(raw["pred_compounds"])) == 28
                )
                for compound, vector in zip(raw["pred_compounds"], raw["pred_means"]):
                    by_compound.setdefault(str(compound), []).append(vector.copy())
        assert all(len(vectors) == 3 for vectors in by_compound.values())
        expected = np.vstack(
            [
                np.mean(by_compound[str(s).split("|")[0].split("=", 1)[1]], axis=0)
                for s in bundle["strata"]
            ]
        )
        assert expected.shape == bundle["pred_means"].shape == (112, 2000)
        # The evaluator tiled these float32 means to cells and the bundle writer
        # averaged them back to strata. Repeated float32 summation is not bitwise
        # identical to the original mean but must agree well below display precision.
        np.testing.assert_allclose(expected, bundle["pred_means"], rtol=0, atol=1e-5)
        reconstructed_score = pearson_delta(
            expected.astype(float),
            bundle["obs_means"].astype(float),
            bundle["control_mean"].astype(float),
            bundle["strata"],
        )["macro"]
        stored_score = score_bundle(path)["pearson_delta"]
        assert abs(reconstructed_score - stored_score) < 1e-5
        max_diff = float(np.abs(expected - bundle["pred_means"]).max())
    evidence_files = sources + [path, ROOT / "scripts/chemcpa_native_op3.py"]
    result = {
        "method_group": "CPA/chemCPA",
        "execution_model": "chemCPA",
        "seeds": [0, 1, 2],
        "seed_aggregation": "mean prediction before scoring",
        "covariate": "constant PBMC",
        "held_compounds": 28,
        "scored_strata": 112,
        "genes": 2000,
        "bundle_score": stored_score,
        "reconstructed_from_seeds_score": float(reconstructed_score),
        "max_absolute_prediction_difference": max_diff,
        "prediction_tolerance": 1e-5,
        "training_performed_by_this_check": False,
        "inputs": [
            {
                "path": (
                    str(p.relative_to(ROOT))
                    if p.is_relative_to(ROOT)
                    else str(p.relative_to(ROOT.parent))
                ),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in evidence_files
        ],
    }
    out = ROOT / "results/_paper/native_chemcpa_provenance.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
