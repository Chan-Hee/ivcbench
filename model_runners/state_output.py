"""Select STATE's prediction artifact, never its paired query/real artifact."""

from pathlib import Path


def prediction_path(output_dir):
    candidates = sorted(Path(output_dir).rglob("adata_pred.h5ad"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one STATE adata_pred.h5ad, found {len(candidates)}; "
            "refusing a generic h5ad fallback or an ambiguous checkpoint."
        )
    return candidates[0]
