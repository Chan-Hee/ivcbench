#!/usr/bin/env python
"""Census value for one cell, computed the way the census computes it.

Two rules that a shard mean or a flat average gets wrong, and that cost a correction once:
  - macro over the BIOLOGICAL UNIT: average within the unit first, then across units;
  - the floor on the SAME unit set, so the margin is paired rather than two different averages.

  python runs/cell_value.py CellOT C5_loct          # model, split prefix
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from assemble_cross_cluster import eligible_bundle          # noqa: E402
from ivcbench.metrics.response import _pearson              # noqa: E402

FLOORS = ("cell-mean", "linear-PCA")


def unit_id(split: str) -> str | None:
    """The part both spellings share: the donor, lineage, fold or holdout the bundle scores."""
    s = split.strip()
    m = re.search(r"_(D\d+)$", s)                       # C2_soskic_LODO_D348 / C2_lodo_D348
    if m:
        return m.group(1)
    m = re.search(r"_loct_(.+)$", s, re.I)               # C5_loct_NK / C1_loct_B
    if m:
        return m.group(1)
    m = re.search(r"_(\d+)$", s)                         # lo_gene_10, lo_ko_25
    if m:
        return s                                         # keep the whole name: 10/25 repeat
    return s or None


def by_unit(model: str, split_prefix: str) -> dict[str, float]:
    out = {}
    for f in glob.glob("predictions/**/*.npz", recursive=True):
        if not eligible_bundle(f):
            continue
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        # Match on the UNIT IDENTIFIER, not the cluster and not the whole split name. Two traps:
        # some floor bundles carry a wrong cluster (C1_LOCT on a bundle whose split is C5_loct_B),
        # and the model and the floor spell the same unit differently -- C2_soskic_LODO_D348
        # against C2_lodo_D348. Matching on either one finds the model's units and none of the
        # floor's, and the paired comparison silently disappears.
        if str(z.get("model", "")) != model:
            continue
        sp = str(z.get("split", ""))
        if unit_id(sp) is None or not sp.lower().startswith(split_prefix.split("_")[0].lower()):
            continue
        if not {"pred_means", "obs_means", "control_mean"} <= set(z.files):
            continue
        P = np.atleast_2d(z["pred_means"].astype(np.float64))
        O = np.atleast_2d(z["obs_means"].astype(np.float64))
        C = z["control_mean"].astype(np.float64)
        keep = np.ones(P.shape[1], bool)
        if "exclude_gene_idx" in z.files:
            ex = np.asarray(z["exclude_gene_idx"], int)
            if ex.size:
                keep[ex] = False
        # unit = the split this bundle scores (lineage, donor, dataset fold)
        unit = unit_id(str(z.get("split", Path(f).stem)))
        out[unit] = float(np.mean([_pearson((P[i] - C)[keep], (O[i] - C)[keep])
                                   for i in range(P.shape[0])]))
    return out


def main() -> int:
    model, prefix = sys.argv[1], sys.argv[2]
    got = {m: by_unit(m, prefix) for m in (model, *FLOORS)}
    sizes = {m: len(v) for m, v in got.items()}
    if not got[model]:
        print(f"{model} · {prefix}*: 번들 없음"); return 1
    common = sorted(set.intersection(*[set(v) for v in got.values()]))
    print(f"{model} · split {prefix}* — unit {sizes}")
    if not common:
        print("  공통 unit 없음 (floor 번들이 같은 split 이름을 쓰지 않는다)")
        v = np.array(list(got[model].values()))
        print(f"  {model} macro = {v.mean():.4f}  (n={len(v)}, floor 비교 불가)")
        return 0
    print(f"  공통 unit {len(common)}개\n")
    s = np.array([got[model][u] for u in common])
    print(f"  {model:12s} macro = {s.mean():.4f}   최소 {s.min():.4f} · 중앙 {np.median(s):.4f} · 최대 {s.max():.4f}")
    binding = -9.0
    for m in FLOORS:
        f = np.array([got[m][u] for u in common])
        binding = max(binding, f.mean())
        print(f"  {m:12s} macro = {f.mean():.4f}   차이 {(s-f).mean():+.4f} · {model} 우세 {int((s>f).sum())}/{len(s)}")
    print(f"\n  구속 floor {binding:.4f} → {'floor 통과' if s.mean() > binding else 'floor 미달'} ({s.mean()-binding:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
