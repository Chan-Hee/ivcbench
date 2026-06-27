#!/usr/bin/env python3
"""Reproduce-by-construction comparator: re-score FRESH bundles vs the frozen DEPOSIT, per unit.

    python scripts/compare_bundles.py <fresh_dir> [deposit_dir]

Matches bundles by basename, re-scores each side with the frozen census scorer
(ivcbench.eval.bundle.score_bundle), and reports per-bundle |fresh - deposited| pearson_delta plus a
summary (max/mean spread = the empirical tolerance for that producer). The deposit is read-only ground
truth; nothing is written. Exit 0 always (reporting tool). Default deposit_dir = predictions/.
"""
import glob, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ivcbench.eval.bundle import score_bundle  # noqa: E402

fresh_dir = sys.argv[1]
deposit_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "predictions")

fresh = sorted(glob.glob(os.path.join(fresh_dir, "**", "*.npz"), recursive=True))
if not fresh:
    print(f"no fresh bundles under {fresh_dir}"); sys.exit(0)

# index deposit by basename (recursive, so predictions/<cluster>/<file>.npz is found)
dep = {os.path.basename(f): f for f in glob.glob(os.path.join(deposit_dir, "**", "*.npz"), recursive=True)}

rows, missing = [], []
for fb in fresh:
    name = os.path.basename(fb)
    if name not in dep:
        missing.append(name); continue
    fp = score_bundle(fb)["pearson_delta"]
    dp = score_bundle(dep[name])["pearson_delta"]
    rows.append((name, fp, dp, abs(fp - dp)))

rows.sort(key=lambda r: -r[3])  # worst diff first
print(f"{'bundle':<46} {'fresh':>8} {'deposit':>8} {'|diff|':>8}")
print("-" * 74)
for name, fp, dp, d in rows:
    flag = "" if d <= 1e-4 else ("  <-- >0.03" if d > 0.03 else "  <-- drift")
    print(f"{name:<46} {fp:>8.4f} {dp:>8.4f} {d:>8.4f}{flag}")
if missing:
    print(f"\n{len(missing)} fresh bundle(s) NOT in deposit (unexpected name):")
    for m in missing[:10]:
        print("   ", m)

diffs = [d for *_, d in rows]
if diffs:
    print("-" * 74)
    print(f"compared {len(rows)} bundles | max|diff|={max(diffs):.4f} mean|diff|={np.mean(diffs):.4f} "
          f"| bit-identical={sum(d<=1e-9 for d in diffs)}/{len(diffs)} | within 0.03={sum(d<=0.03 for d in diffs)}/{len(diffs)}")
