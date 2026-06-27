#!/usr/bin/env python3
"""L2 acceptance: does a from-scratch retrain reproduce the deposited census?

    python scripts/census_accept.py <regenerated_headline.csv> <frozen_headline.csv> [--tol 0.03]

Compares the census assembled from freshly-retrained bundles against the frozen deposited census,
cell by cell. The bar (per the reproduction charter): every cell reproduces within run-to-run
variation (default |Δ pearson_delta| <= 0.03); the two floor-clearing cells the paper's conclusions
turn on (CellOT C2 donor 0.3666, FP-ridge C5 cell-context 0.3874) reproduce essentially exactly; and
the floor-clearance verdict (which cells beat BOTH universal-floor members) is unchanged. Exit 0 = PASS.
"""
import sys
import pandas as pd

KEY = ["cluster", "split", "model"]
FLOOR_CLEARERS = {("C2", "CellOT"), ("C5", "FP-ridge")}
tol = 0.03
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if "--tol" in sys.argv:
    tol = float(sys.argv[sys.argv.index("--tol") + 1])
fresh = pd.read_csv(args[0]); frozen = pd.read_csv(args[1])

m = frozen.merge(fresh, on=KEY, suffixes=("_frozen", "_fresh"), how="outer", indicator=True)
problems, floorclear_ok, verdict_ok = [], True, True

missing = m[m["_merge"] != "both"]
for _, r in missing.iterrows():
    problems.append(f"MISSING cell {r.cluster}/{r.split}/{r.model}: only in {r._merge}")

both = m[m["_merge"] == "both"].copy()
both["d"] = (both["pearson_delta_frozen"] - both["pearson_delta_fresh"]).abs()
print(f"{'cluster':<6}{'model':<16}{'split':<34}{'frozen':>9}{'fresh':>9}{'|Δ|':>8}")
print("-" * 82)
for _, r in both.sort_values("d", ascending=False).iterrows():
    fc = (r.cluster, r.model) in FLOOR_CLEARERS
    bad = r.d > (1e-4 if fc else tol)
    flag = ""
    if fc and r.d > 1e-4:
        flag = "  <-- FLOOR-CLEARER DRIFT"; floorclear_ok = False
    elif bad:
        flag = f"  <-- > {tol}"
    if bad or fc:
        print(f"{r.cluster:<6}{r.model:<16}{str(r.split)[:33]:<34}"
              f"{r.pearson_delta_frozen:>9.4f}{r.pearson_delta_fresh:>9.4f}{r.d:>8.4f}{flag}")
    if bad:
        problems.append(f"{r.cluster}/{r.model}/{r.split}: |Δ|={r.d:.4f} > tol")
    # verdict columns may be named beats_both_floor_members
    vcol = "beats_both_floor_members"
    if f"{vcol}_frozen" in both.columns and bool(r[f"{vcol}_frozen"]) != bool(r[f"{vcol}_fresh"]):
        problems.append(f"{r.cluster}/{r.model}/{r.split}: floor verdict flipped"); verdict_ok = False

print("-" * 82)
n = len(both)
within = int((both["d"] <= tol).sum())
print(f"cells compared      : {n}")
print(f"within tol ({tol})    : {within}/{n}")
print(f"floor-clearers exact : {'YES' if floorclear_ok else 'NO -- DRIFT'}  ({sorted(FLOOR_CLEARERS)})")
print(f"floor verdicts intact: {'YES' if verdict_ok else 'NO -- FLIP'}")
if problems:
    print(f"\nL2 ACCEPTANCE: FAIL  ({len(problems)} problem(s))")
    for p in problems[:40]:
        print("  -", p)
    sys.exit(1)
print(f"\nL2 ACCEPTANCE: PASS  (all {n} cells within {tol}; floor-clearers exact; verdicts intact) "
      "-- a from-scratch retrain reproduces the deposited census.")
