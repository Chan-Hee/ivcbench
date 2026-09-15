#!/usr/bin/env python
"""Register the historical bundle that each re-run replaces, with provenance.

A native re-run deposits into predictions/v2_native/ under the SAME filename as the bundle it
replaces (it is the same cluster/model/split). Both are then visible to the assembler's recursive
glob, which refuses the ambiguity rather than silently picking one -- correctly, because averaging
or first-wins would have been an invisible choice between two different executions.

This resolves it the way the audit requires: the OLD bundle is withdrawn by exact path AND sha256,
naming the run that supersedes it. Nothing is deleted; the history stays on disk and the registry
records why it no longer counts.

  python scripts/supersede_reruns.py [--apply]     (default: report only)
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "predictions" / "v2_native"
REG = ROOT / "results" / "_paper" / "withdrawn_bundles.csv"
REASON = (
    "superseded by {new} -- native re-run of 2026-09-14 under the B3 float64 stratum-mean fix, "
    "with declined-stratum coverage recorded in the bundle"
)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    apply = "--apply" in sys.argv
    # Never withdraw a historical bundle while its replacement is still being written. If the new
    # run then failed, the cell would vanish: the old bundle withdrawn, the new one absent.
    running = [
        p.stem
        for p in (ROOT / "runs").glob("*.status")
        if p.read_text().startswith("RUNNING")
    ]
    if apply and running:
        print(
            "REFUSED: jobs are still running -- "
            + ", ".join(sorted(running))
            + "\nSuperseding now could withdraw a historical bundle whose replacement never lands.",
            file=sys.stderr,
        )
        sys.exit(5)
    if not NEW.is_dir():
        print(f"no re-run directory at {NEW}")
        return
    rows = list(csv.DictReader(open(REG))) if REG.exists() else []
    already = {r["bundle_path"] for r in rows}

    added, missing = [], []
    for new in sorted(NEW.glob("*.npz")):
        old = [
            p
            for p in ROOT.joinpath("predictions").rglob(new.name)
            if p != new and NEW not in p.parents
        ]
        if not old:
            missing.append(new.name)          # a genuinely new cell; nothing to supersede
            continue
        for o in old:
            rel = str(o.relative_to(ROOT))
            if rel in already:
                continue
            added.append(
                {
                    "bundle_path": rel,
                    "sha256": sha256(o),
                    "reason": REASON.format(new=new.relative_to(ROOT)),
                }
            )
            already.add(rel)

    print(f"re-run bundles: {len(list(NEW.glob('*.npz')))}")
    print(f"newly superseded historical bundles: {len(added)}")
    for a in added:
        print(f"  - {a['bundle_path']}")
    if missing:
        print(f"no historical counterpart (new cells): {len(missing)}")
        for m in missing[:10]:
            print(f"  + {m}")
    if not apply:
        print("\n(report only; pass --apply to write the registry)")
        return
    with open(REG, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["bundle_path", "sha256", "reason"])
        w.writeheader()
        w.writerows(rows + added)
    print(f"\nwrote {REG} ({len(rows) + len(added)} entries)")


if __name__ == "__main__":
    main()
