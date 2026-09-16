#!/usr/bin/env python
"""Add the common panel mask to T3/T4 bundles that were deposited before the task carried it.

NATIVE_102_FINAL_REVIEW.md section 247 rules on scFoundation T3/T4 and PertAdapt T3/T4: padding
output genes the model did not predict with the control mean has to go, and its allowed column
names the replacement -- "a common evaluation mask over the genes actually output". Genes absent
from the released OS_scRNA_gene_index.19264.tsv have no input embedding and no output column in
either checkpoint-based model, so neither can predict them.

That is a property of the PANEL, not of a model, so the same genes are removed for every model on
the cell and for both floor members. scripts/run_obligation.py folds the mask into the task's
exclusion set, so runs started after that change carry it already. This script is for the bundles
deposited before it, and for runs that were already in flight when it landed.

The edit is additive and idempotent: it unions the mask indices into `exclude_gene_idx` and
touches nothing else. Writing it into the bundle rather than applying it at scoring time is what
keeps ivcbench.eval.bundle.score_bundle self-contained -- the GPU-free reproduction path
(scripts/reproduce_eval.py) scores from the deposit alone and must not need the released
vocabulary to do it.

Scope: the 2,000-gene RNA panels of the unseen-gene and unseen-knockout splits. The Frangieh
protein readout is excluded -- no checkpoint-based model is scored on it, so masking its
20-marker panel would remove genes for no reason.

ORDERING MATTERS. Rewriting a bundle updates its mtime, and
scripts/verify_cell_provenance.py --strict decides whether a bundle is newer than the job that
produced it by comparing exactly that. Run this AFTER the provenance check and after
supersede_reruns.py, and before assemble_cross_cluster.py, which regenerates
census_bundle_manifest.csv with the post-rewrite checksums. CENSUS_56_CASCADE.md fixes it at
step 4b for this reason.

    python scripts/apply_panel_mask.py              # report only
    python scripts/apply_panel_mask.py --apply      # rewrite, verifying every bundle
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# finish_census.sh calls this from revision_claude/, so a relative glob finds nothing and the
# migration reports "0 candidate bundle(s)" and exits happily, leaving the census half masked.
# Anchor on the repository instead of the working directory.
os.chdir(ROOT)

from ivcbench.eval.bundle import score_bundle  # noqa: E402
from ivcbench.eval.panel_mask import unrepresentable  # noqa: E402

SPLIT_RE = re.compile(r"^(C3_LO_gene|C4_Axis2|C4)__")
PANEL = 2000
# A bundle that has been withdrawn is evidence of a decision, pinned in withdrawn_bundles.csv by
# exact path AND sha256. Rewriting one would leave that record pointing at a checksum no file has.
# Directories kept as evidence rather than as results are skipped for the same reason.
WITHDRAWN_LIST = Path("results/_paper/withdrawn_bundles.csv")
EVIDENCE_DIRS = ("_withdrawn", "_diagnostic", "example", "_invalid")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _withdrawn() -> set[str]:
    if not WITHDRAWN_LIST.is_file():
        return set()
    import csv

    with WITHDRAWN_LIST.open() as handle:
        return {row["bundle_path"] for row in csv.DictReader(handle)}


def targets() -> list[Path]:
    withdrawn = _withdrawn()
    out = []
    for p in sorted(glob.glob("predictions/**/*.npz", recursive=True)):
        name = os.path.basename(p)
        if not SPLIT_RE.match(name) or "frangieh_protein" in name:
            continue
        if p in withdrawn or any(os.sep + d + os.sep in os.sep + p for d in EVIDENCE_DIRS):
            continue
        try:
            d = np.load(p, allow_pickle=True)
        except Exception:
            continue
        if "genes" not in d.files or "pred_means" not in d.files:
            continue
        if len(np.asarray(d["genes"])) != PANEL:
            continue
        out.append(Path(p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite the bundles")
    ap.add_argument("--record", default="results/_paper/panel_mask_migration.json")
    args = ap.parse_args()

    if args.apply:
        # supersede_reruns.py refuses the same way, and for the same reason. A job that started
        # before run_job carried the mask deposits an UNMASKED bundle when it finishes. Migrate
        # while those are in flight and they overwrite their own bundles afterwards, leaving
        # scFoundation and PertAdapt as the only unmasked models among masked peers -- the exact
        # two cells section 247 is about, scored on genes they cannot predict while everything
        # they are compared against is not. Worse than not masking at all.
        running = sorted(
            f.name[:-7]
            for f in Path("runs").glob("*.status")
            if f.read_text().startswith("RUNNING")
        )
        # Status files only cover jobs the dispatcher launched. Units driven by hand -- the
        # scFoundation re-runs are started from a screen session and write no status file --
        # would slip straight past that check, so look for the processes themselves too.
        live = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                argv = (entry / "cmdline").read_bytes().decode("utf8", "replace")
            except OSError:
                continue
            # Match argv ELEMENTS, not the joined string: a shell wrapper whose single argument
            # happens to quote the command would otherwise register as a live run. One did,
            # transiently, and an unnecessary refusal is a real cost when the migration has a
            # narrow window between the last job finishing and the assembler.
            bits = argv.split("\x00")
            if not any(b.endswith("run_obligation.py") for b in bits):
                continue
            if "--model" not in bits or "--task" not in bits:
                continue
            model = bits[bits.index("--model") + 1]
            task = bits[bits.index("--task") + 1]
            live.append(f"{model}@{task}(pid {entry.name})")
        if running or live:
            raise SystemExit(
                "refusing to migrate while work is still depositing.\n"
                + (f"  RUNNING status files: {', '.join(running)}\n" if running else "")
                + (f"  live run_obligation: {', '.join(sorted(set(live)))}\n" if live else "")
                + "They deposit unmasked bundles when they finish. Wait for them."
            )

    paths = targets()
    print(f"{len(paths)} candidate bundle(s)")
    log, changed, already = [], 0, 0
    for path in paths:
        d = np.load(path, allow_pickle=True)
        genes = [str(g) for g in np.asarray(d["genes"])]
        blind = unrepresentable(genes)
        if not blind:
            continue
        pos = {g: i for i, g in enumerate(genes)}
        mask = np.array(sorted(pos[g] for g in blind), dtype=int)
        base = (
            np.asarray(d["exclude_gene_idx"], int)
            if "exclude_gene_idx" in d.files
            else np.array([], int)
        )
        merged = np.union1d(base, mask)
        if merged.size == base.size and np.array_equal(np.sort(base), merged):
            already += 1
            continue

        before = score_bundle(str(path))["pearson_delta"]
        entry = {
            "bundle": str(path),
            "sha256_before": _sha(path),
            # Rewriting a bundle resets its mtime, and verify_cell_provenance.py decides whether a
            # bundle predates the job that produced it by comparing exactly that. After the first
            # migration every rewritten T3/T4 bundle was newer than every job stamp, so that check
            # could no longer fail for them while still printing "0 cell(s)". Record the mtime the
            # bundle had, and the gate can use it instead.
            "mtime_before": os.path.getmtime(path),
            "excluded_before": int(base.size),
            "panel_mask_genes": int(mask.size),
            "excluded_after": int(merged.size),
            "pearson_delta_before": before,
        }
        if args.apply:
            payload = {k: d[k] for k in d.files}
            payload["exclude_gene_idx"] = merged
            # np.savez appends ".npz" to any path that does not already end in it, so a
            # ".npz.tmp" name silently became ".npz.tmp.npz" and the replace below failed.
            # Handing it an open file keeps the name exactly as given.
            tmp = path.with_name(path.name + ".tmp")
            with tmp.open("wb") as handle:
                np.savez(handle, **payload)
            os.replace(tmp, path)
            after = score_bundle(str(path))["pearson_delta"]
            check = np.asarray(np.load(path, allow_pickle=True)["exclude_gene_idx"], int)
            assert np.array_equal(np.sort(check), merged), f"{path}: mask did not persist"
            entry["sha256_after"] = _sha(path)
            entry["pearson_delta_after"] = after
        changed += 1
        log.append(entry)

    verb = "rewrote" if args.apply else "would rewrite"
    print(f"{verb} {changed}; {already} already carried the mask")
    if args.apply and log:
        out = Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(log, indent=1))
        moves = [e["pearson_delta_after"] - e["pearson_delta_before"] for e in log]
        print(f"score moved by {min(moves):+.4f} to {max(moves):+.4f}; record in {out}")
    elif log:
        print(f"  example: {log[0]['bundle']} "
              f"(+{log[0]['panel_mask_genes']} genes to the metric mask)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
