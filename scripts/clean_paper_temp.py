#!/usr/bin/env python3
"""Remove inspection/debug image crops from results/_paper/, keeping only the paper deliverables.

results/_paper/ accumulates many ad-hoc crop PNGs from figure-QC / zoom-inspection sessions
(_crop_*.png, _zoom_*.png, _qc*.png, q_*.png, …) plus the legacy `figure_integrated` composite that is
NOT part of the manuscript (the .docx embeds only the five figure_* deliverables). None of these are
read by any figure script or the .docx. This deletes every IMAGE (.png/.pdf) in results/_paper/ except
the five deliverable figures, and never touches data/records (.md, .csv, .log, .json), the .docx, or
_archive/. Safe to run any time — it is the cleanup step of the figure rebuild chain:

    figure_*.py -> normalize_plate.py -> refresh_docx_figures.py -> update_docx_captions.py -> clean_paper_temp.py

Use --dry-run to preview without deleting.
"""
from __future__ import annotations
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "results" / "_paper"
# the only images that belong in results/_paper/: the manuscript figures (png + pdf each)
KEEP_STEMS = {
    "figure_framework", "figure_ranking", "figure_cellcontext",
    "figure_perturbation", "figure_donor_decision",
    "figure_landscape",          # v2 §2 method × split landscape
    "figure_immune_blindspot",   # v2 §5 immune blind-spot map (Figure 7)
    "figure_within_family_fit",  # v2 §4 within-family consistency + descriptive fit-matrix (Figure 8)
}
IMAGE_SUFFIXES = {".png", ".pdf"}


def main(dry_run: bool) -> None:
    removed, kept_imgs, kept_other = [], [], []
    for p in sorted(PAPER.iterdir()):
        if p.is_dir():
            continue
        if p.suffix.lower() in IMAGE_SUFFIXES:
            if p.stem in KEEP_STEMS:
                kept_imgs.append(p.name)
            else:
                removed.append(p)
        else:
            kept_other.append(p.name)   # .md/.csv/.log/.json/.docx = records/sources, always kept

    total = sum(p.stat().st_size for p in removed)
    verb = "would remove" if dry_run else "removed"
    print(f"{verb} {len(removed)} non-deliverable image file(s) ({total/1e6:.1f} MB):")
    for p in removed:
        if not dry_run:
            p.unlink()
        print("  -", p.name)
    print(f"\nkept {len(kept_imgs)} deliverable figure image(s): {', '.join(sorted(kept_imgs))}")
    print(f"kept {len(kept_other)} record/source file(s): {', '.join(sorted(kept_other))}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
