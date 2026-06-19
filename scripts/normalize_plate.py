#!/usr/bin/env python
"""Plate alignment normalizer (final figure step). The five figures are saved with bbox_inches='tight',
which leaves DIFFERENT fractional outer margins (framework was inset ~3.8%, ranking ~0.7%), so stacked at
equal manuscript width the column looks ragged on the left. This crops each rendered PNG to its content
bounding box and re-pads to ONE identical fractional margin on every side, so all five share a common left
gutter (and panel letters land on a common vertical line). PNG only — the vector PDFs keep their own tight
box. Run AFTER rendering all figures (it re-renders them first for idempotence):
    .venv/bin/python scripts/normalize_plate.py
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
PAPER = ROOT / "results" / "_paper"
SCRIPTS = ["figure_framework", "figure_landscape", "figure_ranking", "figure_cellcontext",
           "figure_perturbation", "figure_donor_decision",
           "figure_immune_blindspot",   # v2 §5 immune blind-spot map
           "figure_within_family_fit"]  # v2 §4 within-family consistency + descriptive fit-matrix
MARGIN_FRAC = 0.012   # uniform outer margin as a fraction of the cropped content width


def content_bbox(im, thresh=248):
    g = np.asarray(im.convert("L"))
    ink = g < thresh
    cols = np.where(ink.any(axis=0))[0]
    rows = np.where(ink.any(axis=1))[0]
    if len(cols) == 0 or len(rows) == 0:
        return (0, 0, im.width, im.height)
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def normalize(png, rerender=True):
    if rerender:
        r = subprocess.run([PY, str(ROOT / "scripts" / f"{png}.py")], capture_output=True, text=True, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"  !! {png} render failed:\n{r.stderr[-400:]}"); return None
    p = PAPER / f"{png}.png"
    im = Image.open(p).convert("RGB")
    l, t, rgt, b = content_bbox(im)
    crop = im.crop((l, t, rgt, b))
    m = max(1, round(MARGIN_FRAC * crop.width))
    canvas = Image.new("RGB", (crop.width + 2 * m, crop.height + 2 * m), "white")
    canvas.paste(crop, (m, m))
    canvas.save(p, dpi=(400, 400))
    return 100 * m / canvas.width


def main():
    print("normalizing plate outer margins to a uniform %.1f%% gutter ..." % (MARGIN_FRAC * 100))
    for s in SCRIPTS:
        lm = normalize(s)
        if lm is not None:
            im = Image.open(PAPER / f"{s}.png")
            print(f"  {s:24} left-margin {lm:4.1f}%   ({im.width}x{im.height})")
    # verify spread
    margins = []
    for s in SCRIPTS:
        im = Image.open(PAPER / f"{s}.png")
        l, _, r, _ = content_bbox(im)
        margins.append(100 * l / im.width)
    print(f"\nleft-margin spread after normalize: {min(margins):.2f}%–{max(margins):.2f}% "
          f"(was 0.69%–3.84%); target ≤ ±0.3% around {MARGIN_FRAC*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
