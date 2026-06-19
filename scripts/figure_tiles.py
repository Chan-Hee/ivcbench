#!/usr/bin/env python
"""Zoom-tiler for meticulous figure QC. Splits a rendered figure PNG into an overlapping grid of
full-resolution tiles so a reviewer can inspect label/marker collisions, clipped text, leader-line
crossings, and margin inconsistencies that are invisible at whole-figure scale. Prints the tile paths.
    .venv/bin/python scripts/figure_tiles.py figure_cellcontext
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results" / "_paper"


def tile(name, tilepx=1250, overlap=0.20, outdir="/tmp/figtiles"):
    im = Image.open(PAPER / f"{name}.png").convert("RGB")
    W, H = im.size
    out = Path(outdir) / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    step = max(1, int(tilepx * (1 - overlap)))
    ncol = (math.ceil((W - tilepx) / step) + 1) if W > tilepx else 1
    nrow = (math.ceil((H - tilepx) / step) + 1) if H > tilepx else 1
    paths = []
    for r in range(nrow):
        for c in range(ncol):
            x0 = min(c * step, max(0, W - tilepx)); y0 = min(r * step, max(0, H - tilepx))
            x1 = min(x0 + tilepx, W); y1 = min(y0 + tilepx, H)
            t = im.crop((x0, y0, x1, y1))
            p = out / f"tile_r{r}_c{c}.png"
            t.save(p)
            paths.append(str(p))
    print(f"{name}: {W}x{H} -> {len(paths)} tiles ({nrow} rows x {ncol} cols), {int(tilepx*overlap)}px overlap")
    for p in paths:
        print(p)
    return paths


if __name__ == "__main__":
    tile(sys.argv[1] if len(sys.argv) > 1 else "figure_cellcontext")
