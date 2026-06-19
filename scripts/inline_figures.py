#!/usr/bin/env python3
"""Place the five benchmark figures INLINE — each right after the paragraph that first references it —
instead of grouping them all at the end of the Results section.

For each figure N (2..6): the caption is the paragraph starting "Figure N.", the image is the nearest
image paragraph before it, and the target is the FIRST body paragraph that mentions "Figure N" (a body
reference such as "(Figure 4a)", not a caption). The image+caption pair is moved to immediately after the
target, so the figure appears where the reader first meets it. Idempotent: re-running re-anchors the pair
to the same reference paragraph.

This MOVES paragraphs only — it changes no text, style, or image. Run it LAST in the chain (after the
typography + Table-3 steps), because moving figures inline would otherwise confuse a region detector that
keys on the last figure caption.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from docx import Document

WD = Path(__file__).resolve().parents[1]
BLIP = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
NUMS = ["2", "3", "4", "5", "6"]


def inline(docx_path: Path) -> None:
    doc = Document(str(docx_path))
    paras = doc.paragraphs

    cap_el, img_el = {}, {}
    for i, p in enumerate(paras):
        m = re.match(r"^Figure\s+([2-6])[.]", p.text.strip())
        if not m:
            continue
        n = m.group(1)
        cap_el.setdefault(n, p._element)
        for j in range(i - 1, -1, -1):                       # nearest preceding image paragraph
            if paras[j]._element.findall(".//" + BLIP):
                img_el.setdefault(n, paras[j]._element)
                break

    tgt_el = {}
    for n in NUMS:
        for p in paras:
            t = p.text.strip()
            if t.startswith("Figure "):                      # skip captions / figure-legend lines
                continue
            if f"Figure {n}" in t:                            # first body reference (e.g. "(Figure 4a)")
                tgt_el[n] = p._element
                break

    moved = []
    for n in NUMS:
        if n not in img_el or n not in cap_el or n not in tgt_el:
            print(f"  [skip] Figure {n}: "
                  f"img={n in img_el} cap={n in cap_el} ref={n in tgt_el}")
            continue
        # move pair to right after the reference: target -> image -> caption
        tgt_el[n].addnext(cap_el[n])
        tgt_el[n].addnext(img_el[n])
        moved.append(n)
    doc.save(str(docx_path))
    print(f"  {docx_path.name}: placed Figures {moved} inline after their first reference")


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or [
        WD / "results" / "_paper" / "results_section.docx",
        WD.parent / "Toward Immune Virtual Cells (benchmark Results inserted).docx",
    ]
    for p in paths:
        if not p.exists():
            sys.exit(f"FAIL: docx not found: {p}")
        print(f"inlining figures in {p}")
        inline(p)
    print("done.")
