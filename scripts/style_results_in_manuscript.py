#!/usr/bin/env python3
"""Restyle the benchmark Results region of a .docx to the host manuscript's typography.

The manuscript "Toward Immune Virtual Cells.docx" is Times New Roman throughout: body 10 pt, sub-headings
12 pt bold, the section heading 16 pt bold, figure/table captions 9 pt (bold lead), table footnotes 9 pt.
The inserted Results paragraphs were left at the 11 pt default with no explicit font, so they did not blend
in. This script walks the Results region and sets every run to Times New Roman at the manuscript size for
its paragraph class, preserving bold/italic, and sets Table-3 cells to Times New Roman. It also normalises
any stray prose "Fig. N" -> "Figure N" (the manuscript uses "Figure N").

Region = from the "An Immune-Aware Benchmark for Perturbation Prediction" section heading through the last
figure-caption paragraph. In the standalone results_section.docx that is the whole document; in the
manuscript copy it is just the inserted block (References etc. after it are untouched).

Run AFTER refresh_docx_figures.py + update_docx_captions.py:
    figure_*.py -> normalize_plate.py -> refresh_docx_figures.py -> update_docx_captions.py
                -> style_results_in_manuscript.py -> clean_paper_temp.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt
from docx.table import Table
from docx.text.paragraph import Paragraph

WD = Path(__file__).resolve().parents[1]
MD = WD / "results" / "_paper" / "results_section.md"
FONT = "Times New Roman"
SECTION_PT, SUBHEAD_PT, BODY_PT, CAPTION_PT = 16.0, 12.0, 10.0, 9.0


def headings_from_md():
    """Return (section_heading_text, set_of_subheading_texts) from the markdown."""
    sec, subs = None, set()
    for line in MD.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("## ") and not s.startswith("### "):
            sec = s[3:].strip()
        elif s.startswith("### "):
            subs.add(s[4:].strip())
    if not sec:
        sys.exit("FAIL: section heading (## ...) not found in md")
    return sec, subs


def set_runs(para: Paragraph, size_pt: float) -> None:
    """Times New Roman + size on every run; bold/italic preserved; prose 'Fig. N' -> 'Figure N'."""
    for r in para.runs:
        if r.text and "Fig. " in r.text:
            r.text = r.text.replace("Fig. ", "Figure ")
        r.font.name = FONT
        r.font.size = Pt(size_pt)


def style(docx_path: Path) -> None:
    doc = Document(str(docx_path))
    sec_text, subs = headings_from_md()
    blocks = list(doc.element.body.iterchildren())

    # locate region: from the section-heading paragraph to just before the first post-Results marker
    # ("References" or the "Toward Immune Virtual Cells" footer), else to the end of the document. This
    # is robust to figures being placed inline (the region is NOT defined by the last figure caption).
    POST = {"References", "Toward Immune Virtual Cells"}
    start = None
    end = len(blocks) - 1
    for i, el in enumerate(blocks):
        if el.tag.split("}")[-1] != "p":
            continue
        t = Paragraph(el, doc).text.strip()
        if start is None and t == sec_text:
            start = i
        elif start is not None and t in POST:
            end = i - 1
            break
    if start is None:
        sys.exit(f"FAIL {docx_path.name}: Results section heading not found")

    n_para = n_tbl = 0
    for i in range(start, end + 1):
        el = blocks[i]
        tag = el.tag.split("}")[-1]
        if tag == "tbl":
            for row in Table(el, doc).rows:        # Table 3 cells -> Times New Roman (keep size)
                for cell in row.cells:
                    for p in cell.paragraphs:
                        for r in p.runs:
                            r.font.name = FONT
            n_tbl += 1
            continue
        if tag != "p":
            continue
        p = Paragraph(el, doc)
        t = p.text.strip()
        if not [r for r in p.runs if r.text.strip()]:
            continue                                # empty / image-only paragraph
        if t == sec_text:
            size = SECTION_PT
        elif t in subs:
            size = SUBHEAD_PT
        elif re.match(r"^Figure\s+[2-6][.]", t) or re.match(r"^Table\s+[3-6][.]", t):
            size = CAPTION_PT                       # figure / table captions
        elif t[:1] in {"†", "‡"} or t.startswith("Simple baselines"):
            size = CAPTION_PT                       # table footnotes
        else:
            size = BODY_PT
        set_runs(p, size)
        n_para += 1

    doc.save(str(docx_path))
    print(f"  {docx_path.name}: styled {n_para} paragraphs + {n_tbl} table(s) "
          f"(region blocks {start}-{end}) to {FONT}")


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or [
        WD / "results" / "_paper" / "results_section.docx",
        WD.parent / "Toward Immune Virtual Cells (benchmark Results inserted).docx",
    ]
    for p in paths:
        if not p.exists():
            sys.exit(f"FAIL: docx not found: {p}")
        print(f"styling {p}")
        style(p)
    print("done.")
