#!/usr/bin/env python3
"""Rebuild Table 3 (the statistical-decision table) in the manuscript's own table style.

The inserted Table 3 used AUTOFIT layout, which Microsoft Word re-flows from cell content — with nine
dense columns that produced a broken layout. The manuscript's Methods tables (Table 1/2) instead use a
FIXED layout with explicit per-column widths, borderless, bold Times-New-Roman header. This rewrites
Table 3 to match: fixed layout, explicit column grid, the table extended a little past the text column
toward the page edge (as Table 1 is) so nine columns get room at a readable size, Times New Roman header
bold. Content is preserved verbatim — only the table geometry/style/fonts change.

Run as part of the manuscript-integration chain (after the figures/captions/typography steps)."""
from __future__ import annotations
import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Twips
from docx.table import Table

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
WD = Path(__file__).resolve().parents[1]
FONT, FONT_PT = "Times New Roman", 9.0
# per-column width fractions (sum 1.0), allocated by content:
# Axis | Dataset/split | Eval unit(n) | Primary baseline | Best native cond | Distrib comparator | Δ(CI) | Test | Verdict
FRACS = [0.0576, 0.1418, 0.0908, 0.1329, 0.1196, 0.0953, 0.1551, 0.1019, 0.1050]


def qn(tag: str) -> str:
    return W + tag


def find_table3(doc):
    for el in doc.element.body.iterchildren():
        if el.tag == qn("tbl"):
            t = Table(el, doc)
            if t.rows and t.rows[0].cells[0].text.strip() == "Axis":
                return el, t
    return None, None


def rebuild(docx_path: Path) -> None:
    doc = Document(str(docx_path))
    sec = doc.sections[0]
    # table width: from the left text margin out toward the page edge, leaving ~0.28in right margin
    # (matches Table 1, which extends past the body text column). twips = EMU/635.
    table_w = int((sec.page_width - sec.left_margin - int(0.28 * 914400)) / 635)
    el, t = find_table3(doc)
    if el is None:
        sys.exit(f"FAIL {docx_path.name}: Table 3 (header cell 'Axis') not found")
    ncol = len(t.columns)
    if ncol != len(FRACS):
        sys.exit(f"FAIL: expected {len(FRACS)} columns, found {ncol}")
    widths = [max(400, round(table_w * f)) for f in FRACS]
    widths[-1] += table_w - sum(widths)        # absorb rounding into the last column

    tblPr = el.find(qn("tblPr"))
    tblW = tblPr.find(qn("tblW"))
    tblW.set(qn("w"), str(table_w)); tblW.set(qn("type"), "dxa")
    layout = tblPr.find(qn("tblLayout"))
    if layout is None:
        from lxml import etree
        layout = etree.SubElement(tblPr, qn("tblLayout"))
    layout.set(qn("type"), "fixed")            # <- the fix: Word honours the column widths, no reflow

    grid = el.find(qn("tblGrid"))
    for c, w in zip(grid.findall(qn("gridCol")), widths):
        c.set(qn("w"), str(w))

    for ri, row in enumerate(t.rows):
        for ci, cell in enumerate(row.cells):
            cell.width = Twips(widths[ci])     # set each cell's tcW to match the grid
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.name = FONT
                    r.font.size = Pt(FONT_PT)
                    if ri == 0:
                        r.font.bold = True
    doc.save(str(docx_path))
    print(f"  {docx_path.name}: Table 3 -> fixed layout, {table_w} twips ({table_w/1440:.2f}in), "
          f"{FONT} {FONT_PT}pt; col widths {widths}")


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or [
        WD / "results" / "_paper" / "results_section.docx",
        WD.parent / "Toward Immune Virtual Cells (benchmark Results inserted).docx",
    ]
    for p in paths:
        if not p.exists():
            sys.exit(f"FAIL: docx not found: {p}")
        print(f"rebuilding Table 3 in {p}")
        rebuild(p)
    print("done.")
