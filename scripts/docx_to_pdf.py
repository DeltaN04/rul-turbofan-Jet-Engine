"""Render report.docx to a full PDF with fpdf2 (fallback when Word/LibreOffice fail).

Usage: python scripts/docx_to_pdf.py [input.docx] [output.pdf]
Parses python-docx body order: headings, normal text (bold/italic runs),
bullets, tables, inline images. A4, Helvetica (latin-1 sanitised).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt


def clean(t: str) -> str:
    out = []
    for ch in t:
        o = ord(ch)
        if o < 128 or ch in "éèêàâîôûç°µ²":
            out.append(ch)
        elif ch in "“”":
            out.append('"')
        elif ch in "‘’":
            out.append("'")
        elif ch in "—–":
            out.append("-")
        elif ch in "≥":
            out.append(">=")
        elif ch in "≤":
            out.append("<=")
        elif ch in "→":
            out.append("->")
        elif ch in "×":
            out.append("x")
        else:
            out.append("?")
    return "".join(out)


def main(inp: str = "report/report.docx", out: str = "report/report_full.pdf"):
    from fpdf import FPDF

    doc = Document(inp)
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, 20)
    pdf.set_margins(20, 15, 20)
    pdf.add_page()

    def mc(w, h, txt):
        pdf.multi_cell(w, h, txt)
        pdf.set_x(pdf.l_margin)

    # map inline images in order
    images = []
    for sh in doc.inline_shapes:
        try:
            rId = sh._inline.graphic.graphicData.pic.blipFill.blip.embed
            part = doc.part.related_parts[rId]
            images.append(part.blob)
        except Exception:
            pass
    img_iter = iter(images)
    import tempfile
    tmpdir = Path(tempfile.mkdtemp())

    def style_of(p):
        return p.style.name if p.style else "Normal"

    # walk body elements in order
    body = doc.element.body
    for el in body:
        tag = el.tag.split("}")[-1]
        if tag == "p":
            from docx.text.paragraph import Paragraph as _P
            p = _P(el, doc)
            txt = clean(p.text).strip()
            st = style_of(p)
            has_img = bool(el.findall(f".//{{http://schemas.openxmlformats.org/wordprocessingml/2006/main}}drawing"))
            if has_img:
                try:
                    blob = next(img_iter)
                    fp = tmpdir / f"img{len(list(tmpdir.glob('*')))}.png"
                    fp.write_bytes(blob)
                    if pdf.get_y() > 230:
                        pdf.add_page()
                    pdf.image(str(fp), x=25, w=160)
                    pdf.ln(2)
                except StopIteration:
                    pass
                continue
            if not txt:
                pdf.ln(3)
                continue
            if st.startswith("Heading 1"):
                pdf.add_page() if pdf.get_y() > 40 else None
                pdf.set_font("Helvetica", "B", 15)
                pdf.set_text_color(30, 80, 160)
                mc(0, 8, txt)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(2)
            elif st.startswith("Heading 2"):
                pdf.set_font("Helvetica", "B", 12)
                pdf.set_text_color(30, 80, 160)
                mc(0, 7, txt)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(1)
            elif "List" in st or (txt.startswith(("•", "-", "*")) and len(txt) < 400):
                pdf.set_font("Helvetica", "", 9.5)
                mc(0, 5, "  - " + txt.lstrip("•-* "))
            else:
                # body: honour bold/italic lead runs approximately
                if p.runs and p.runs[0].bold and len(txt) < 220:
                    pdf.set_font("Helvetica", "B", 9.5)
                elif p.runs and p.runs[0].italic:
                    pdf.set_font("Helvetica", "I", 9)
                else:
                    pdf.set_font("Helvetica", "", 9.5)
                mc(0, 5, txt)
                pdf.ln(1)
        elif tag == "tbl":
            from docx.table import Table as _T
            t = _T(el, doc)
            rows = [[clean(c.text).strip()[:160] for c in r.cells] for r in t.rows]
            if not rows:
                continue
            pdf.set_font("Helvetica", "", 7)
            try:
                with pdf.table(width=170, text_align="LEFT", line_height=5) as table:
                    for ri, row in enumerate(rows):
                        tr = table.row()
                        for cell in row:
                            tr.cell(cell)
            except Exception:
                for row in rows:
                    mc(0, 5, " | ".join(row)[:220])
            pdf.ln(3)
    pdf.output(out)
    print("wrote", out)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "report/report.docx"
    b = sys.argv[2] if len(sys.argv) > 2 else "report/report_full.pdf"
    main(a, b)
