"""Build the PDF RAG internship report from verified repository evidence."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

from figures import build_all

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "docs" / "demo-results"
FIG_DIR = Path(__file__).resolve().parent / "figures"
OUT = ROOT / "docs" / "PDF_RAG_INTERNSHIP_PROJECT_REPORT.docx"

TITLE = (
    "Development of an Web Crawler for Automated Knowledge Base Creation "
    "and RAG-Based Information Retrieval"
)

NAVY = RGBColor(0x1B, 0x3A, 0x4B)
TEAL = RGBColor(0x0B, 0x6E, 0x4F)
INK = RGBColor(0x1B, 0x24, 0x30)


class Report:
    def __init__(self) -> None:
        self.doc = Document()
        self.figures: list[str] = []
        self.tables: list[str] = []
        self._fig_paths = build_all(FIG_DIR)
        self._setup()

    def _setup(self) -> None:
        section = self.doc.sections[0]
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.top_margin = Cm(1.8)
        section.bottom_margin = Cm(1.8)
        section.left_margin = Cm(1.9)
        section.right_margin = Cm(1.9)
        section.header_distance = Cm(0.6)
        section.footer_distance = Cm(0.6)
        section.different_first_page_header_footer = True
        self.doc.core_properties.title = TITLE
        self.doc.core_properties.subject = "Internship project report"
        normal = self.doc.styles["Normal"]
        normal.font.name = "Calibri"
        normal.font.size = Pt(11)
        normal.font.color.rgb = INK
        normal.paragraph_format.space_after = Pt(8)
        normal.paragraph_format.space_before = Pt(0)
        normal.paragraph_format.line_spacing = 1.08
        for style_name, size, before, after in (
            ("Heading 1", 16, 16, 8),
            ("Heading 2", 13, 12, 4),
            ("Heading 3", 12, 10, 4),
        ):
            style = self.doc.styles[style_name]
            style.font.name = "Calibri"
            style.font.bold = True
            style.font.color.rgb = NAVY
            style.font.size = Pt(size)
            style.paragraph_format.space_before = Pt(before)
            style.paragraph_format.space_after = Pt(after)
            style.paragraph_format.keep_with_next = False
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer.add_run("C-DAC PDF Retrieval-Augmented Generation  ·  Internship project report  ·  ")
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x52, 0x60, 0x70)
        run.font.name = "Calibri"
        self._page_field(footer)
        header = section.header.paragraphs[0]
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = header.add_run("Vyom Desai  ·  C:\\CDAC\\RAG  ·  Version 1.0")
        hrun.font.size = Pt(9)
        hrun.font.color.rgb = TEAL
        hrun.font.name = "Calibri"

    def _page_field(self, paragraph) -> None:
        run = paragraph.add_run()
        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = " PAGE "
        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        run._r.append(begin)
        run._r.append(instr)
        run._r.append(end)
        run.font.size = Pt(9)
        run.font.name = "Calibri"

    def _shade(self, cell, color: str) -> None:
        tc = cell._tePr if hasattr(cell, "_tePr") else cell._tc
        tc_pr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), color)
        shd.set(qn("w:val"), "clear")
        tc_pr.append(shd)

    def h1(self, text: str) -> None:
        self.doc.add_heading(text, level=1)

    def h2(self, text: str) -> None:
        self.doc.add_heading(text, level=2)

    def h3(self, text: str) -> None:
        self.doc.add_heading(text, level=3)

    def p(self, text: str, *, bold: bool = False, italic: bool = False, center: bool = False, size: int = 11) -> None:
        paragraph = self.doc.add_paragraph()
        if center:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(text)
        run.bold = bold
        run.italic = italic
        run.font.name = "Calibri"
        run.font.size = Pt(size)
        run.font.color.rgb = INK

    def bullet(self, text: str) -> None:
        paragraph = self.doc.add_paragraph(style="List Bullet")
        paragraph.clear()
        run = paragraph.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(11)

    def code(self, text: str) -> None:
        paragraph = self.doc.add_paragraph()
        paragraph.paragraph_format.space_before = Pt(2)
        paragraph.paragraph_format.space_after = Pt(2)
        paragraph.paragraph_format.line_spacing = 1.0
        run = paragraph.add_run(text)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x1B, 0x2A, 0x38)

    def caption(self, text: str) -> None:
        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(2)
        paragraph.paragraph_format.space_after = Pt(10)
        run = paragraph.add_run(text)
        run.italic = True
        run.font.size = Pt(10)
        run.font.name = "Calibri"
        run.font.color.rgb = NAVY

    def figure(self, key: str, caption: str, width: float = 6.55) -> None:
        self._picture(self._fig_paths[key], caption, width)

    def image(self, path: Path, caption: str, width: float = 6.55) -> None:
        self._picture(path, caption, width)

    def _picture(self, path: Path, caption: str, width: float) -> None:
        """Place a picture and its caption inside one page.

        A4 body height is 26.1 cm. The picture is scaled down, keeping its
        aspect ratio, until the picture and a two-line caption fit in that
        body with a gap above the footer. The caption stays with the picture.
        """
        self.figures.append(caption)
        fitted = self._fitted_width(path, width)
        self.doc.add_picture(str(path), width=Inches(fitted))
        picture = self.doc.paragraphs[-1]
        picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
        picture.paragraph_format.space_before = Pt(6)
        picture.paragraph_format.space_after = Pt(0)
        picture.paragraph_format.line_spacing = 1.0
        picture.paragraph_format.keep_together = True
        picture.paragraph_format.keep_with_next = True
        self.caption(f"Figure {len(self.figures)}. {caption}")

    @staticmethod
    def _fitted_width(path: Path, width: float) -> float:
        # Body is 10.28 in. Leave room for the caption and the footer gap.
        max_height = 8.85
        with Image.open(path) as im:
            pixel_w, pixel_h = im.size
        if pixel_w <= 0 or pixel_h <= 0:
            return width
        height = width * pixel_h / pixel_w
        if height <= max_height:
            return width
        return max_height * pixel_w / pixel_h

    def table(self, headers: list[str], rows: list[list[str]], caption: str) -> None:
        self.tables.append(caption)
        self.caption(f"Table {len(self.tables)}. {caption}")
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.style = "Table Grid"
        table.autofit = True
        for index, header in enumerate(headers):
            cell = table.rows[0].cells[index]
            cell.text = ""
            paragraph = cell.paragraphs[0]
            run = paragraph.add_run(header)
            run.bold = True
            run.font.name = "Calibri"
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            self._shade(cell, "1B3A4B")
        for row_index, row in enumerate(rows):
            cells = table.add_row().cells
            for index, value in enumerate(row):
                cells[index].text = ""
                paragraph = cells[index].paragraphs[0]
                run = paragraph.add_run(str(value))
                run.font.name = "Calibri"
                run.font.size = Pt(9)
                if row_index % 2 == 1:
                    self._shade(cells[index], "F4F7F8")
        self.p("")

    def page_break(self) -> None:
        self.doc.add_page_break()

    def toc_field(self) -> None:
        paragraph = self.doc.add_paragraph()
        run = paragraph.add_run()
        begin = OxmlElement("w:fldChar")
        begin.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText")
        instr.set(qn("xml:space"), "preserve")
        instr.text = r' TOC \o "1-2" \h \z \u '
        separate = OxmlElement("w:fldChar")
        separate.set(qn("w:fldCharType"), "separate")
        placeholder = OxmlElement("w:t")
        placeholder.text = "Right-click and update this field in Word if the page numbers are not shown."
        end = OxmlElement("w:fldChar")
        end.set(qn("w:fldCharType"), "end")
        run._r.append(begin)
        run._r.append(instr)
        run2 = paragraph.add_run()
        run2._r.append(separate)
        run3 = paragraph.add_run()
        run3._r.append(placeholder)
        run4 = paragraph.add_run()
        run4._r.append(end)

    def save(self) -> None:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(OUT)


def main() -> None:
    from content import write_all

    report = Report()
    write_all(report)
    report.save()
    print(OUT)


def load_demo() -> list[dict]:
    rows = []
    for path in sorted(DEMO.glob("q*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def cyber_excerpt() -> str:
    path = ROOT / "data" / "extracted" / "pdf_82f38954477f2c53.json"
    if not path.exists():
        return ""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    target = "doc_cc7477d8bc498b35d746bfa461d023b3357533e1845706862ad12ef53c441bc4_chunk_002"
    for chunk in artifact.get("chunks", []):
        if chunk.get("chunk_id") == target:
            return (chunk.get("content") or "").strip()
    return ""


if __name__ == "__main__":
    main()
