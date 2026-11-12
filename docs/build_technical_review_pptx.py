"""Technical review deck for the C-DAC unified RAG internship.

Facts come from C:\\CDAC\\RAG: the internship report, settings, CLI, and
docs/demo-results. They are not taken from the separate pdf-rag review deck.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

OUT = Path(r"C:\CDAC\RAG\docs\CDAC_Unified_RAG_Internship_Review_Vyom_Desai.pptx")
SHOT = Path(r"C:\CDAC\RAG\docs\demo-results\screenshots\q5-cyber-forensics.png")

NAVY = RGBColor(0x0C, 0x23, 0x40)
BLUE = RGBColor(0x1F, 0x5C, 0x99)
INK = RGBColor(0x1A, 0x23, 0x32)
MUTED = RGBColor(0x5C, 0x6B, 0x7A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xD3, 0xDC, 0xE6)
PANEL = RGBColor(0xF4, 0xF7, 0xFB)
GREEN = RGBColor(0x1B, 0x6B, 0x45)
GREEN_BG = RGBColor(0xE6, 0xF5, 0xEC)
AMBER = RGBColor(0x8A, 0x5A, 0x10)
AMBER_BG = RGBColor(0xFB, 0xF3, 0xE4)
SLATE = RGBColor(0x3E, 0x4C, 0x5E)
SLATE_BG = RGBColor(0xEE, 0xF2, 0xF6)
RED = RGBColor(0x8C, 0x3A, 0x3A)
RED_BG = RGBColor(0xF8, 0xEE, 0xEE)

STATUS = {
    "DONE": (GREEN, GREEN_BG),
    "PARTIAL": (AMBER, AMBER_BG),
    "PENDING": (SLATE, SLATE_BG),
    "NEEDS LIVE VALIDATION": (BLUE, RGBColor(0xE7, 0xF0, 0xF8)),
}

W, H, TOTAL = 13.333, 7.5, 26


def add_shape(slide, kind, x, y, w, h, fill, line=None, line_pt=1.0, radius=None):
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_pt)
    if radius is not None and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            shape.adjustments[0] = radius
        except Exception:
            pass
    effect = shape._element.spPr.find(qn("a:effectLst"))
    if effect is not None:
        shape._element.spPr.remove(effect)
    return shape


def rrect(slide, x, y, w, h, fill, line=None, radius=0.08):
    return add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, line, radius=radius)


def rect(slide, x, y, w, h, fill):
    return add_shape(slide, MSO_SHAPE.RECTANGLE, x, y, w, h, fill)


def oval(slide, x, y, w, h, fill):
    return add_shape(slide, MSO_SHAPE.OVAL, x, y, w, h, fill)


def fill_text(shape, blocks, anchor=MSO_ANCHOR.TOP, ml=0.08, mr=0.08, mt=0.05, mb=0.04):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(ml)
    tf.margin_right = Inches(mr)
    tf.margin_top = Inches(mt)
    tf.margin_bottom = Inches(mb)
    for i, block in enumerate(blocks):
        text, size, bold, color = block[:4]
        align = block[4] if len(block) > 4 else PP_ALIGN.LEFT
        before = block[5] if len(block) > 5 else 0
        font = block[6] if len(block) > 6 else "Calibri"
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_before = Pt(before)
        p.space_after = Pt(0)
        run = p.add_run()
        run.text = text
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return shape


def tb(slide, x, y, w, h, text, size=12, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    fill_text(box, [(text, size, bold, color, align)], anchor=anchor, ml=0.02, mr=0.02, mt=0, mb=0)
    return box


def pill(slide, x, y, w, h, label):
    fg, bg = STATUS[label]
    shape = rrect(slide, x, y, w, h, bg, radius=0.5)
    fill_text(shape, [(label, 9, True, fg, PP_ALIGN.CENTER)], anchor=MSO_ANCHOR.MIDDLE, ml=0.03, mr=0.03, mt=0, mb=0)


def footer(slide, page):
    rect(slide, 0, 7.18, W, 0.32, NAVY)
    tb(slide, 0.28, 7.20, 10.6, 0.26,
       "Vyom Desai   ·   C-DAC Pune   ·   15 June 2026 – 15 September 2026",
       10, False, WHITE, anchor=MSO_ANCHOR.MIDDLE)
    tb(slide, 11.2, 7.20, 1.85, 0.26, f"{page:02d}   /   {TOTAL:02d}",
       10, True, WHITE, PP_ALIGN.RIGHT, MSO_ANCHOR.MIDDLE)


def header(slide, title, page, status=None):
    rect(slide, 0, 0, W, 0.66, NAVY)
    rect(slide, 0, 0.66, W, 0.05, BLUE)
    tb(slide, 0.30, 0.08, 9.4 if status else 12.6, 0.50, title, 20, True, WHITE, anchor=MSO_ANCHOR.MIDDLE)
    if status:
        labels = status if isinstance(status, (list, tuple)) else [status]
        x = 13.02
        for label in reversed(labels):
            width = min(2.55, 0.16 + 0.075 * len(label))
            x -= width
            pill(slide, x, 0.17, width, 0.32, label)
            x -= 0.08
    footer(slide, page)


def new(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def hflow(slide, items, x, y, w, h, fill, color, line=None):
    n = len(items)
    gap = 0.16
    bw = (w - gap * (n - 1)) / n
    for i, item in enumerate(items):
        bx = x + i * (bw + gap)
        shape = rrect(slide, bx, y, bw, h, fill, line, radius=0.12)
        lines = item if isinstance(item, (list, tuple)) else [item]
        blocks = [(line_text, 10 if j else 12, True, color, PP_ALIGN.CENTER, 1 if j else 0) for j, line_text in enumerate(lines)]
        fill_text(shape, blocks, anchor=MSO_ANCHOR.MIDDLE, ml=0.04, mr=0.04)
        if i < n - 1:
            tb(slide, bx + bw - 0.02, y, gap + 0.04, h, "›", 16, True, BLUE, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)


def set_cell_border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
    for tag in ("lnL", "lnR", "lnT", "lnB"):
        for child in list(tcPr):
            if child.tag == f"{{{ns}}}{tag}":
                tcPr.remove(child)
        ln = etree.SubElement(tcPr, f"{{{ns}}}{tag}")
        ln.set("w", "7620")
        sf = etree.SubElement(ln, f"{{{ns}}}solidFill")
        c = etree.SubElement(sf, f"{{{ns}}}srgbClr")
        c.set("val", "D3DCE6")


def style_cell(cell, text, size, bold, color, fill, align=PP_ALIGN.LEFT):
    cell.text = ""
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Inches(0.06)
    cell.margin_right = Inches(0.04)
    cell.margin_top = Inches(0.02)
    cell.margin_bottom = Inches(0.02)
    p = cell.text_frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = "Calibri"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    cell.text_frame.word_wrap = True
    set_cell_border(cell)


def add_table(slide, rows, x, y, w, h, widths, font=11):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    table = shape.table
    for i, width in enumerate(widths):
        table.columns[i].width = Inches(width)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if r == 0:
                style_cell(table.cell(r, c), value, font, True, WHITE, NAVY)
            else:
                style_cell(table.cell(r, c), value, font, False, INK, WHITE if r % 2 else PANEL)
    return shape


def metric(slide, x, y, w, h, value, label):
    card = rrect(slide, x, y, w, h, WHITE, LINE, radius=0.1)
    fill_text(card, [(value, 26, True, NAVY, PP_ALIGN.CENTER), (label, 11, False, MUTED, PP_ALIGN.CENTER, 1)],
              anchor=MSO_ANCHOR.MIDDLE, ml=0.04, mr=0.04)


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def s01(prs):
    s = new(prs)
    rect(s, 0, 0, 4.7, H, NAVY)
    rect(s, 4.7, 0, 0.07, H, BLUE)
    tb(s, 0.32, 0.32, 4.1, 0.24, "CENTRE FOR DEVELOPMENT OF ADVANCED COMPUTING", 9, True, RGBColor(0x9F, 0xC4, 0xE8))
    tb(s, 0.32, 0.58, 4.1, 0.36, "C-DAC, Pune", 20, True, WHITE)
    tb(s, 0.32, 1.02, 4.1, 0.28, "Internship project review", 13, False, RGBColor(0xC5, 0xD7, 0xE8))
    facts = [
        ("Period", "15 June 2026 – 15 September 2026"),
        ("Duration", "3 months"),
        ("Student", "Vyom Desai"),
        ("Guide", "Pavan Dhote"),
        ("Application", "C-DAC Unified RAG"),
        ("Repository", r"C:\CDAC\RAG"),
    ]
    for i, (k, v) in enumerate(facts):
        y = 1.65 + i * 0.72
        tb(s, 0.36, y, 4.0, 0.22, k.upper(), 10, True, RGBColor(0x9F, 0xC4, 0xE8))
        tb(s, 0.36, y + 0.22, 4.0, 0.32, v, 14, True, WHITE)
    tb(s, 5.05, 0.38, 7.9, 0.24, "INTERNSHIP PROJECT", 12, True, BLUE)
    tb(s, 5.05, 0.72, 7.9, 2.15,
       "Development of an Web Crawler\nfor Automated Knowledge Base\nCreation and RAG-Based\nInformation Retrieval",
       26, True, NAVY)
    tb(s, 5.05, 3.05, 7.8, 0.7,
       "Page-level provenance, hybrid BM25 and vector retrieval, and evidence-grounded answers over the crawled C-DAC corpus and its PDFs.",
       14, False, MUTED)
    chips = [
        ("DONE", "PDF path and live index"),
        ("PARTIAL", "OCR on 31 sparse pages"),
        ("PENDING", "Labelled retrieval score"),
    ]
    for i, (label, caption) in enumerate(chips):
        x = 5.05 + i * 2.65
        pill(s, x, 3.95, 2.45, 0.32, label)
        tb(s, x, 4.32, 2.5, 0.4, caption, 11, False, MUTED, PP_ALIGN.CENTER)
    note = rrect(s, 5.05, 4.9, 7.85, 1.85, PANEL, radius=0.08)
    fill_text(note, [
        ("RECORDED IN THIS REPOSITORY", 11, True, BLUE),
        ("Index rag_corpus_v1 on Elasticsearch 8.12.2 holds 12,054 documents.", 13, False, INK, PP_ALIGN.LEFT, 4),
        ("60 crawled PDFs, 1,479 pages, 2,880 text chunks.", 13, False, INK, PP_ALIGN.LEFT, 2),
        ("Five saved answers from Llama 3.2 through Ollama. None abstained.", 13, False, INK, PP_ALIGN.LEFT, 2),
    ], ml=0.14, mt=0.1)


def s02(prs):
    s = new(prs)
    header(s, "Problem statement", 2)
    banner = rrect(s, 0.32, 0.88, 12.69, 1.05, NAVY, radius=0.08)
    fill_text(banner, [
        ("ENGINEERING PROBLEM", 10, True, RGBColor(0x9F, 0xC4, 0xE8)),
        ("Given crawled pages and PDFs, answer a natural-language question from those sources, name the supporting passage, and abstain when the evidence is too weak.",
         15, True, WHITE, PP_ALIGN.LEFT, 3),
    ], ml=0.16, mt=0.08)
    items = [
        ("01", "A general model", "A language model can write fluently and still miss a fact that exists only in a PDF acquired after training."),
        ("02", "Long documents", "A full PDF does not fit a context window on every question. The retrievable unit has to be a passage."),
        ("03", "Messy page text", "pypdf text carries hyphenation, control characters, and running headers that should not become search terms."),
        ("04", "Sparse pages", "A scanned page yields almost no characters. The code detects that and can call Tesseract when the binary is present."),
        ("05", "Two retrieval failures", "BM25 misses a paraphrase. A vector search can miss a rare identifier. Their raw scores are not on one scale."),
        ("06", "Unsupported answers", "A fluent completion can still go beyond the hits. The ask path checks hit count, a minimum score, and role strings first."),
        ("07", "No audit trail", "An answer without a chunk id and a page cannot be checked. Those fields are stored on the text chunk and copied into the source list."),
        ("08", "Mixed corpus", "The live index holds HTML and crawled PDFs together. A PDF-only question has to set --sources pdf."),
    ]
    for i, (n, title, body) in enumerate(items):
        col, row = i % 4, i // 4
        x, y = 0.32 + col * 3.20, 2.10 + row * 2.45
        card = rrect(s, x, y, 3.06, 2.30, WHITE, LINE, radius=0.08)
        oval(s, x + 0.12, y + 0.12, 0.34, 0.34, RGBColor(0xE7, 0xF0, 0xF8))
        tb(s, x + 0.12, y + 0.12, 0.34, 0.34, n, 10, True, BLUE, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
        tb(s, x + 0.54, y + 0.12, 2.35, 0.34, title, 13, True, NAVY, anchor=MSO_ANCHOR.MIDDLE)
        tb(s, x + 0.12, y + 0.56, 2.8, 1.6, body, 12, False, INK)


def s03(prs):
    s = new(prs)
    header(s, "Objectives and scope", 3)
    tb(s, 0.32, 0.82, 12.6, 0.28, "Status labels are the ones used in the internship report: DONE, PARTIAL, and PENDING.", 13, False, MUTED)
    rows = [
        ["Objective", "Status", "Evidence in this repository"],
        ["Accept a PDF and reject a missing or empty file", "DONE", "PDFProcessor.process_pdf"],
        ["Identify the file by SHA-256 of its bytes", "DONE", "generate_document_id"],
        ["Extract and clean page text", "DONE", "1,479 stored pages; 0 failed pages"],
        ["Detect sparse pages and run OCR", "PARTIAL", "31 pages flagged; Tesseract was not on PATH"],
        ["Turn embedded images into chunks", "PARTIAL", "Code exists; image_chunks_generated summed to 0"],
        ["Heading-aware chunks with overlap and merge", "DONE", "2,880 PDF chunks; cyber PDF had 3 merges"],
        ["Page numbers on text chunks", "DONE", "Cyber PDF hit lists page 1"],
        ["Page numbers on summary records", "PARTIAL", "summary_to_indexable leaves the page list empty"],
        ["384-d embeddings and Elasticsearch", "DONE", "rag_corpus_v1, 12,054 documents"],
        ["BM25, cosine kNN, min-max hybrid 0.3/0.7", "DONE", "Default mode; five saved answers are hybrid"],
        ["Abstain and return a source list", "DONE", "Ask path; the five demos did not abstain"],
        ["Score a labelled set with nDCG", "PENDING", "metrics.py has hit rate, recall, and MRR only"],
    ]
    add_table(s, rows, 0.28, 1.16, 12.76, 5.85, [5.3, 1.35, 6.11], font=11)


def s04(prs):
    s = new(prs)
    header(s, "System architecture", 4, "DONE")
    tb(s, 0.32, 0.82, 12.6, 0.24, "ACQUISITION", 11, True, BLUE)
    hflow(s, [["1", "Crawl"], ["2", "HTML"], ["3", "PDF download"], ["4", "Extract"], ["5", "Chunk"]],
          0.32, 1.08, 12.69, 0.72, NAVY, WHITE)
    tb(s, 0.32, 1.92, 12.6, 0.22, "INDEX AND ANSWER", 11, True, BLUE)
    hflow(s, [["6", "Embed"], ["7", "Index"], ["8", "BM25"], ["9", "HNSW"], ["10", "Fusion"], ["11", "Gate"], ["12", "Answer"]],
          0.32, 2.16, 12.69, 0.72, WHITE, NAVY, NAVY)
    cards = [
        ("Crawl", "Scrapy spider cdac starts at https://cdac.gov.in/, stays on that domain, and writes HTML sections plus a PDF manifest."),
        ("PDF path", "ingest_uploaded_pdf, process-pdfs, and PDFRAGPipeline.ingest_pdf share pypdf extraction, cleaning, and structural chunking."),
        ("Answer path", "search_corpus, then grounding_failure, then the LLM at temperature 0. RAGAnswer carries the source list."),
    ]
    for i, (title, body) in enumerate(cards):
        card = rrect(s, 0.32 + i * 4.26, 3.08, 4.12, 1.45, PANEL, radius=0.08)
        fill_text(card, [(title, 14, True, NAVY), (body, 12, False, INK, PP_ALIGN.LEFT, 3)], ml=0.12, mt=0.08)
    note = rrect(s, 0.32, 4.68, 12.69, 2.32, WHITE, LINE, radius=0.08)
    fill_text(note, [
        ("WHAT THE LIVE INDEX ACTUALLY HOLDS", 12, True, BLUE),
        ("kb.source is all, so rag_corpus_v1 mixes HTML and crawled PDFs. Of 12,054 documents, 6,076 are crawled_pdf and 5,978 are html.", 14, False, INK, PP_ALIGN.LEFT, 4),
        ("The PDF portion is 2,880 chunks + 2,880 extractive summaries + 316 composite summaries. Uploaded-PDF artifacts in the scan: 0.", 14, False, INK, PP_ALIGN.LEFT, 3),
        ("Knowledge-graph rows are added at question time when --include-kg is set. The five saved answers each have 5 hybrid hits and 5 graph hits.", 14, False, INK, PP_ALIGN.LEFT, 3),
    ], ml=0.16, mt=0.1)


def s05(prs):
    s = new(prs)
    header(s, "Technology stack", 5)
    tb(s, 0.32, 0.80, 12.6, 0.26, "Versions were read from this project’s environment and from the cluster at localhost:9200.", 12, False, MUTED)
    rows = [
        ["Technology", "Version observed", "Role in this repository"],
        ["Python", "3.12.10", "Runtime; Docker image python:3.12-slim-bookworm"],
        ["Scrapy", "Project spider cdac", "Bounded crawl of cdac.gov.in"],
        ["pypdf", "5.9.0", "Page text and PDF metadata"],
        ["PyMuPDF", "2.1", "Page render and embedded images; import name fitz"],
        ["pytesseract / Tesseract", "0.3.13 / binary not on PATH", "OCR only when the binary is present"],
        ["Pillow", "13.0.0", "Open rendered page images"],
        ["sentence-transformers", "3.4.1", "Local all-MiniLM-L6-v2, 384 dimensions"],
        ["Elasticsearch", "server 8.12.2; client 8.19.3", "BM25, kNN, and storage"],
        ["Pydantic / PyYAML / dotenv", "2.14.3 / 6.0.3 / 1.2.4", "Settings, models, and masked secrets"],
        ["pytest", "8.4.2", "Unit and contract tests"],
        ["Ollama model in the demo", "Llama 3.2", "The five stored answers; mock remains the default"],
    ]
    add_table(s, rows, 0.28, 1.12, 12.76, 5.9, [3.3, 3.5, 5.96], font=12)


def s06(prs):
    s = new(prs)
    header(s, "Configuration", 6, "DONE")
    left = [
        ("File", "config/settings.yaml, then environment variables by upper-case field name."),
        ("Secrets", "LLM_API_KEY and ELASTICSEARCH_PASSWORD are SecretStr. get_masked_dict hides them."),
        ("Index", "http://localhost:9200, index rag_corpus_v1."),
        ("Embeddings", "provider local, sentence-transformers/all-MiniLM-L6-v2, dimension 384."),
        ("LLM default", "provider mock. A live call needs LLM_ENDPOINT. The saved demo used Llama 3.2."),
    ]
    for i, (title, body) in enumerate(left):
        card = rrect(s, 0.32, 0.86 + i * 1.02, 6.35, 0.94, WHITE, LINE, radius=0.08)
        fill_text(card, [(title, 13, True, NAVY), (body, 12, False, INK, PP_ALIGN.LEFT, 2)], ml=0.12, mt=0.08)
    right = rrect(s, 6.85, 0.86, 6.16, 5.95, PANEL, radius=0.08)
    fill_text(right, [
        ("RETRIEVAL AND CHUNK VALUES", 12, True, BLUE),
        ("mode hybrid, top_k 5, pool 20, num_candidates 50", 14, False, INK, PP_ALIGN.LEFT, 8),
        ("keyword_weight 0.3, semantic_weight 0.7, must sum to 1", 14, False, INK, PP_ALIGN.LEFT, 4),
        ("fusion_strategy min_max; rrf_k 60 is unused until strategy is rrf", 14, False, INK, PP_ALIGN.LEFT, 4),
        ("figure boost 4.0, image boost 2.0", 14, False, INK, PP_ALIGN.LEFT, 4),
        ("grounding_min_score 0.35", 14, False, INK, PP_ALIGN.LEFT, 4),
        ("HTML chunks: 1200 / overlap 150 / min 80", 14, False, INK, PP_ALIGN.LEFT, 8),
        ("PDF chunks: 1200 / overlap 100 / min 300", 14, False, INK, PP_ALIGN.LEFT, 4),
        ("Near-duplicate threshold 0.92", 14, False, INK, PP_ALIGN.LEFT, 4),
        ("Crawl: cdac.gov.in, max 500 URLs, delay 1.5 s, 3 retries", 14, False, INK, PP_ALIGN.LEFT, 4),
    ], ml=0.16, mt=0.12)


def s07(prs):
    s = new(prs)
    header(s, "Web crawl and PDF acquisition", 7, "DONE")
    steps = [
        ("1", "Seed", "python run.py crawl starts Scrapy spider cdac at https://cdac.gov.in/."),
        ("2", "Boundary", "allowed domain cdac.gov.in, URL cap 500, response cap 5 MB, download delay 1.5 seconds."),
        ("3", "HTML", "Sections are extracted and chunked. The page itself is not indexed as raw HTML."),
        ("4", "PDFs", "Discovered PDFs are downloaded to data/pdfs with a manifest, then process-pdfs builds artifacts."),
        ("5", "Retries", "A failed fetch is retried. The crawl ledger and success.log / error.log record the outcome."),
        ("6", "Index", "python run.py index embeds the selected records and bulk-writes rag_corpus_v1."),
    ]
    for i, (n, title, body) in enumerate(steps):
        col, row = i % 3, i // 3
        card = rrect(s, 0.32 + col * 4.26, 0.88 + row * 2.15, 4.12, 2.02, WHITE, LINE, radius=0.08)
        fill_text(card, [(f"{n}   {title}", 16, True, NAVY), (body, 13, False, INK, PP_ALIGN.LEFT, 6)], ml=0.14, mt=0.12)
    bar = rrect(s, 0.32, 5.28, 12.69, 1.72, NAVY, radius=0.08)
    fill_text(bar, [
        ("WHAT THE STORED PDF CORPUS IS", 12, True, RGBColor(0x9F, 0xC4, 0xE8)),
        ("All 60 PDF artifacts have source_kind crawled_pdf. Uploaded-PDF artifacts in that scan: 0. The files live under data/pdfs, including 82f38954477f2c53.pdf, the cyber-forensics document used in the walkthrough.",
         14, False, WHITE, PP_ALIGN.LEFT, 4),
        ("A separate ingest --pdf command copies one local file into data/raw and marks it uploaded_pdf. That path was not the source of the 60-file corpus.",
         14, False, WHITE, PP_ALIGN.LEFT, 3),
    ], ml=0.16, mt=0.1)


def s08(prs):
    s = new(prs)
    header(s, "PDF extraction and cleaning", 8, "DONE")
    flow = [
        ("Bytes", "The file is hashed with SHA-256 before pages are read. The same bytes always produce the same document_id."),
        ("Pages", "pypdf PdfReader.pages[i].extract_text() supplies the raw string. Page numbers stay 1-based."),
        ("Cleaning", "clean_extracted_text repairs hyphenation and whitespace. remove_repeated_boilerplate drops running headers."),
        ("Blocks", "detect_block_type separates headings from paragraphs. The first section name, before any heading, is Introduction."),
        ("Failure", "A page that cannot be read is stored as failed and the rest of the file continues. Across 60 PDFs, failed_pages is 0."),
        ("Record", "The cyber PDF cc7477d8… opened 25 pages, 95 blocks, in 0.700 seconds of stored extraction time."),
    ]
    for i, (title, body) in enumerate(flow):
        col, row = i % 3, i // 3
        card = rrect(s, 0.32 + col * 4.26, 0.88 + row * 2.05, 4.12, 1.92, WHITE, LINE, radius=0.08)
        fill_text(card, [(title, 16, True, NAVY), (body, 13, False, INK, PP_ALIGN.LEFT, 4)], ml=0.14, mt=0.1)
    foot = rrect(s, 0.32, 5.08, 12.69, 1.92, PANEL, radius=0.08)
    fill_text(foot, [
        ("CORPUS EXTRACTION, SUMMED FROM THE 60 ARTIFACTS", 12, True, NAVY),
        ("1,479 pages. 1,448 counted successfully extracted. 0 failed. 31 still flagged requires_ocr. Every document’s extraction_status is SUCCESS, including pages left IMAGE_ONLY.",
         14, False, INK, PP_ALIGN.LEFT, 4),
        ("Sum of stored extraction durations: 44.38 seconds. Mean 0.74 seconds. Maximum 3.165 seconds. That sum is the artifact field, not a new stopwatch run.",
         14, False, INK, PP_ALIGN.LEFT, 3),
    ], ml=0.16, mt=0.1)


def s09(prs):
    s = new(prs)
    header(s, "OCR and embedded images", 9, "PARTIAL")
    left = rrect(s, 0.32, 0.88, 6.28, 4.05, WHITE, LINE, radius=0.08)
    right = rrect(s, 6.76, 0.88, 6.25, 4.05, AMBER_BG, radius=0.08)
    fill_text(left, [
        ("WHAT THE CODE DOES", 13, True, NAVY),
        ("A page with very little extracted text sets requires_ocr.", 14, False, INK, PP_ALIGN.LEFT, 8),
        ("If Tesseract and PyMuPDF are available, the page is rendered at 300 dpi and recognised.", 14, False, INK, PP_ALIGN.LEFT, 6),
        ("Embedded rasters at least 50 px are saved under data/extracted/images.", 14, False, INK, PP_ALIGN.LEFT, 6),
        ("OCR text of at least 15 characters can become an image chunk.", 14, False, INK, PP_ALIGN.LEFT, 6),
        ("If the binary is missing, the processor logs a warning and leaves the sparse page unchanged.", 14, False, INK, PP_ALIGN.LEFT, 6),
    ], ml=0.16, mt=0.12)
    fill_text(right, [
        ("WHAT THIS CORPUS SHOWS", 13, True, AMBER),
        ("31 pages remain flagged as requiring OCR.", 15, False, INK, PP_ALIGN.LEFT, 8),
        ("The test re-run found the Tesseract binary not on PATH.", 15, False, INK, PP_ALIGN.LEFT, 6),
        ("image_chunks_generated across the 60 PDFs is 0.", 15, False, INK, PP_ALIGN.LEFT, 6),
        ("The Docker image installs tesseract-ocr. The host run that produced these artifacts did not use it.", 15, False, INK, PP_ALIGN.LEFT, 6),
        ("No screenshot-answer from this corpus is claimed.", 15, True, NAVY, PP_ALIGN.LEFT, 6),
    ], ml=0.16, mt=0.12)
    bar = rrect(s, 0.32, 5.10, 12.69, 1.90, NAVY, radius=0.08)
    fill_text(bar, [
        ("STATUS", 12, True, RGBColor(0x9F, 0xC4, 0xE8)),
        ("OCR decision and fallback: PARTIAL. Image extraction to chunks: PARTIAL. Tests cover the graceful path when OCR cannot run. They do not certify recognition of the 31 sparse pages.",
         15, False, WHITE, PP_ALIGN.LEFT, 6),
    ], ml=0.16, mt=0.12)


def s10(prs):
    s = new(prs)
    header(s, "Context-aware chunking", 10, "DONE")
    code = rrect(s, 0.32, 0.86, 6.15, 3.55, NAVY, radius=0.08)
    fill_text(code, [
        ("STRUCTURAL RULE", 11, True, RGBColor(0x9F, 0xC4, 0xE8)),
        ("section := Introduction", 13, False, WHITE, PP_ALIGN.LEFT, 6, "Consolas"),
        ("a heading closes the group and names the next section", 13, False, WHITE, PP_ALIGN.LEFT, 2, "Consolas"),
        ("emit when the next block would pass 1,200 characters", 13, False, WHITE, PP_ALIGN.LEFT, 2, "Consolas"),
        ("carry the last 100 characters, trimmed to a word", 13, False, WHITE, PP_ALIGN.LEFT, 2, "Consolas"),
        ("reject near-duplicates at similarity 0.92", 13, False, WHITE, PP_ALIGN.LEFT, 2, "Consolas"),
        ("merge same-section chunks under 300 characters", 13, False, WHITE, PP_ALIGN.LEFT, 2, "Consolas"),
        ("ceiling is 1.3 × 1,200 = 1,560 characters", 13, False, WHITE, PP_ALIGN.LEFT, 2, "Consolas"),
    ], ml=0.14, mt=0.1)
    right = rrect(s, 6.64, 0.86, 6.37, 3.55, PANEL, radius=0.08)
    fill_text(right, [
        ("WORKED PDF, NOT A SYNTHETIC EXAMPLE", 12, True, NAVY),
        ("File 82f38954477f2c53.pdf", 14, True, INK, PP_ALIGN.LEFT, 6),
        ("25 pages, 95 blocks in.", 14, False, INK, PP_ALIGN.LEFT, 3),
        ("61 chunks out. Mean length 842.", 14, False, INK, PP_ALIGN.LEFT, 2),
        ("Minimum 89, maximum 1,339.", 14, False, INK, PP_ALIGN.LEFT, 2),
        ("3 merges, 0 duplicates, 0 LLM calls.", 14, False, INK, PP_ALIGN.LEFT, 2),
        ("Chunking time stored: 0.304 seconds.", 14, False, INK, PP_ALIGN.LEFT, 2),
        ("The maximum above 1,200 is the merge ceiling.", 14, False, INK, PP_ALIGN.LEFT, 4),
    ], ml=0.16, mt=0.1)
    wide = rrect(s, 0.32, 4.56, 12.69, 2.44, WHITE, LINE, radius=0.08)
    fill_text(wide, [
        ("ALL 60 PDFS", 12, True, BLUE),
        ("Text-chunk lengths: minimum 49, median 1,073, mean 933.6, maximum 1,481 characters. Sum of stored chunking durations: 15.73 seconds.", 14, False, INK, PP_ALIGN.LEFT, 4),
        ("HTML uses a different setting: max 1,200, overlap 150, minimum 80, composite group size 10. LLM-assisted organisation exists in code and is off. The cyber artifact records llm_calls_made as 0.", 14, False, INK, PP_ALIGN.LEFT, 4),
        ("Limitation recorded in the report: the heading detector can fire on a short line that is not a heading, and the overlap is a character tail.", 14, False, INK, PP_ALIGN.LEFT, 4),
    ], ml=0.16, mt=0.1)


def s11(prs):
    s = new(prs)
    header(s, "Metadata, summaries, and lineage", 11)
    rows = [
        ["Field", "Meaning", "Where it is set"],
        ["document_id", "SHA-256 of the PDF bytes", "generate_document_id"],
        ["chunk_id", "doc_<id>_chunk_<nnn>", "Chunker, renumbered after merge"],
        ["page_numbers", "Pages of the blocks in the group", "Text chunks only"],
        ["section", "Heading that owned the group", "Chunker"],
        ["source_kind", "html, crawled_pdf, or uploaded_pdf", "Ingestion"],
        ["content_kind", "chunk, summary, or composite", "Indexer"],
        ["source_url, depth", "Crawl lineage for downloaded files", "PDF manifest and HTML extractor"],
    ]
    add_table(s, rows, 0.28, 0.88, 12.76, 4.15, [2.6, 5.3, 4.86], font=13)
    a = rrect(s, 0.32, 5.18, 6.22, 1.82, AMBER_BG, radius=0.08)
    b = rrect(s, 6.70, 5.18, 6.31, 1.82, PANEL, radius=0.08)
    fill_text(a, [
        ("SUMMARY HITS HAVE NO PAGE", 12, True, AMBER),
        ("Each PDF chunk also has an extractive summary. The demo summary that matched chunk 002 stored an empty page list, at the same fused score.", 13, False, INK, PP_ALIGN.LEFT, 3),
    ], ml=0.12, mt=0.08)
    fill_text(b, [
        ("LINEAGE ON THE CYBER PDF HIT", 12, True, NAVY),
        ("pdf id 82f38954477f2c53, parent page the tenders EOI listing, depth 2, downloaded 11 November 2026.", 13, False, INK, PP_ALIGN.LEFT, 3),
    ], ml=0.12, mt=0.08)


def s12(prs):
    s = new(prs)
    header(s, "Embedding generation", 12, "DONE")
    hflow(s, [["Chunk text"], ["all-MiniLM-L6-v2"], ["384 floats"], ["dense_vector"]],
          0.32, 0.90, 12.69, 0.78, NAVY, WHITE)
    cards = [
        ("One model both ways", "Queries and documents use the active provider. The active provider is local. Mock and custom HTTP providers exist and are not selected."),
        ("Dimension check", "The encoder’s width must equal embedding_dimension, which is 384. A mismatch is rejected before indexing."),
        ("Cosine", "The code does not set normalize_embeddings. Cosine is declared on the Elasticsearch field, and the cluster normalises for that similarity."),
        ("What is not stored", "The cyber walkthrough says the vector is 384 floats in the index. The vector itself is not printed in the report or the demo JSON."),
    ]
    for i, (title, body) in enumerate(cards):
        col, row = i % 2, i // 2
        card = rrect(s, 0.32 + col * 6.42, 1.90 + row * 1.55, 6.26, 1.42, WHITE, LINE, radius=0.08)
        fill_text(card, [(title, 14, True, NAVY), (body, 13, False, INK, PP_ALIGN.LEFT, 3)], ml=0.14, mt=0.1)
    note = rrect(s, 0.32, 5.15, 12.69, 1.85, AMBER_BG, radius=0.08)
    fill_text(note, [
        ("NO PROJECT SIMILARITY SCORE IS INVENTED HERE", 12, True, AMBER),
        ("MiniLM is a compact English encoder. A long technical chunk becomes one 384-dimensional vector. Changing the model means a new index name, because an existing index dimension is not silently reused.",
         14, False, INK, PP_ALIGN.LEFT, 4),
    ], ml=0.16, mt=0.1)


def s13(prs):
    s = new(prs)
    header(s, "Elasticsearch knowledge base", 13, "DONE")
    tiles = [
        ("12,054", "Index documents"),
        ("6,076", "crawled_pdf"),
        ("5,978", "html"),
        ("5,616", "content chunks"),
        ("5,616", "summaries"),
        ("822", "composites"),
    ]
    for i, (v, label) in enumerate(tiles):
        metric(s, 0.32 + (i % 6) * 2.15, 0.88, 2.05, 1.35, v, label)
    rows = [
        ["Mapping fact", "Value in this project"],
        ["Index name", "rag_corpus_v1"],
        ["Server", "Elasticsearch 8.12.2, single node, Docker Compose"],
        ["Text field", "BM25 match query; the project does not reimplement BM25"],
        ["embedding", "dense_vector, dims 384, index true, similarity cosine"],
        ["HNSW m and ef_construction", "Not set. They remain Elasticsearch defaults"],
        ["Document _id", "The chunk identifier, so a re-index overwrites"],
        ["Heap", "512 MB JVM, from docker-compose.yml"],
    ]
    add_table(s, rows, 0.32, 2.40, 12.69, 4.55, [4.4, 8.29], font=13)


def s14(prs):
    s = new(prs)
    header(s, "BM25 keyword retrieval", 14, "DONE")
    formula = rrect(s, 0.32, 0.88, 12.69, 1.15, NAVY, radius=0.08)
    fill_text(formula, [
        ("BM25(D, Q) = Σ IDF(qi) × [tf × (k1+1)] / [tf + k1 × (1 − b + b × |D| / avgdl)]", 16, True, WHITE, PP_ALIGN.CENTER),
        ("The project sends an Elasticsearch match query. It does not set k1 or b.", 13, False, RGBColor(0xC5, 0xD7, 0xE8), PP_ALIGN.CENTER, 4),
    ], anchor=MSO_ANCHOR.MIDDLE)
    parts = [
        ("qi", "A query term after analysis."),
        ("tf", "How often that term occurs in the chunk."),
        ("IDF", "A term that appears in few chunks counts more."),
        ("|D| / avgdl", "Length normalisation. Long chunks are pulled back."),
    ]
    for i, (t, body) in enumerate(parts):
        card = rrect(s, 0.32 + i * 3.24, 2.20, 3.10, 1.25, WHITE, LINE, radius=0.08)
        fill_text(card, [(t, 16, True, BLUE), (body, 13, False, INK, PP_ALIGN.LEFT, 3)], ml=0.1, mt=0.08)
    g = rrect(s, 0.32, 3.62, 6.22, 1.7, GREEN_BG, radius=0.08)
    b = rrect(s, 6.70, 3.62, 6.31, 1.7, AMBER_BG, radius=0.08)
    fill_text(g, [("WHERE IT IS USED", 13, True, GREEN),
                  ("Exact names, programme titles, and identifiers. The hybrid call runs this list before fusion. A standalone BM25 ranking of the five demo questions was not saved.",
                   13, False, INK, PP_ALIGN.LEFT, 4)], ml=0.14, mt=0.1)
    fill_text(b, [("WHERE IT MISSES", 13, True, AMBER),
                  ("A paraphrase that shares almost no terms with the passage. That is why the default mode also runs the vector retriever.",
                   13, False, INK, PP_ALIGN.LEFT, 4)], ml=0.14, mt=0.1)
    q = rrect(s, 0.32, 5.48, 12.69, 1.52, PANEL, radius=0.08)
    fill_text(q, [
        ("REFERENCE BOOST", 12, True, NAVY),
        ("When a question names a figure or table number, the BM25 query can add a terms clause, and fusion later adds figure_reference_boost 4.0. The five saved questions do not exercise that boost, so no boosted score is shown.",
         14, False, INK, PP_ALIGN.LEFT, 4),
    ], ml=0.16, mt=0.1)


def s15(prs):
    s = new(prs)
    header(s, "Vector retrieval and HNSW", 15, "DONE")
    panel = rrect(s, 0.32, 0.86, 6.3, 4.55, PANEL, radius=0.08)
    fill_text(panel, [("SEARCH THE PROJECT ACTUALLY CALLS", 11, True, BLUE)], ml=0.14, mt=0.08)
    layers = [(1.55, 3, "Top"), (2.45, 5, "Middle"), (3.40, 8, "Base")]
    centers = []
    for y, count, _ in layers:
        gap = 3.9 / count
        centers.append([(2.35 + i * gap + 0.12, y + 0.12) for i in range(count)])
    for upper, lower in zip(centers, centers[1:]):
        for ui, (x1, y1) in enumerate(upper):
            start = int(ui * len(lower) / len(upper))
            stop = min(len(lower), max(start + 1, int((ui + 1) * len(lower) / len(upper)) + 1))
            for vi in range(start, stop):
                x2, y2 = lower[vi]
                conn = s.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
                conn.line.color.rgb = RGBColor(0x9A, 0xB4, 0xCC)
                conn.line.width = Pt(1.15)
    for (y, count, label), row in zip(layers, centers):
        tb(s, 0.46, y, 1.7, 0.28, label, 11, True, MUTED)
        for cx, cy in row:
            oval(s, cx - 0.12, cy - 0.12, 0.24, 0.24, NAVY if y < 2 else BLUE)
    tb(s, 0.48, 4.35, 5.9, 0.85, "Elasticsearch builds the graph. This code supplies the query vector, k, and num_candidates 50.", 12, False, INK)
    notes = [
        ("Query", "The question is embedded with the same local model, then sent as a knn clause."),
        ("Similarity", "Cosine, on the dense_vector field. A smaller angle ranks higher."),
        ("Not measured", "m and ef_construction are unset. hnsw_benchmark.py can compare a NumPy scan with kNN. No output file is stored."),
        ("Scale", "12,054 vectors of 384 floats are about 18.5 MB of raw payload before index overhead. That is a size calculation, not a disk reading."),
    ]
    for i, (title, body) in enumerate(notes):
        card = rrect(s, 6.80, 0.86 + i * 1.15, 6.20, 1.06, WHITE, LINE, radius=0.08)
        fill_text(card, [(title, 13, True, NAVY), (body, 12, False, INK, PP_ALIGN.LEFT, 2)], ml=0.12, mt=0.06)
    foot = rrect(s, 0.32, 5.55, 12.69, 1.45, PANEL, radius=0.08)
    fill_text(foot, [
        ("STATUS", 12, True, NAVY),
        ("The index setting is DONE. A saved HNSW recall or latency number is NEEDS LIVE VALIDATION. The five demo JSON files do not store a separate vector ranking.",
         14, False, INK, PP_ALIGN.LEFT, 3),
    ], ml=0.16, mt=0.1)


def s16(prs):
    s = new(prs)
    header(s, "Hybrid retrieval", 16, "DONE")
    hflow(s, [["Pool 20"], ["Two scores"], ["Min-max"], ["0.3 / 0.7"], ["Boosts"], ["Top 5"]],
          0.32, 0.86, 12.69, 0.62, NAVY, WHITE)
    eq = rrect(s, 0.32, 1.62, 12.69, 0.7, RGBColor(0xE7, 0xF0, 0xF8), radius=0.08)
    fill_text(eq, [("Final = 0.3 × normalised BM25 + 0.7 × normalised vector + reference boost + image boost",
                    15, True, NAVY, PP_ALIGN.CENTER)], anchor=MSO_ANCHOR.MIDDLE)
    left = rrect(s, 0.32, 2.48, 6.22, 2.35, WHITE, LINE, radius=0.08)
    right = rrect(s, 6.70, 2.48, 6.31, 2.35, WHITE, LINE, radius=0.08)
    fill_text(left, [
        ("MIN-MAX, THE ACTIVE STRATEGY", 12, True, BLUE),
        ("s' = (s − min) / (max − min), separately for each list.", 13, False, INK, PP_ALIGN.LEFT, 4),
        ("If every score in a list is equal, each becomes 1.", 13, False, INK, PP_ALIGN.LEFT, 3),
        ("A chunk missing from one list scores 0 on that side.", 13, False, INK, PP_ALIGN.LEFT, 3),
        ("BM25 is unbounded and cosine is not, so the raw scores are not added.", 13, False, INK, PP_ALIGN.LEFT, 3),
    ], ml=0.12, mt=0.08)
    fill_text(right, [
        ("RRF IS IMPLEMENTED AND INACTIVE", 12, True, BLUE),
        ("RRF = Σ 1 / (60 + rank), ranks starting at 1.", 16, True, NAVY, PP_ALIGN.LEFT, 4),
        ("It runs only when fusion_strategy is rrf. The setting is min_max.", 13, False, INK, PP_ALIGN.LEFT, 3),
        ("The same boosts are added after either strategy. Ties break on chunk_id.", 13, False, INK, PP_ALIGN.LEFT, 3),
    ], ml=0.12, mt=0.08)
    ill = rrect(s, 0.32, 4.98, 12.69, 2.02, AMBER_BG, radius=0.08)
    fill_text(ill, [
        ("ARITHMETIC ILLUSTRATION FROM THE REPORT, NOT A SAVED RETRIEVAL", 12, True, AMBER),
        ("BM25 scores 12, 6, and 9 normalise to 1, 0, and 0.5. Vector scores 0.40, 0.90, and 0.65 normalise to 0, 1, and 0.5. Hybrids are 0.30, 0.70, and 0.50. The second item ranks first because the semantic weight is larger.",
         13, False, INK, PP_ALIGN.LEFT, 3),
        ("The demo JSON stores only the fused score. It does not store the BM25 and vector components, so the live scores are not fitted to this example.",
         13, False, INK, PP_ALIGN.LEFT, 2),
    ], ml=0.14, mt=0.08)


def s17(prs):
    s = new(prs)
    header(s, "Grounding and answer generation", 17, ["DONE", "PARTIAL"])
    rules = [
        "Use only the retrieved context.",
        "Do not use outside knowledge.",
        "Keep the source terminology.",
        "If the context is insufficient, use the fixed refusal sentence.",
        "Name the document and the page when the context supports the answer.",
    ]
    for i, text in enumerate(rules):
        card = rrect(s, 0.32, 0.86 + i * 0.72, 7.7, 0.64, WHITE, LINE, radius=0.08)
        fill_text(card, [(f"{i+1}     {text}", 14, False, INK)], anchor=MSO_ANCHOR.MIDDLE, ml=0.12)
    side = rrect(s, 8.18, 0.86, 4.83, 2.35, NAVY, radius=0.08)
    fill_text(side, [
        ("ASK-PATH GATE", 12, True, RGBColor(0x9F, 0xC4, 0xE8)),
        ("No hits.", 14, False, WHITE, PP_ALIGN.LEFT, 6),
        ("Leading score below 0.35.", 14, False, WHITE, PP_ALIGN.LEFT, 3),
        ("A named role is absent from the hit text.", 14, False, WHITE, PP_ALIGN.LEFT, 3),
        ("Temperature on a call that passes is 0.", 14, False, WHITE, PP_ALIGN.LEFT, 3),
    ], ml=0.14, mt=0.1)
    mock = rrect(s, 8.18, 3.35, 4.83, 1.45, PANEL, radius=0.08)
    live = rrect(s, 8.18, 4.94, 4.83, 1.95, GREEN_BG, radius=0.08)
    fill_text(mock, [
        ("DEFAULT CLIENT", 12, True, NAVY),
        ("settings set llm_provider to mock. That client returns a fixed sentence and is not evidence of answer quality.",
         13, False, INK, PP_ALIGN.LEFT, 3),
    ], ml=0.12, mt=0.08)
    fill_text(live, [
        ("STORED DEMO", 12, True, GREEN),
        ("responses.md names Llama 3.2 through Ollama. Five questions, hybrid, include_kg true, none abstained.",
         13, False, INK, PP_ALIGN.LEFT, 3),
    ], ml=0.12, mt=0.08)


def s18(prs):
    s = new(prs)
    header(s, "Worked query: cyber forensics", 18)
    q = rrect(s, 0.32, 0.84, 12.69, 0.55, NAVY, radius=0.08)
    fill_text(q, [("What cyber forensics work does C-DAC describe?", 16, True, WHITE, PP_ALIGN.CENTER)],
              anchor=MSO_ANCHOR.MIDDLE)
    steps = [
        ("File", "data/pdfs/82f38954477f2c53.pdf was already extracted to data/extracted/pdf_82f38954477f2c53.json."),
        ("Pages", "25 pages, 0 failed, 0 OCR-flagged. 95 blocks became 61 chunks. Mean length 842. 3 merges. 0 LLM calls."),
        ("Hit", "chunk_002, section “Expression of interest for reselling software and hardware products”, page 1."),
        ("Score", "Hybrid fused score 0.6678465341224721. Its summary is a second hit at the same score, with no pages."),
        ("Gate", "The leading score is above 0.35, so the model was called. Five knowledge-graph rows were then added."),
        ("Citation", "The application source for the PDF is page 1. The model’s own provenance paragraph names a different chunk."),
    ]
    for i, (title, body) in enumerate(steps):
        col, row = i % 2, i // 2
        card = rrect(s, 0.32 + col * 6.48, 1.54 + row * 1.35, 6.34, 1.24, WHITE, LINE, radius=0.08)
        fill_text(card, [(title, 13, True, NAVY), (body, 12, False, INK, PP_ALIGN.LEFT, 2)], ml=0.12, mt=0.08)
    bar = rrect(s, 0.32, 5.66, 12.69, 1.34, AMBER_BG, radius=0.08)
    fill_text(bar, [
        ("GENERATION GAP, RECORDED AND NOT REPAIRED", 12, True, AMBER),
        ("Llama 3.2 says the cited provenance does not explicitly mention the work, then still infers that C-DAC’s research may include cyber forensics. Retrieval found a real page. The extra inference is a generation-quality gap.",
         13, False, INK, PP_ALIGN.LEFT, 3),
    ], ml=0.14, mt=0.08)


def s19(prs):
    s = new(prs)
    header(s, "Saved answer page", 19)
    s.shapes.add_picture(str(SHOT), Inches(0.28), Inches(0.84), Inches(8.15), Inches(4.34))
    card = rrect(s, 8.55, 0.84, 4.48, 6.16, PANEL, radius=0.08)
    fill_text(card, [
        ("Q5, FROM q5.json", 12, True, NAVY),
        ("Abstained: false", 13, False, INK, PP_ALIGN.LEFT, 8),
        ("Hits: 10", 13, False, INK, PP_ALIGN.LEFT, 2),
        ("Mode: hybrid", 13, False, INK, PP_ALIGN.LEFT, 2),
        ("Model: Llama 3.2", 13, False, INK, PP_ALIGN.LEFT, 2),
        ("Source 1", 13, True, NAVY, PP_ALIGN.LEFT, 8),
        ("HTML summary, 0.8489", 12, False, INK, PP_ALIGN.LEFT, 1),
        ("Project Details", 12, False, INK, PP_ALIGN.LEFT, 1),
        ("Source 2", 13, True, NAVY, PP_ALIGN.LEFT, 6),
        ("HTML summary, 0.7108", 12, False, INK, PP_ALIGN.LEFT, 1),
        ("Cyber Security & Cyber Forensics", 12, False, INK, PP_ALIGN.LEFT, 1),
        ("Source 3", 13, True, NAVY, PP_ALIGN.LEFT, 6),
        ("Crawled PDF, page 1", 12, False, INK, PP_ALIGN.LEFT, 1),
        ("Score 0.6678", 12, False, INK, PP_ALIGN.LEFT, 1),
        ("Screenshot file: q5-cyber-forensics.png", 11, False, MUTED, PP_ALIGN.LEFT, 8),
    ], ml=0.12, mt=0.1)


def s20(prs):
    s = new(prs)
    header(s, "Five recorded questions", 20)
    rows = [
        ["#", "Question", "Abstain", "Hits", "Source mix"],
        ["1", "What is C-DAC's Aadhaar Data Vault service?", "no", "10", "HTML only"],
        ["2", "Where is the C-DAC Pune centre located?", "no", "10", "9 HTML, 1 PDF"],
        ["3", "What has C-DAC done in High Performance Computing?", "no", "10", "HTML only"],
        ["4", "What training does the C-DAC ACTS programme provide?", "no", "10", "9 HTML, 1 PDF"],
        ["5", "What cyber forensics work does C-DAC describe?", "no", "10", "8 HTML, 2 PDF"],
    ]
    add_table(s, rows, 0.28, 0.86, 12.76, 3.55, [0.6, 6.5, 1.15, 0.8, 3.71], font=12)
    cards = [
        ("Aadhaar, question 1", "Top HTML chunk score 0.8853512993505596, section “Aadhaar Data Vault (ADV) as a service”."),
        ("Every question", "Retrieval mode hybrid. Five corpus hits plus five knowledge-graph hits. include_kg was true."),
        ("Not a benchmark", "BM25-only and vector-only lists for these questions were not saved. No precision or nDCG is stated."),
    ]
    for i, (title, body) in enumerate(cards):
        card = rrect(s, 0.32 + i * 4.26, 4.60, 4.12, 2.38, PANEL, radius=0.08)
        fill_text(card, [(title, 14, True, NAVY), (body, 13, False, INK, PP_ALIGN.LEFT, 6)], ml=0.12, mt=0.1)


def s21(prs):
    s = new(prs)
    header(s, "Corpus results", 21)
    tb(s, 0.32, 0.80, 12.6, 0.26, "Counted from data/extracted, the live index, and docs/demo-results. Query latency was not stored.", 12, False, MUTED)
    tiles = [
        ("60", "Crawled PDFs"),
        ("1,479", "Pages"),
        ("0", "Failed pages"),
        ("31", "Pages needing OCR"),
        ("2,880", "PDF text chunks"),
        ("316", "PDF composites"),
        ("12,054", "Index documents"),
        ("384", "Vector dimensions"),
        ("44.38 s", "Extraction sum"),
        ("15.73 s", "Chunking sum"),
    ]
    for i, (v, label) in enumerate(tiles):
        metric(s, 0.32 + (i % 5) * 2.58, 1.16 + (i // 5) * 1.55, 2.46, 1.42, v, label)
    ban = rrect(s, 0.32, 4.40, 12.69, 2.58, AMBER_BG, radius=0.08)
    fill_text(ban, [
        ("HOW TO READ THESE NUMBERS", 13, True, AMBER),
        ("6,076 crawled-PDF index documents equal 2,880 chunks + 2,880 summaries + 316 composites. That is the check that the artifact scan and the index agree.",
         14, False, INK, PP_ALIGN.LEFT, 4),
        ("Content kinds across HTML and PDF: 5,616 chunks, 5,616 summaries, 822 composites.", 14, False, INK, PP_ALIGN.LEFT, 3),
        ("These figures show that ingestion and indexing ran. They are not a retrieval-accuracy benchmark. Hit rate, recall, MRR, and latency have code and no stored scored run.",
         14, False, INK, PP_ALIGN.LEFT, 3),
    ], ml=0.16, mt=0.1)


def s22(prs):
    s = new(prs)
    header(s, "Repository and commands", 22, "DONE")
    rows = [
        ["Command", "Result"],
        ["python run.py status", "JSON with elasticsearch_available; exit 0 even if the cluster is down"],
        ["python run.py crawl", "Scrapy spider cdac, domain and URL cap from config"],
        ["python run.py process-pdfs", "Artifacts for manifest PDFs, source_kind crawled_pdf"],
        ["python run.py ingest --pdf FILE", "One uploaded PDF artifact and chunk count"],
        ["python run.py index", "Embed and bulk-index; prints successfully_indexed"],
        ["python run.py build-kg", "Nodes and edges in data/kg/graph_v1.json"],
        ["python run.py query \"...\" --sources pdf", "Hits and the grounding flag, no LLM call"],
        ["python run.py ask \"...\" --include-kg", "RAGAnswer JSON; graph rows only when the flag is set"],
        ["python run.py ui --port 8765", "Inspector on 127.0.0.1"],
        ["python -m pytest tests -q", "Recorded re-run: 281 passed, 2 skipped, 7 teardown errors"],
    ]
    add_table(s, rows, 0.28, 0.84, 12.76, 5.55, [5.5, 7.26], font=12)
    tb(s, 0.32, 6.48, 12.6, 0.55,
       "Package rag has the crawl, extract, index, retrieve, answer, graph, eval, and UI modules. Elasticsearch is docker compose up -d elasticsearch.",
       13, False, MUTED)


def s23(prs):
    s = new(prs)
    header(s, "Testing and validation", 23)
    tb(s, 0.32, 0.80, 12.6, 0.32,
       "Re-run while the report was written: python -m pytest tests -q --tb=no. 281 passed, 2 skipped, 7 errors, 74.85 seconds.",
       13, False, INK)
    rows = [
        ["Area", "What a pass shows", "What it does not show"],
        ["PDF extraction", "Multi-page text, empty pages, per-page failure", "Teardown hit WinError 32 on a locked temp file"],
        ["OCR", "The pipeline continues when OCR cannot run", "Tesseract was absent, so the 31 pages were not recognised"],
        ["Chunking and references", "Split, overlap, merge, figure and table parsing", "No live cluster"],
        ["Embeddings and index", "Mock vectors, dimension checks, mapping shape", "The stand-in client is not the live 12,054-document index"],
        ["BM25, vector, hybrid", "Query shape, fusion arithmetic, weight sum", "No saved ranking of the five demo questions by mode"],
        ["Ask contract", "Prompt text, source filters, abstain without a model", "No live Llama call inside the suite"],
    ]
    add_table(s, rows, 0.28, 1.22, 12.76, 4.15, [2.5, 5.2, 5.06], font=12)
    note = rrect(s, 0.32, 5.52, 12.69, 1.48, AMBER_BG, radius=0.08)
    fill_text(note, [
        ("EARLIER NOTE IN CURRENT_STATE.md", 12, True, AMBER),
        ("That file records 280 passed and 3 skipped. Chapter 27 uses the later re-run of 281 passed, 2 skipped, and 7 teardown errors. The test bodies passed; the errors are at path.unlink.",
         14, False, INK, PP_ALIGN.LEFT, 3),
    ], ml=0.14, mt=0.08)


def s24(prs):
    s = new(prs)
    header(s, "Limitations", 24)
    left = rrect(s, 0.32, 0.86, 6.28, 6.14, WHITE, LINE, radius=0.08)
    right = rrect(s, 6.74, 0.86, 6.27, 6.14, WHITE, LINE, radius=0.08)
    fill_text(left, [("OBSERVED", 14, True, AMBER)], ml=0.16, mt=0.1)
    fill_text(right, [("STRUCTURAL, NOT A MEASURED FAILURE RATE", 13, True, NAVY)], ml=0.16, mt=0.1)
    observed = [
        "31 pages still require OCR. The test host had no Tesseract binary.",
        "No image chunk was produced for the 60 PDFs.",
        "pypdf does not recover column order.",
        "Heading and header rules are regular expressions and can misfire.",
        "Summary documents are retrieved without page numbers.",
        "One outlier in a pool of 20 rescales every other min-max score.",
        "The default LLM is a mock. Quality depends on the configured endpoint.",
        "Question 5’s model inferred past the hits after the gate had passed.",
        "Windows teardown can raise WinError 32 while PyMuPDF holds the file.",
    ]
    structural = [
        "A 384-d MiniLM vector can miss a technical paraphrase or an exact identifier. Hybrid reduces that risk and does not remove it.",
        "Elasticsearch is required for a live question. status reports an outage and still exits 0.",
        "The index mixes HTML and PDF. A PDF-only question must pass --sources pdf.",
        "There is no measured limit for a PDF of many hundreds of pages.",
        "Hit rate, recall, MRR, and latency have no stored run on this corpus.",
        "Precision and nDCG are not implemented.",
        "HNSW recall against a brute-force scan was not saved.",
        "Query time and model time are absent from the demo JSON.",
        "Agent tool routing is not implemented. rag.services exposes ordinary functions.",
    ]
    for i, text in enumerate(observed):
        y = 1.32 + i * 0.60
        oval(s, 0.50, y + 0.06, 0.14, 0.14, AMBER)
        tb(s, 0.76, y, 5.6, 0.55, text, 12, False, INK)
    for i, text in enumerate(structural):
        y = 1.32 + i * 0.60
        oval(s, 6.92, y + 0.06, 0.14, 0.14, BLUE)
        tb(s, 7.18, y, 5.6, 0.55, text, 12, False, INK)


def s25(prs):
    s = new(prs)
    header(s, "Future scope", 25, "PENDING")
    rows = [
        ["Proposal", "Gap it addresses", "Phase"],
        ["Run OCR on the 31 sparse pages", "Tesseract was not available on the test host", "1"],
        ["Copy page numbers onto summaries", "Summary hits cite no page", "1"],
        ["Close the PyMuPDF handle before unlink", "WinError 32 during test teardown", "1"],
        ["Layout-aware extraction", "Columns and tables are plain text", "1"],
        ["Cross-encoder over the pool of 20", "Fusion has no second reader", "2"],
        ["Query rewriting", "One string is embedded", "2"],
        ["Citation check on the completion", "Question 5 inferred past the hits", "2"],
        ["Labelled set and hit rate, recall, MRR", "metrics.py has no stored run", "2"],
        ["A documented --sources pdf preset", "The demo searched all sources", "3"],
    ]
    add_table(s, rows, 0.28, 0.86, 12.76, 5.15, [5.3, 5.7, 1.76], font=13)
    tb(s, 0.32, 6.15, 12.6, 0.85,
       "Every row is proposed. None of it is described as already running. Agentic orchestration is outside the current code.",
       14, False, MUTED)


def s26(prs):
    s = new(prs)
    header(s, "Conclusion", 26)
    body = rrect(s, 0.32, 0.88, 12.69, 3.55, PANEL, radius=0.08)
    fill_text(body, [
        ("WHAT THIS INTERNSHIP BUILT", 13, True, NAVY),
        ("A crawler collects pages and PDFs from cdac.gov.in. Each PDF is hashed, extracted page by page, cleaned, and packed into chunks that keep their pages. Those chunks are embedded with all-MiniLM-L6-v2 and stored in Elasticsearch beside a BM25 field and a cosine vector.",
         15, False, INK, PP_ALIGN.LEFT, 6),
        ("Questions use a min-max hybrid of BM25 and vector search, weights 0.3 and 0.7, a pool of 20, and a final cut of 5. The ask path can abstain. It returns the source list that retrieval actually produced.",
         15, False, INK, PP_ALIGN.LEFT, 5),
        ("The stored corpus is 60 PDFs, 1,479 pages, and 2,880 chunks, inside an index of 12,054 documents. Five Llama 3.2 answers are the generation record. 281 tests passed, with seven Windows teardown errors that did not fail the assertions.",
         15, False, INK, PP_ALIGN.LEFT, 5),
    ], ml=0.18, mt=0.12)
    open_items = rrect(s, 0.32, 4.58, 12.69, 1.15, AMBER_BG, radius=0.08)
    fill_text(open_items, [
        ("STILL OPEN", 12, True, AMBER),
        ("OCR on the sparse pages, page numbers on summary hits, a labelled retrieval score, and a check that the model’s sentences stay inside the source list.",
         14, False, INK, PP_ALIGN.LEFT, 2),
    ], ml=0.16, mt=0.08)
    thanks = rrect(s, 0.32, 5.88, 12.69, 1.12, NAVY, radius=0.08)
    fill_text(thanks, [
        ("THANK YOU  —  QUESTIONS AND DISCUSSION", 18, True, WHITE, PP_ALIGN.CENTER),
        ("Vyom Desai    ·    Guide: Pavan Dhote    ·    C-DAC, Pune    ·    15 June 2026 – 15 September 2026",
         12, False, RGBColor(0xC5, 0xD7, 0xE8), PP_ALIGN.CENTER, 2),
    ], anchor=MSO_ANCHOR.MIDDLE)


def build():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    prs.core_properties.title = (
        "Development of an Web Crawler for Automated Knowledge Base Creation "
        "and RAG-Based Information Retrieval"
    )
    prs.core_properties.author = "Vyom Desai"
    prs.core_properties.subject = "C-DAC Pune internship, 15 June 2026 to 15 September 2026"
    prs.core_properties.category = "Internship project review"
    for fn in (s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12, s13, s14, s15,
               s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26):
        fn(prs)
    if len(prs.slides) != TOTAL:
        raise SystemExit(f"slide count {len(prs.slides)}")
    sw, sh = prs.slide_width, prs.slide_height
    for i, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if shape.left + shape.width > sw + Emu(30000) or shape.top + shape.height > sh + Emu(30000):
                print("OVER", i, int(shape.left), int(shape.top), int(shape.width), int(shape.height))
    prs.save(OUT)
    stale = Path(r"C:\CDAC\RAG\docs\CDAC_PDF_RAG_Technical_Review_Vyom_Desai.pptx")
    other = Path(r"C:\CDAC\pdf-rag\docs\CDAC_PDF_RAG_Technical_Review_Vyom_Desai.pptx")
    if stale.exists():
        stale.unlink()
    if other.exists():
        other.unlink()
    print(OUT)
    print("slides", len(prs.slides))


if __name__ == "__main__":
    build()
