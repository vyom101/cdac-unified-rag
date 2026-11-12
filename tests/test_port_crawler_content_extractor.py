"""Tests for HTML extraction and chunking."""

from pathlib import Path

from rag.index.chunker import SectionChunker
from rag.crawl.html_extractor import extract_sections
from rag.crawl.pipelines import page_id_from_url

FIXTURE = Path(__file__).parent / "fixtures" / "sample_about.html"


def test_extract_sections_removes_boilerplate_and_keeps_table():
    html = FIXTURE.read_text(encoding="utf-8")
    page_id = page_id_from_url("https://cdac.gov.in/about")
    sections = extract_sections(html, page_id=page_id, source_url="https://cdac.gov.in/about", title="About")
    assert sections
    combined = "\n".join(s.text for s in sections)
    assert "Centre for Development of Advanced Computing" in combined
    assert "Pune" in combined
    assert "Footer should be removed" not in combined
    assert "City" in combined and "Country" in combined


def test_extract_sections_handles_null_tag_attrs():
    from bs4 import BeautifulSoup

    html = "<html><body><div><p>C-DAC operates advanced computing programmes from Pune, India.</p></div></body></html>"
    soup = BeautifulSoup(html, "lxml")
    div = soup.find("div")
    div.attrs = None
    sections = extract_sections(
        str(soup),
        page_id=page_id_from_url("https://cdac.gov.in/"),
        source_url="https://cdac.gov.in/",
        title="Home",
    )
    assert sections
    assert "C-DAC" in sections[0].text


def test_chunker_produces_traceable_chunks():
    html = FIXTURE.read_text(encoding="utf-8")
    page_id = page_id_from_url("https://cdac.gov.in/about")
    sections = extract_sections(html, page_id=page_id, source_url="https://cdac.gov.in/about", title="About")
    chunks, stats = SectionChunker(min_chunk_size=20).process_sections(
        sections,
        document_name="About",
        source_url="https://cdac.gov.in/about",
        parent_url="https://cdac.gov.in/",
        depth=1,
        sub_link="/about",
    )
    assert stats.accepted_chunks >= 1
    assert all(c.source_url == "https://cdac.gov.in/about" for c in chunks)
    assert all(c.parent_url == "https://cdac.gov.in/" for c in chunks)
    assert all(c.depth == 1 for c in chunks)
    assert all(c.chunk_id.startswith("doc_") for c in chunks)

