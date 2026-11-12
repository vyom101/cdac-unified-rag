from rag.crawl.html_extractor import extract_sections
from rag.crawl.pipelines import page_id_from_url
from rag.index.chunker import SectionChunker

HTML = """<!doctype html><html><body>
<nav>Home</nav><main><h1>About C-DAC</h1>
<p>Centre for Development of Advanced Computing (C-DAC) is a premier R&amp;D organization.</p>
<h2>Locations</h2><p>C-DAC operates from Pune, India.</p>
<table><tr><th>City</th><th>Country</th></tr><tr><td>Pune</td><td>India</td></tr></table>
</main><footer>Footer should be removed</footer></body></html>"""


def test_extract_sections_removes_boilerplate_and_keeps_table():
    sections = extract_sections(
        HTML, page_id=page_id_from_url("https://cdac.gov.in/about"),
        source_url="https://cdac.gov.in/about", title="About",
    )
    assert sections
    combined = "\n".join(section.text for section in sections)
    assert "Centre for Development of Advanced Computing" in combined
    assert "Pune" in combined
    assert "Footer should be removed" not in combined
    assert "City" in combined and "Country" in combined


def test_extract_sections_handles_null_tag_attrs():
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(
        "<html><body><div><p>C-DAC operates advanced computing programmes from Pune, India.</p></div></body></html>",
        "lxml",
    )
    soup.find("div").attrs = None
    sections = extract_sections(
        str(soup), page_id=page_id_from_url("https://cdac.gov.in/"),
        source_url="https://cdac.gov.in/", title="Home",
    )
    assert sections
    assert "C-DAC" in sections[0].text


def test_chunker_produces_traceable_chunks():
    page_id = page_id_from_url("https://cdac.gov.in/about")
    sections = extract_sections(
        HTML, page_id=page_id, source_url="https://cdac.gov.in/about", title="About"
    )
    chunks, stats = SectionChunker(min_chunk_size=20).process_sections(
        sections, document_name="About", source_url="https://cdac.gov.in/about",
        parent_url="https://cdac.gov.in/", depth=1, sub_link="/about",
    )
    assert stats.accepted_chunks >= 1
    assert all(chunk.source_url == "https://cdac.gov.in/about" for chunk in chunks)
    assert all(chunk.parent_url == "https://cdac.gov.in/" for chunk in chunks)
    assert all(chunk.depth == 1 for chunk in chunks)
    assert all(chunk.chunk_id.startswith("doc_") for chunk in chunks)
