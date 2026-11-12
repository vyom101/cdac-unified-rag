from pathlib import Path

from rag.crawl.pdf_downloader import PdfDownloadStore, pdf_id_from_url
from rag.crawl.url_utils import is_irrelevant_asset_url, is_pdf_url, is_processable_url


def test_is_pdf_url():
    assert is_pdf_url("https://cdac.gov.in/docs/report.pdf")
    assert not is_pdf_url("https://cdac.gov.in/about")


def test_irrelevant_assets_skipped():
    assert is_irrelevant_asset_url("https://cdac.gov.in/static/style.css")
    assert is_irrelevant_asset_url("https://cdac.gov.in/images/logo.png")
    assert not is_irrelevant_asset_url("https://cdac.gov.in/docs/report.pdf")


def test_processable_url():
    assert is_processable_url("https://cdac.gov.in/index.aspx")
    assert is_processable_url("https://cdac.gov.in/docs/report.pdf")
    assert not is_processable_url("https://cdac.gov.in/app.js")


def test_pdf_download_store(tmp_path):
    store = PdfDownloadStore(str(tmp_path / "pdfs"), str(tmp_path / "manifest.jsonl"))
    record = store.save(
        source_url="https://cdac.gov.in/docs/report.pdf",
        canonical_url="https://cdac.gov.in/docs/report.pdf",
        body=b"%PDF-1.4 fake content",
        parent_url="https://cdac.gov.in/", sub_link="/docs/report.pdf", depth=2,
    )
    assert record.pdf_id == pdf_id_from_url("https://cdac.gov.in/docs/report.pdf")
    assert Path(record.local_path).exists()
    assert len(store.list_downloaded()) == 1
