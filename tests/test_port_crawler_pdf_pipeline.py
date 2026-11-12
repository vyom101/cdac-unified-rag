"""Tests for PDF URL detection and offline processing."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.config import AppConfig
from rag.models import PdfRecord
from rag.crawl.pdf_downloader import PdfDownloadStore, pdf_id_from_url
from rag.crawl.pdf_processor import pages_to_sections, process_pdf_offline
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
    body = b"%PDF-1.4 fake content"
    record = store.save(
        source_url="https://cdac.gov.in/docs/report.pdf",
        canonical_url="https://cdac.gov.in/docs/report.pdf",
        body=body,
        parent_url="https://cdac.gov.in/",
        sub_link="/docs/report.pdf",
        depth=2,
    )
    assert record.pdf_id == pdf_id_from_url("https://cdac.gov.in/docs/report.pdf")
    assert Path(record.local_path).exists()
    listed = store.list_downloaded()
    assert len(listed) == 1


@patch("rag.crawl.pdf_processor.extract_pdf_pages")
def test_process_pdf_offline(mock_extract, tmp_path):
    mock_extract.return_value = [
        (1, "C-DAC develops advanced computing systems in Pune."),
        (2, "PARAM supercomputers are a key product line."),
    ]
    record = PdfRecord(
        pdf_id="abc123",
        source_url="https://cdac.gov.in/docs/report.pdf",
        canonical_url="https://cdac.gov.in/docs/report.pdf",
        parent_url="https://cdac.gov.in/",
        sub_link="/docs/report.pdf",
        depth=2,
        local_path=str(tmp_path / "abc123.pdf"),
        file_size=100,
        content_hash="hash",
    )
    Path(record.local_path).write_bytes(b"%PDF")
    config = AppConfig(composite_group_size=10, min_chunk_size=20)
    artifact = process_pdf_offline(record, config)
    assert artifact.source_type == "pdf"
    assert artifact.parent_url == "https://cdac.gov.in/"
    assert artifact.depth == 2
    assert len(artifact.chunks) >= 1
    assert len(artifact.chunk_summaries) == len(artifact.chunks)
    assert artifact.chunks[0].metadata["source_type"] == "pdf"


def test_pages_to_sections():
    sections = pages_to_sections("pdf1", "https://cdac.gov.in/a.pdf", [(1, "Page one text")])
    assert len(sections) == 1
    assert sections[0].heading_path == ["Page 1"]

