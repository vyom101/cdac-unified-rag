import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pypdf.errors import PdfReadError

from rag.extract.pdf_processor import (
    PDFProcessor,
    check_requires_ocr,
    clean_extracted_text,
    detect_block_type,
    generate_document_id,
)
from rag.models import DocumentModel


def test_clean_extracted_text():
    raw = (
        "This is a hyphen-\nated word. Here is a control char \x01 and \r\n"
        "repeated empty lines.\n\n\n\nAnd double  spaces."
    )
    cleaned = clean_extracted_text(raw)
    assert "hyphenated" in cleaned
    assert "\x01" not in cleaned
    assert "\r\n" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "double spaces." in cleaned


def test_check_requires_ocr():
    assert check_requires_ocr("") is True
    assert check_requires_ocr("   \n   ") is True
    assert check_requires_ocr("Short text") is True
    assert check_requires_ocr(
        "This is a sufficiently long line of text that has plenty of characters."
    ) is False


def test_detect_block_type():
    assert detect_block_type("1. Introduction") == "heading"
    assert detect_block_type("1.3.2 Methodology Details") == "heading"
    assert detect_block_type("INTRODUCTION") == "heading"
    assert detect_block_type("This is a standard sentence ending with a period.") == "paragraph"


def test_deterministic_document_id():
    first = generate_document_id(b"%PDF-1.4 mock pdf contents here")
    second = generate_document_id(b"%PDF-1.4 mock pdf contents here")
    different = generate_document_id(b"%PDF-1.4 different contents")
    assert first == second
    assert first != different
    assert len(first) == 64


@pytest.fixture
def mock_pdf_file():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary:
        temporary.write(b"%PDF-1.4 mock bytes")
        path = Path(temporary.name)
    yield path
    path.unlink(missing_ok=True)


@patch("rag.extract.pdf_processor.PdfReader")
def test_valid_multipage_pdf_processing(reader_class, mock_pdf_file):
    page_one = MagicMock()
    page_one.extract_text.return_value = (
        "1. Introduction\n\nThis is paragraph one of page 1."
    )
    page_two = MagicMock()
    page_two.extract_text.return_value = (
        "2. Architecture\n\nThis is paragraph two of page 2."
    )
    reader = MagicMock()
    reader.pages = [page_one, page_two]
    reader.metadata = {"/Title": "CDAC Blueprint", "/Author": "Mentor Team"}
    reader_class.return_value = reader
    document = PDFProcessor().process_pdf(mock_pdf_file)
    assert isinstance(document, DocumentModel)
    assert document.total_pages == 2
    assert document.successfully_extracted_pages == 2
    assert document.failed_pages == 0
    assert document.extraction_status == "SUCCESS"
    assert document.metadata["Title"] == "CDAC Blueprint"
    assert len(document.pages[0].blocks) == 2
    assert document.pages[0].blocks[0].block_type == "heading"
    assert document.pages[1].page_number == 2


@patch("rag.extract.pdf_processor.PdfReader")
def test_empty_and_ocr_page_handling(reader_class, mock_pdf_file):
    pages = [MagicMock(), MagicMock()]
    pages[0].extract_text.return_value = "   "
    pages[1].extract_text.return_value = "Scan"
    reader = MagicMock(pages=pages, metadata=None)
    reader_class.return_value = reader
    document = PDFProcessor().process_pdf(mock_pdf_file)
    assert document.successfully_extracted_pages == 0
    assert document.pages_requiring_ocr == 2
    assert document.pages[0].requires_ocr is True
    assert document.pages[0].extraction_status == "IMAGE_ONLY"


@patch("rag.extract.pdf_processor.PdfReader")
def test_per_page_failure_handling(reader_class, mock_pdf_file):
    pages = [MagicMock(), MagicMock()]
    pages[0].extract_text.return_value = "Page 1 extracted content is perfectly fine."
    pages[1].extract_text.side_effect = RuntimeError("Extraction stream corruption")
    reader = MagicMock(pages=pages, metadata={})
    reader_class.return_value = reader
    document = PDFProcessor().process_pdf(mock_pdf_file)
    assert document.successfully_extracted_pages == 1
    assert document.failed_pages == 1
    assert document.extraction_status == "PARTIAL_SUCCESS"
    assert "Extraction stream corruption" in document.pages[1].extraction_error


def test_nonexistent_pdf_handling():
    with pytest.raises(FileNotFoundError):
        PDFProcessor().process_pdf(Path("data/raw/does_not_exist_xyz.pdf"))


@patch("rag.extract.pdf_processor.PdfReader")
def test_corrupted_or_empty_pdf_handling(reader_class, mock_pdf_file):
    reader_class.return_value = MagicMock(pages=[])
    with pytest.raises(ValueError, match="Empty PDF file containing 0 pages"):
        PDFProcessor().process_pdf(mock_pdf_file)
    reader_class.side_effect = PdfReadError("PDF header is corrupted")
    with pytest.raises(PdfReadError, match="Invalid, corrupted, or unreadable PDF"):
        PDFProcessor().process_pdf(mock_pdf_file)
