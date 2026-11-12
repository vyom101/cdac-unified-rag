import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from pydantic import ValidationError
from pypdf.errors import PdfReadError

from rag.extract.pdf_processor import (
    PDFProcessor,
    clean_extracted_text,
    check_requires_ocr,
    detect_block_type,
    generate_document_id,
    DocumentModel,
    PageModel,
    BlockModel
)
from rag.config import load_config

# ==============================================================================
# INDEPENDENT UNIT TESTS FOR UTILITY HELPERS
# ==============================================================================

def test_clean_extracted_text():
    """
    Validates text cleaning rules (control characters, carriage returns,
    line-wrap hyphens, multiple blank lines, and padding spaces).
    """
    raw_text = "This is a hyphen-\nated word. Here is a control char \x01 and \r\nrepeated empty lines.\n\n\n\nAnd double  spaces."
    cleaned = clean_extracted_text(raw_text)
    
    assert "hyphenated" in cleaned
    assert "\x01" not in cleaned
    assert "\r\n" not in cleaned
    assert "\n\n\n" not in cleaned  # collapsed excess blank lines
    assert "double spaces." in cleaned


def test_check_requires_ocr():
    """
    Validates threshold heuristics for flagging scanned or image-only pages.
    """
    assert check_requires_ocr("") is True
    assert check_requires_ocr("   \n   ") is True
    assert check_requires_ocr("Short text") is True  # < 40 chars
    assert check_requires_ocr("This is a sufficiently long line of text that has plenty of characters to extract cleanly.") is False


def test_detect_block_type():
    """
    Validates heuristics for distinguishing headings from normal paragraphs.
    """
    assert detect_block_type("1. Introduction") == "heading"
    assert detect_block_type("1.3.2 Methodology Details") == "heading"
    assert detect_block_type("INTRODUCTION") == "heading"
    assert detect_block_type("This is a standard sentence ending with a period.") == "paragraph"
    assert detect_block_type("This is a short line but it has multiple sentences. It is a paragraph.") == "paragraph"


def test_deterministic_document_id():
    """
    Validates that identical PDF byte arrays yield identical SHA-256 hashes (idempotence).
    """
    content_a = b"%PDF-1.4 mock pdf contents here"
    content_b = b"%PDF-1.4 mock pdf contents here"
    content_c = b"%PDF-1.4 different contents"
    
    hash_a = generate_document_id(content_a)
    hash_b = generate_document_id(content_b)
    hash_c = generate_document_id(content_c)
    
    assert hash_a == hash_b
    assert hash_a != hash_c
    assert len(hash_a) == 64  # valid sha256 hex string length


# ==============================================================================
# PIPELINE INTEGRATION TESTS USING MOCKING
# ==============================================================================

@pytest.fixture
def mock_pdf_file():
    """
    Yields a temporary physical file representing a local PDF path.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.4 mock bytes")
        tmp_path = Path(tmp.name)
    yield tmp_path
    if tmp_path.exists():
        tmp_path.unlink()


@patch("rag.extract.pdf_processor.PdfReader")
def test_valid_multipage_pdf_processing(mock_reader_class, mock_pdf_file):
    """
    Exercises a standard, successful multi-page PDF extraction.
    Validates page counts, page numbering (1-based), block preservation, and metadata.
    """
    # Configure mock page objects
    page_1 = MagicMock()
    page_1.extract_text.return_value = "1. Introduction\n\nThis is paragraph one of page 1."
    
    page_2 = MagicMock()
    page_2.extract_text.return_value = "2. Architecture\n\nThis is paragraph two of page 2."
    
    # Configure mock reader
    mock_reader = MagicMock()
    mock_reader.pages = [page_1, page_2]
    mock_reader.metadata = {"/Title": "CDAC Blueprint", "/Author": "Mentor Team"}
    mock_reader_class.return_value = mock_reader

    # Run extraction
    processor = PDFProcessor()
    doc = processor.process_pdf(mock_pdf_file)
    
    # Assertions
    assert isinstance(doc, DocumentModel)
    assert doc.document_name == mock_pdf_file.name
    assert doc.total_pages == 2
    assert doc.successfully_extracted_pages == 2
    assert doc.failed_pages == 0
    assert doc.extraction_status == "SUCCESS"
    assert doc.metadata["Title"] == "CDAC Blueprint"
    assert doc.metadata["Author"] == "Mentor Team"
    
    # Validate page model preservation and 1-based page index checks
    assert len(doc.pages) == 2
    p1 = doc.pages[0]
    assert p1.page_number == 1
    assert p1.extraction_status == "SUCCESS"
    assert len(p1.blocks) == 2
    assert p1.blocks[0].block_type == "heading"
    assert p1.blocks[0].text == "1. Introduction"
    assert p1.blocks[1].block_type == "paragraph"
    assert p1.blocks[1].text == "This is paragraph one of page 1."
    
    p2 = doc.pages[1]
    assert p2.page_number == 2
    assert p2.blocks[0].text == "2. Architecture"


@patch("rag.extract.pdf_processor.PdfReader")
def test_empty_and_ocr_page_handling(mock_reader_class, mock_pdf_file):
    """
    Validates that empty pages do not crash extraction, but are flagged
    as requiring OCR (image-only) and correctly indexed.
    """
    page_1 = MagicMock()
    page_1.extract_text.return_value = "   "  # whitespace empty page
    
    page_2 = MagicMock()
    page_2.extract_text.return_value = "Scan" # extremely short text (<15 non-whitespace chars), flags OCR
    
    mock_reader = MagicMock()
    mock_reader.pages = [page_1, page_2]
    mock_reader.metadata = None
    mock_reader_class.return_value = mock_reader

    processor = PDFProcessor()
    doc = processor.process_pdf(mock_pdf_file)
    
    assert doc.total_pages == 2
    assert doc.successfully_extracted_pages == 0
    assert doc.pages_requiring_ocr == 2
    assert doc.extraction_status == "SUCCESS"
    
    assert doc.pages[0].requires_ocr is True
    assert doc.pages[0].extraction_status == "IMAGE_ONLY"
    assert doc.pages[1].requires_ocr is True


@patch("rag.extract.pdf_processor.PdfReader")
def test_per_page_failure_handling(mock_reader_class, mock_pdf_file):
    """
    Validates that a crash during a single page's text extraction is isolated,
    recorded as FAILED, and does not stop the entire document extraction.
    The overall document-level status should be PARTIAL_SUCCESS.
    """
    page_1 = MagicMock()
    page_1.extract_text.return_value = "Page 1 extracted content is perfectly fine."
    
    page_2 = MagicMock()
    # Force a runtime exception on page 2
    page_2.extract_text.side_effect = RuntimeError("Extraction stream corruption")
    
    mock_reader = MagicMock()
    mock_reader.pages = [page_1, page_2]
    mock_reader.metadata = {}
    mock_reader_class.return_value = mock_reader

    processor = PDFProcessor()
    doc = processor.process_pdf(mock_pdf_file)
    
    assert doc.total_pages == 2
    assert doc.successfully_extracted_pages == 1
    assert doc.failed_pages == 1
    assert doc.extraction_status == "PARTIAL_SUCCESS"
    
    # Page 1 checks
    assert doc.pages[0].extraction_status == "SUCCESS"
    
    # Page 2 checks
    failed_page = doc.pages[1]
    assert failed_page.page_number == 2
    assert failed_page.extraction_status == "FAILED"
    assert "Extraction stream corruption" in failed_page.extraction_error
    assert failed_page.text == ""
    assert len(failed_page.blocks) == 0


def test_nonexistent_pdf_handling():
    """
    Validates that processing a missing file path raises a FileNotFoundError.
    """
    processor = PDFProcessor()
    nonexistent_path = Path("data/raw/does_not_exist_xyz.pdf")
    
    with pytest.raises(FileNotFoundError):
        processor.process_pdf(nonexistent_path)


@patch("rag.extract.pdf_processor.PdfReader")
def test_corrupted_or_empty_pdf_handling(mock_reader_class, mock_pdf_file):
    """
    Validates that invalid, empty (0 pages), or corrupted PDF headers trigger descriptive failures.
    """
    processor = PDFProcessor()

    # 1. Test 0 pages (empty PDF)
    mock_reader = MagicMock()
    mock_reader.pages = []
    mock_reader_class.return_value = mock_reader
    
    with pytest.raises(ValueError) as excinfo:
        processor.process_pdf(mock_pdf_file)
    assert "Empty PDF file containing 0 pages" in str(excinfo.value)

    # 2. Test corrupted layout parse exception
    mock_reader_class.side_effect = PdfReadError("PDF header is corrupted")
    with pytest.raises(PdfReadError) as excinfo:
        processor.process_pdf(mock_pdf_file)
    assert "Invalid, corrupted, or unreadable PDF" in str(excinfo.value)


@patch("rag.extract.pdf_processor.PdfReader")
def test_json_output_saving(mock_reader_class, mock_pdf_file):
    """
    Validates that a processed DocumentModel is correctly saved into the
    local data/extracted/ directory as valid JSON and retains all page structures.
    """
    page_1 = MagicMock()
    page_1.extract_text.return_value = "Page text to save."
    mock_reader = MagicMock()
    mock_reader.pages = [page_1]
    mock_reader.metadata = None
    mock_reader_class.return_value = mock_reader

    # Load configuration directories
    config = load_config()
    extracted_dir = Path(config.extracted_data_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    processor = PDFProcessor()
    doc = processor.process_pdf(mock_pdf_file)
    
    # Save the file
    saved_file_path = processor.save_extraction(doc)
    assert saved_file_path.exists()
    assert saved_file_path.suffix == ".json"
    
    # Read and parse JSON back
    with open(saved_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["document_id"] == doc.document_id
    assert data["total_pages"] == 1
    assert data["pages"][0]["text"] == "Page text to save."
    assert data["pages"][0]["extraction_status"] == "SUCCESS"

    # Clean up output file
    if saved_file_path.exists():
        saved_file_path.unlink()

