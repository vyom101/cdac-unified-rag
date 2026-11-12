from pathlib import Path
from unittest.mock import MagicMock, patch

from rag.config import load_config
from rag.extract.ocr_processor import OCRPageContext, OCRProcessor, ocr_dependencies_available

# ==============================================================================
# GRACEFUL DEGRADATION TESTS (OCR/image extraction must never crash the pipeline)
# ==============================================================================


def test_ocr_dependencies_unavailable_when_pytesseract_missing():
    """
    If pytesseract itself isn't importable, OCR must be reported unavailable.
    """
    with patch("rag.extract.ocr_processor.PYTESSERACT_AVAILABLE", False):
        assert ocr_dependencies_available() is False


def test_ocr_dependencies_unavailable_when_tesseract_binary_missing():
    """
    pytesseract may be installed while the underlying Tesseract binary is not
    (common on fresh machines) - this must be detected and reported as unavailable.
    """
    with patch("rag.extract.ocr_processor.PYTESSERACT_AVAILABLE", True), \
         patch("rag.extract.ocr_processor.pytesseract") as mock_tess:
        mock_tess.get_tesseract_version.side_effect = Exception("tesseract not found")
        assert ocr_dependencies_available() is False


def test_processor_disables_gracefully_when_dependencies_missing():
    """
    OCRProcessor must not raise even when OCR is enabled in config but the
    underlying dependencies are missing; it should just report unavailable.
    """
    config = load_config()
    config.ocr_enabled = True
    with patch("rag.extract.ocr_processor.ocr_dependencies_available", return_value=False):
        processor = OCRProcessor(config)
    assert processor.available is False


def test_processor_available_when_dependencies_present():
    config = load_config()
    config.ocr_enabled = True
    with patch("rag.extract.ocr_processor.ocr_dependencies_available", return_value=True):
        processor = OCRProcessor(config)
    assert processor.available is True


def test_processor_available_false_when_ocr_disabled_in_config():
    config = load_config()
    config.ocr_enabled = False
    with patch("rag.extract.ocr_processor.ocr_dependencies_available", return_value=True):
        processor = OCRProcessor(config)
    assert processor.available is False


def test_open_returns_none_when_pymupdf_unavailable():
    config = load_config()
    with patch("rag.extract.ocr_processor.PYMUPDF_AVAILABLE", False):
        processor = OCRProcessor(config)
        assert processor.open(Path("irrelevant.pdf")) is None


def test_open_returns_none_on_invalid_pdf_bytes(tmp_path):
    """
    Opening a corrupted/non-PDF file via PyMuPDF must fail gracefully (return None)
    rather than raising, so ingestion of a bad file doesn't crash the pipeline.
    """
    config = load_config()
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"this is definitely not a valid pdf")

    processor = OCRProcessor(config)
    assert processor.open(bad_pdf) is None


# ==============================================================================
# OCR PAGE TEXT RECOVERY TESTS
# ==============================================================================


def test_ocr_page_text_empty_when_unavailable():
    config = load_config()
    with patch("rag.extract.ocr_processor.ocr_dependencies_available", return_value=False):
        processor = OCRProcessor(config)
    ctx = OCRPageContext(processor, MagicMock())
    assert ctx.ocr_page_text(1) == ""


def test_ocr_page_text_handles_internal_errors_gracefully():
    """
    Even if OCR is 'available', unexpected failures while rasterizing/OCR-ing a page
    must be caught and degrade to an empty string rather than propagating.
    """
    config = load_config()
    with patch("rag.extract.ocr_processor.ocr_dependencies_available", return_value=True):
        processor = OCRProcessor(config)

    # A bare MagicMock document will not produce real image bytes, so the OCR
    # pipeline should fail internally and be caught.
    ctx = OCRPageContext(processor, MagicMock())
    assert ctx.ocr_page_text(1) == ""


# ==============================================================================
# EMBEDDED IMAGE EXTRACTION TESTS
# ==============================================================================


def test_extract_page_images_returns_empty_when_disabled():
    config = load_config()
    config.image_extraction_enabled = False
    processor = OCRProcessor(config)
    ctx = OCRPageContext(processor, MagicMock())
    assert ctx.extract_page_images(1, "doc123") == []


def test_extract_page_images_skips_tiny_images(tmp_path):
    """
    Small embedded images (icons, bullets, decorative separators) should be
    skipped based on image_min_dimension, and never saved to disk.
    """
    config = load_config()
    config.image_extraction_enabled = True
    config.image_min_dimension = 50
    config.image_output_dir = str(tmp_path / "images")
    processor = OCRProcessor(config)

    mock_page = MagicMock()
    mock_page.get_images.return_value = [(999, 0, 0, 0, 0, 0, 0, 0, 0)]
    mock_doc = MagicMock()
    mock_doc.__getitem__.return_value = mock_page
    mock_doc.extract_image.return_value = {
        "image": b"fakebytes", "ext": "png", "width": 10, "height": 10
    }

    ctx = OCRPageContext(processor, mock_doc)
    results = ctx.extract_page_images(1, "doc123")

    assert results == []
    assert not (tmp_path / "images").exists() or not any((tmp_path / "images").rglob("*.png"))


def test_extract_page_images_saves_image_without_ocr(tmp_path):
    """
    A sufficiently large embedded image should be saved to disk even when OCR
    is unavailable; ocr_text/ocr_applied should reflect that OCR was skipped.
    """
    config = load_config()
    config.image_extraction_enabled = True
    config.image_min_dimension = 10
    config.image_output_dir = str(tmp_path / "images")
    processor = OCRProcessor(config)
    processor._available = False  # force no-OCR path deterministically

    mock_page = MagicMock()
    mock_page.get_images.return_value = [(999, 0, 0, 0, 0, 0, 0, 0, 0)]
    mock_doc = MagicMock()
    mock_doc.__getitem__.return_value = mock_page
    mock_doc.extract_image.return_value = {
        "image": b"fakebytes", "ext": "png", "width": 100, "height": 100
    }

    ctx = OCRPageContext(processor, mock_doc)
    results = ctx.extract_page_images(1, "doc123")

    assert len(results) == 1
    image = results[0]
    assert image.width == 100
    assert image.height == 100
    assert image.ocr_applied is False
    assert image.ocr_text == ""
    assert Path(image.file_path).exists()
    assert image.page_number == 1
    assert image.image_index == 1


def test_extract_page_images_handles_extraction_failure_gracefully():
    """
    If a single image xref fails to extract, the whole page's extraction must
    not crash - it should just skip that image.
    """
    config = load_config()
    config.image_extraction_enabled = True
    processor = OCRProcessor(config)

    mock_page = MagicMock()
    mock_page.get_images.return_value = [(999, 0, 0, 0, 0, 0, 0, 0, 0)]
    mock_doc = MagicMock()
    mock_doc.__getitem__.return_value = mock_page
    mock_doc.extract_image.side_effect = RuntimeError("corrupted image stream")

    ctx = OCRPageContext(processor, mock_doc)
    results = ctx.extract_page_images(1, "doc123")

    assert results == []

