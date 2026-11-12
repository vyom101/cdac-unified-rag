from pathlib import Path
from unittest.mock import MagicMock, patch

from rag.config import load_config
from rag.extract.ocr_processor import (
    OCRPageContext,
    OCRProcessor,
    ocr_dependencies_available,
)


def test_ocr_dependencies_unavailable_when_pytesseract_missing():
    with patch("rag.extract.ocr_processor.PYTESSERACT_AVAILABLE", False):
        assert ocr_dependencies_available() is False


def test_tesseract_binary_missing_is_graceful():
    with patch("rag.extract.ocr_processor.PYTESSERACT_AVAILABLE", True), patch(
        "rag.extract.ocr_processor.pytesseract"
    ) as tesseract:
        tesseract.get_tesseract_version.side_effect = Exception("tesseract not found")
        assert ocr_dependencies_available() is False


def test_processor_disables_gracefully_when_dependencies_missing():
    config = load_config()
    config.ocr_enabled = True
    with patch(
        "rag.extract.ocr_processor.ocr_dependencies_available", return_value=False
    ):
        processor = OCRProcessor(config)
    assert processor.available is False


def test_open_returns_none_when_pymupdf_unavailable():
    with patch("rag.extract.ocr_processor.PYMUPDF_AVAILABLE", False):
        assert OCRProcessor(load_config()).open(Path("irrelevant.pdf")) is None


def test_ocr_page_text_empty_when_unavailable():
    with patch(
        "rag.extract.ocr_processor.ocr_dependencies_available", return_value=False
    ):
        processor = OCRProcessor(load_config())
    assert OCRPageContext(processor, MagicMock()).ocr_page_text(1) == ""


def test_extract_page_images_returns_empty_when_disabled():
    config = load_config()
    config.image_extraction_enabled = False
    context = OCRPageContext(OCRProcessor(config), MagicMock())
    assert context.extract_page_images(1, "doc123") == []


def test_extract_page_images_skips_tiny_images(tmp_path):
    config = load_config()
    config.image_extraction_enabled = True
    config.image_min_dimension = 50
    config.image_output_dir = str(tmp_path / "images")
    processor = OCRProcessor(config)
    page = MagicMock()
    page.get_images.return_value = [(999, 0, 0, 0, 0, 0, 0, 0, 0)]
    document = MagicMock()
    document.__getitem__.return_value = page
    document.extract_image.return_value = {
        "image": b"fakebytes", "ext": "png", "width": 10, "height": 10
    }
    assert OCRPageContext(processor, document).extract_page_images(1, "doc123") == []


def test_extract_page_images_saves_image_without_ocr(tmp_path):
    config = load_config()
    config.image_extraction_enabled = True
    config.image_min_dimension = 10
    config.image_output_dir = str(tmp_path / "images")
    processor = OCRProcessor(config)
    processor._available = False
    page = MagicMock()
    page.get_images.return_value = [(999, 0, 0, 0, 0, 0, 0, 0, 0)]
    document = MagicMock()
    document.__getitem__.return_value = page
    document.extract_image.return_value = {
        "image": b"fakebytes", "ext": "png", "width": 100, "height": 100
    }
    results = OCRPageContext(processor, document).extract_page_images(1, "doc123")
    assert len(results) == 1
    assert results[0].width == 100
    assert results[0].ocr_applied is False
    assert results[0].ocr_text == ""
    assert Path(results[0].file_path).exists()
