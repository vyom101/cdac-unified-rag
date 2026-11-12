"""Optional scanned-page and embedded-image OCR for PDF extraction."""

import io
import logging
from pathlib import Path
from typing import Any, List, Optional

from rag.config import AppConfig, load_config
from rag.models import ExtractedImage

logger = logging.getLogger("ocr_processor")
fitz: Any
pytesseract: Any
Image: Any

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    Image = None
    PYTESSERACT_AVAILABLE = False


def ocr_dependencies_available() -> bool:
    if not PYTESSERACT_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception as exc:
        logger.warning("Tesseract OCR binary not found or not runnable: %s", exc)
        return False


class OCRPageContext:
    def __init__(self, processor: "OCRProcessor", fitz_doc: Any):
        self._processor = processor
        self._doc = fitz_doc

    def close(self) -> None:
        try:
            self._doc.close()
        except Exception:
            pass

    def __enter__(self) -> "OCRPageContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def ocr_page_text(self, page_number: int) -> str:
        config = self._processor.config
        if not self._processor.available:
            return ""
        try:
            page = self._doc[page_number - 1]
            matrix = fitz.Matrix(config.ocr_dpi / 72.0, config.ocr_dpi / 72.0)
            pixmap = page.get_pixmap(matrix=matrix)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            return pytesseract.image_to_string(
                image, lang=config.ocr_language
            ) or ""
        except Exception as exc:
            logger.error("OCR failed for page %s: %s", page_number, exc)
            return ""

    def extract_page_images(
        self, page_number: int, document_id: str
    ) -> List[ExtractedImage]:
        config = self._processor.config
        if not config.image_extraction_enabled:
            return []
        results: List[ExtractedImage] = []
        try:
            image_list = self._doc[page_number - 1].get_images(full=True)
            if not image_list:
                return []
            output_dir = Path(config.image_output_dir) / document_id
            output_dir.mkdir(parents=True, exist_ok=True)
            for index, image_info in enumerate(image_list, start=1):
                xref = image_info[0]
                try:
                    raw = self._doc.extract_image(xref)
                    image_bytes = raw["image"]
                    extension = raw.get("ext", "png")
                    width = raw.get("width", 0)
                    height = raw.get("height", 0)
                    if (
                        width < config.image_min_dimension
                        or height < config.image_min_dimension
                    ):
                        continue
                    image_id = f"doc_{document_id}_p{page_number}_img{index}"
                    file_path = output_dir / f"{image_id}.{extension}"
                    file_path.write_bytes(image_bytes)
                    ocr_text = ""
                    ocr_applied = False
                    if self._processor.available:
                        try:
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            ocr_text = pytesseract.image_to_string(
                                pil_image, lang=config.ocr_language
                            ).strip()
                            ocr_applied = True
                        except Exception as exc:
                            logger.warning("Image OCR failed for '%s': %s", image_id, exc)
                    results.append(ExtractedImage(
                        image_id=image_id,
                        page_number=page_number,
                        image_index=index,
                        file_path=str(file_path),
                        width=width,
                        height=height,
                        ocr_text=ocr_text,
                        ocr_applied=ocr_applied,
                    ))
                except Exception as exc:
                    logger.warning(
                        "Failed to extract image xref %s on page %s: %s",
                        xref, page_number, exc,
                    )
        except Exception as exc:
            logger.error("Image extraction failed for page %s: %s", page_number, exc)
        return results


class OCRProcessor:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.logger = logging.getLogger("ocr_processor")
        if self.config.ocr_tesseract_cmd and PYTESSERACT_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = self.config.ocr_tesseract_cmd
        self._available = (
            ocr_dependencies_available() if self.config.ocr_enabled else False
        )
        if self.config.ocr_enabled and not self._available:
            self.logger.warning(
                "OCR is enabled but dependencies are missing; continuing without "
                "OCR text recovery."
            )
        if self.config.image_extraction_enabled and not PYMUPDF_AVAILABLE:
            self.logger.warning(
                "Image extraction is enabled but 'pymupdf' is not installed."
            )

    @property
    def available(self) -> bool:
        return self._available

    def open(self, pdf_path: Path) -> Optional[OCRPageContext]:
        if not PYMUPDF_AVAILABLE:
            return None
        try:
            return OCRPageContext(self, fitz.open(str(pdf_path)))
        except Exception as exc:
            self.logger.warning("Could not open '%s' with PyMuPDF: %s", pdf_path, exc)
            return None
