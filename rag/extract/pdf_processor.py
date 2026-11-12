"""Page- and block-level PDF extraction with optional OCR."""

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag.config import AppConfig, load_config
from rag.extract.ocr_processor import OCRProcessor
from rag.models import BlockModel, DocumentModel, PageModel


def generate_document_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    cleaned = "".join(
        character
        for character in text
        if ord(character) >= 32 or character in {"\t", "\n", "\r"}
    )
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"(\w+)-\s*\n\s*([a-z]\w*)", r"\1\2", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def is_boilerplate_block(text: str, page_number: int) -> bool:
    if page_number == 1:
        return False
    stripped = text.strip()
    lines = stripped.split("\n")
    if len(stripped) >= 150:
        return False
    boilerplate = 0
    nonempty = [line for line in lines if line.strip()]
    for line in nonempty:
        value = line.strip()
        if re.match(r"^[A-Z]{2,4}\s*\d+[-\s]\d+$", value):
            boilerplate += 1
        elif re.match(
            r"^(NAME|ROLL\s*NO|CLASS|SECTION|STUDENT|ID)\s*[:=]",
            value,
            re.IGNORECASE,
        ):
            boilerplate += 1
        elif re.match(r"^(page\s*)?\d+(/\d+)?$", value, re.IGNORECASE):
            boilerplate += 1
        else:
            words = value.split()
            if (
                2 <= len(words) <= 4
                and all(word[0].isupper() for word in words if word)
                and not any(char.isdigit() or char in "[]{}()<>@#$%^&*" for char in value)
            ):
                boilerplate += 1
    return bool(nonempty) and boilerplate >= len(nonempty) / 2


def remove_repeated_boilerplate(
    text: str, page_number: int, total_pages: int
) -> str:
    del total_pages
    if page_number == 1:
        return text
    filtered = []
    for line in text.split("\n"):
        value = line.strip()
        skip = bool(re.match(r"^[A-Z]{2,4}\s*\d+[-\s]\d+$", value))
        skip = skip or bool(re.match(
            r"^(NAME|ROLL\s*NO|CLASS|SECTION|STUDENT|ID|DATE)\s*[:=]",
            value,
            re.IGNORECASE,
        ))
        skip = skip or bool(re.match(r"^(page\s*)?\d+(/\d+)?$", value, re.IGNORECASE))
        skip = skip or bool(re.match(r"^[-–]\s*\d+\s*[-–]$", value))
        words = value.split()
        if (
            2 <= len(words) <= 4
            and all(word[0].isupper() for word in words if word)
            and not any(char.isdigit() or char in "[]{}()<>@#$%^&*" for char in value)
        ):
            skip = True
        if not skip:
            filtered.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(filtered)).strip()


def check_requires_ocr(text: str) -> bool:
    return len(re.sub(r"\s+", "", text)) < 15


def detect_block_type(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return "empty"
    lines = cleaned.split("\n")
    first_line = lines[0].strip()
    if len(cleaned) < 150 and len(lines) <= 2:
        if cleaned[-1] in {".", "?", "!"} and not re.search(r"\b[A-Z]\.$", cleaned):
            return "paragraph"
        patterns = (
            r"^\d+(\.\d+)*\.?\s+[A-Z]",
            r"^[A-Z]\.\s+[A-Z]",
            r"^[●•○]\s+[A-Z]",
            r"^(introduction|theory|methodology|implementation|objective|aim|"
            r"architecture|results|discussion|conclusion|references|appendix|"
            r"abstract|chapter|section|definition|purpose|formula|output|example)\b",
        )
        if (
            any(re.search(pattern, cleaned, re.IGNORECASE) for pattern in patterns)
            or (cleaned.isupper() and any(char.isalpha() for char in cleaned))
            or (first_line.endswith(":") and len(first_line) < 80)
        ):
            return "heading"
    return "paragraph"


class PDFProcessor:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.logger = logging.getLogger("pdf_processor")
        self.ocr_processor = OCRProcessor(self.config)

    def process_pdf(
        self, pdf_path: Path, source_url: Optional[str] = None
    ) -> DocumentModel:
        start_time = time.time()
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Nonexistent PDF path: {pdf_path}")
        try:
            pdf_bytes = pdf_path.read_bytes()
        except Exception as exc:
            raise OSError(f"Failed to read bytes from file '{pdf_path}': {exc}") from exc
        document_id = generate_document_id(pdf_bytes)
        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
        except Exception as exc:
            raise PdfReadError(
                f"Invalid, corrupted, or unreadable PDF: {pdf_path}. Details: {exc}"
            ) from exc
        if total_pages == 0:
            raise ValueError(f"Empty PDF file containing 0 pages: {pdf_path}")

        metadata = {}
        if reader.metadata:
            metadata = {
                key.lstrip("/"): str(value) if value is not None else ""
                for key, value in reader.metadata.items()
            }
        pages: List[PageModel] = []
        successful_pages = failed_pages = ocr_pages = 0
        ocr_context = self.ocr_processor.open(pdf_path)
        try:
            for index in range(total_pages):
                page_number = index + 1
                try:
                    raw_text = reader.pages[index].extract_text() or ""
                    text = remove_repeated_boilerplate(
                        clean_extracted_text(raw_text), page_number, total_pages
                    )
                    requires_ocr = check_requires_ocr(text)
                    ocr_applied = False
                    if (
                        requires_ocr
                        and ocr_context is not None
                        and self.ocr_processor.available
                    ):
                        recovered = clean_extracted_text(
                            ocr_context.ocr_page_text(page_number)
                        )
                        if not check_requires_ocr(recovered):
                            text = recovered
                            ocr_applied = True
                    status = (
                        "IMAGE_ONLY"
                        if requires_ocr and not ocr_applied
                        else "SUCCESS"
                    )
                    if status == "IMAGE_ONLY":
                        ocr_pages += 1
                    else:
                        successful_pages += 1
                    blocks = []
                    raw_blocks = [
                        block.strip() for block in text.split("\n\n") if block.strip()
                    ]
                    for block_index, block_text in enumerate(raw_blocks, start=1):
                        if is_boilerplate_block(block_text, page_number):
                            continue
                        blocks.append(BlockModel(
                            block_id=(
                                f"doc_{document_id}_p{page_number}_b{block_index}"
                            ),
                            page_number=page_number,
                            text=block_text,
                            block_type=detect_block_type(block_text),
                            position_order=block_index,
                        ))
                    images = []
                    if (
                        ocr_context is not None
                        and self.config.image_extraction_enabled
                    ):
                        images = ocr_context.extract_page_images(
                            page_number, document_id
                        )
                    pages.append(PageModel(
                        page_number=page_number,
                        text=text,
                        blocks=blocks,
                        extraction_status=status,
                        requires_ocr=requires_ocr,
                        ocr_applied=ocr_applied,
                        images=images,
                        metadata={},
                    ))
                except Exception as exc:
                    failed_pages += 1
                    pages.append(PageModel(
                        page_number=page_number,
                        text="",
                        blocks=[],
                        extraction_status="FAILED",
                        extraction_error=str(exc),
                        requires_ocr=False,
                        metadata={},
                    ))
        finally:
            if ocr_context is not None:
                ocr_context.close()

        end_time = time.time()
        status = (
            "SUCCESS"
            if failed_pages == 0
            else "PARTIAL_SUCCESS"
            if failed_pages < total_pages
            else "FAILED"
        )
        return DocumentModel(
            document_id=document_id,
            document_name=pdf_path.name,
            source_path=str(pdf_path.resolve()),
            source_url=source_url,
            total_pages=total_pages,
            successfully_extracted_pages=successful_pages,
            failed_pages=failed_pages,
            pages_requiring_ocr=ocr_pages,
            metadata=metadata,
            pages=pages,
            extraction_status=status,
            start_time=start_time,
            end_time=end_time,
            processing_duration_seconds=end_time - start_time,
        )

    def save_extraction(
        self, document: DocumentModel, output_path: Optional[Path] = None
    ) -> Path:
        if output_path is None:
            output_dir = Path(self.config.extracted_data_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{Path(document.source_path).stem}.json"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            document.model_dump_json(indent=2), encoding="utf-8"
        )
        return output_path
