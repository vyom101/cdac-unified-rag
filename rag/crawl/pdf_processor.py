"""Offline crawled-PDF helpers kept for crawl-test compatibility.

Production ``process-pdfs`` uses the pdf-rag extraction path in
``rag.ingestion`` so scanned PDFs are OCR'd instead of being dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from rag.config import AppConfig
from rag.crawl.pdf_downloader import content_hash_bytes, pdf_id_from_url
from rag.index.chunker import SectionChunker
from rag.index.summary_builder import build_chunk_summaries, build_composite_summaries
from rag.models import PageContentArtifact, PdfRecord, SectionRecord


def extract_pdf_pages(pdf_path: str | Path) -> List[Tuple[int, str]]:
    """Extract text per page using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((idx, text))
    return pages


def pages_to_sections(
    pdf_id: str, source_url: str, pages: List[Tuple[int, str]]
) -> List[SectionRecord]:
    sections: list[SectionRecord] = []
    for page_num, text in pages:
        section_id = f"{pdf_id}_page_{page_num:03d}"
        sections.append(
            SectionRecord(
                section_id=section_id,
                page_id=pdf_id,
                heading_path=[f"Page {page_num}"],
                text=text,
                word_count=len(text.split()),
                content_hash=content_hash_bytes(text.encode("utf-8")),
                source_url=source_url,
                metadata={"page_number": page_num, "source_type": "pdf"},
            )
        )
    return sections


def process_pdf_offline(record: PdfRecord, config: AppConfig) -> PageContentArtifact:
    """Process one downloaded PDF into chunks/summaries (offline step)."""
    pages = extract_pdf_pages(record.local_path)
    if not pages:
        raise ValueError(f"No extractable text in PDF: {record.local_path}")

    pdf_id = record.pdf_id or pdf_id_from_url(record.canonical_url)
    sections = pages_to_sections(pdf_id, record.source_url, pages)
    chunker = SectionChunker(
        max_chunk_size=config.max_chunk_size,
        chunk_overlap=config.chunk_overlap,
        min_chunk_size=config.min_chunk_size,
        duplicate_similarity_threshold=config.duplicate_similarity_threshold,
    )
    title = Path(record.local_path).stem
    chunks, _stats = chunker.process_sections(
        sections,
        document_name=title,
        source_url=record.source_url,
        parent_url=record.parent_url,
        depth=record.depth,
        sub_link=record.sub_link,
    )
    for chunk in chunks:
        chunk.source_kind = "crawled_pdf"
        chunk.metadata["source_type"] = "pdf"
        chunk.metadata["pdf_id"] = pdf_id

    chunk_summaries = build_chunk_summaries(chunks)
    composite_summaries = build_composite_summaries(
        chunk_summaries,
        group_size=config.composite_group_size,
    )

    return PageContentArtifact(
        page_id=pdf_id,
        url=record.source_url,
        title=title,
        source_type="pdf",
        parent_url=record.parent_url,
        depth=record.depth,
        sections=sections,
        chunks=chunks,
        chunk_summaries=chunk_summaries,
        composite_summaries=composite_summaries,
    )


def process_all_pending_pdfs(config: AppConfig) -> dict:
    from rag.crawl.pdf_downloader import PdfDownloadStore

    store = PdfDownloadStore(config.pdf_data_dir, config.pdf_manifest_path)
    records = store.list_downloaded()
    output_dir = Path(config.extracted_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    errors: list[str] = []

    for record in records:
        out_path = output_dir / f"pdf_{record.pdf_id}.json"
        if out_path.exists():
            skipped += 1
            continue
        try:
            artifact = process_pdf_offline(record, config)
            out_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
            processed += 1
        except Exception as exc:
            errors.append(f"{record.source_url}: {exc}")

    return {
        "total_in_manifest": len(records),
        "processed": processed,
        "skipped_existing": skipped,
        "errors": errors,
    }
