"""Unified PDF processing and corpus artifact/index orchestration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from rag.config import AppConfig, load_config
from rag.crawl.pdf_downloader import PdfDownloadStore
from rag.extract.pdf_processor import PDFProcessor
from rag.index.indexer import (
    IndexableRecord,
    chunk_to_indexable,
    composite_to_indexable,
    summary_to_indexable,
)
from rag.index.pdf_chunker import Chunker
from rag.index.summary_builder import build_chunk_summaries, build_composite_summaries
from rag.models import ChunkModel, PageContentArtifact, PdfRecord


def _artifact_path(
    config: AppConfig, document_id: str, source_kind: str, pdf_id: str | None = None
) -> Path:
    prefix = "pdf" if source_kind == "crawled_pdf" else "uploaded_pdf"
    return Path(config.extracted_data_dir) / f"{prefix}_{pdf_id or document_id}.json"


def process_pdf(
    pdf_path: str | Path,
    config: AppConfig | None = None,
    *,
    source_kind: str = "uploaded_pdf",
    crawl_record: PdfRecord | None = None,
) -> PageContentArtifact:
    """Extract and chunk either an uploaded or crawled PDF into one artifact shape."""
    config = config or load_config()
    if source_kind not in {"uploaded_pdf", "crawled_pdf"}:
        raise ValueError("PDF source_kind must be uploaded_pdf or crawled_pdf")
    path = Path(pdf_path)
    source_url = crawl_record.source_url if crawl_record else None
    document = PDFProcessor(config).process_pdf(path, source_url=source_url)
    chunks, statistics = Chunker(config).process_document(document)
    lineage: dict[str, Any] = {
        "source_kind": source_kind,
        "source_path": str(path.resolve()),
        "document_id": document.document_id,
    }
    parent_url = sub_link = None
    depth = 0
    if crawl_record:
        parent_url = crawl_record.parent_url
        sub_link = crawl_record.sub_link
        depth = crawl_record.depth
        lineage.update({
            "pdf_id": crawl_record.pdf_id,
            "canonical_url": crawl_record.canonical_url,
            "parent_url": parent_url,
            "sub_link": sub_link,
            "depth": depth,
            "downloaded_at": crawl_record.downloaded_at,
            "content_hash": crawl_record.content_hash,
        })
    enriched: list[ChunkModel] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata["lineage"] = dict(lineage)
        enriched.append(chunk.model_copy(update={
            "source_kind": source_kind,
            "source_url": source_url,
            "parent_url": parent_url,
            "sub_link": sub_link,
            "depth": depth,
            "metadata": metadata,
        }))
    summaries = [
        summary.model_copy(update={
            "metadata": {**summary.metadata, "lineage": dict(lineage)}
        })
        for summary in build_chunk_summaries(enriched)
    ]
    composites = [
        composite.model_copy(update={
            "metadata": {**composite.metadata, "lineage": dict(lineage)}
        })
        for composite in build_composite_summaries(
        summaries, group_size=config.composite_group_size
        )
    ]
    artifact = PageContentArtifact(
        page_id=crawl_record.pdf_id if crawl_record else document.document_id,
        url=source_url or path.resolve().as_uri(),
        title=document.document_name,
        source_type=source_kind,
        parent_url=parent_url,
        depth=depth,
        chunks=enriched,
        chunk_summaries=summaries,
        composite_summaries=composites,
        metadata={
            "source_kind": source_kind,
            "lineage": lineage,
            "extraction": document.model_dump(),
            "chunking_statistics": statistics.model_dump(),
        },
    )
    output = _artifact_path(
        config, document.document_id, source_kind,
        crawl_record.pdf_id if crawl_record else None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return artifact


def process_crawled_pdfs(config: AppConfig | None = None) -> list[PageContentArtifact]:
    config = config or load_config()
    records = PdfDownloadStore(
        config.pdf_data_dir, config.pdf_manifest_path
    ).list_downloaded()
    return [
        process_pdf(
            record.local_path, config,
            source_kind="crawled_pdf", crawl_record=record,
        )
        for record in records
    ]


def ingest_uploaded_pdf(
    pdf_path: str | Path, config: AppConfig | None = None
) -> PageContentArtifact:
    config = config or load_config()
    source = Path(pdf_path)
    if not source.exists():
        raise FileNotFoundError(f"PDF not found: {source}")
    raw_dir = Path(config.raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return process_pdf(destination, config, source_kind="uploaded_pdf")


def ingest_all_uploaded_pdfs(
    config: AppConfig | None = None
) -> list[PageContentArtifact]:
    config = config or load_config()
    raw_dir = Path(config.raw_data_dir)
    if not raw_dir.exists():
        return []
    return [
        process_pdf(path, config, source_kind="uploaded_pdf")
        for path in sorted(raw_dir.glob("*.pdf"))
    ]


def iter_artifacts(config: AppConfig | None = None) -> Iterable[PageContentArtifact]:
    config = config or load_config()
    directory = Path(config.extracted_data_dir)
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.json")):
        if path.name == "pages_index.json":
            continue
        try:
            yield PageContentArtifact.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (ValueError, json.JSONDecodeError):
            continue


def _selected(source: str, source_kind: str) -> bool:
    allowed = {
        "html_only": {"html"},
        "pdf_only": {"uploaded_pdf", "crawled_pdf"},
        "html_and_pdf": {"html", "uploaded_pdf", "crawled_pdf"},
        "uploaded_pdf_only": {"uploaded_pdf"},
        "all": {"html", "uploaded_pdf", "crawled_pdf"},
    }
    return source_kind in allowed[source]


def load_index_records(
    config: AppConfig | None = None,
) -> list[IndexableRecord]:
    """Load all artifact record types selected by the exact kb.source value."""
    config = config or load_config()
    records: list[IndexableRecord] = []
    for artifact in iter_artifacts(config):
        source_kind = (
            artifact.chunks[0].source_kind
            if artifact.chunks
            else artifact.metadata.get("source_kind", artifact.source_type)
        )
        if not _selected(config.kb_source, source_kind):
            continue
        records.extend(
            chunk_to_indexable(chunk, artifact.title, source_kind=source_kind)
            for chunk in artifact.chunks
        )
        records.extend(
            summary_to_indexable(summary, artifact.title)
            for summary in artifact.chunk_summaries
        )
        records.extend(
            composite_to_indexable(composite, artifact.title)
            for composite in artifact.composite_summaries
        )
    return records


def load_all_chunks(config: AppConfig | None = None) -> list[ChunkModel]:
    config = config or load_config()
    return [
        chunk
        for artifact in iter_artifacts(config)
        for chunk in artifact.chunks
        if _selected(config.kb_source, chunk.source_kind)
    ]


def index_corpus(config: AppConfig | None = None) -> dict[str, Any]:
    config = config or load_config()
    from rag.index.embedder import get_embedding_provider
    from rag.index.indexer import ElasticsearchIndexer

    records = load_index_records(config)
    indexer = ElasticsearchIndexer(config)
    indexer.create_index_if_not_exists()
    embeddings = get_embedding_provider(config).embed_documents(
        [record.content for record in records]
    ) if records else []
    return indexer.index_records(records, embeddings)
