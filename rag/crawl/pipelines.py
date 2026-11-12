"""Crawler extraction and PDF persistence pipelines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rag.crawl.html_extractor import extract_sections
from rag.crawl.pdf_downloader import PdfDownloadStore
from rag.index.chunker import SectionChunker
from rag.index.summary_builder import build_chunk_summaries, build_composite_summaries
from rag.models import PageContentArtifact


def page_id_from_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


class ContentExtractionPipeline:
    def __init__(
        self, output_dir: str, composite_group_size: int = 10, *,
        max_chunk_size: int = 1200, chunk_overlap: int = 150,
        min_chunk_size: int = 80, duplicate_similarity_threshold: float = 0.92,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunker = SectionChunker(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            duplicate_similarity_threshold=duplicate_similarity_threshold,
        )
        self.composite_group_size = composite_group_size
        self.artifacts: list[PageContentArtifact] = []

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            crawler.settings.get("CONTENT_ARTIFACT_DIR", "data/extracted"),
            composite_group_size=int(crawler.settings.get("COMPOSITE_GROUP_SIZE", 10)),
            max_chunk_size=int(crawler.settings.get("HTML_MAX_CHUNK_SIZE", 1200)),
            chunk_overlap=int(crawler.settings.get("HTML_CHUNK_OVERLAP", 150)),
            min_chunk_size=int(crawler.settings.get("HTML_MIN_CHUNK_SIZE", 80)),
            duplicate_similarity_threshold=float(
                crawler.settings.get("DUPLICATE_SIMILARITY_THRESHOLD", 0.92)
            ),
        )

    def process_item(self, item, spider):
        if not isinstance(item, dict) or item.get("record_type"):
            return item
        if not item.get("content_type", "").lower().startswith("text/html"):
            return item
        response = item.pop("_response", None)
        if response is None:
            return item
        url = item.get("canonical_url") or item.get("raw_url")
        page_id = page_id_from_url(url)
        parent_url = item.get("parent_url")
        depth = int(item.get("depth", 0))
        sub_link = item.get("sub_link")
        sections = extract_sections(
            response.text, page_id=page_id, source_url=url, title=item.get("title", "")
        )
        chunks, stats = self.chunker.process_sections(
            sections,
            document_name=item.get("title") or url,
            source_url=url,
            parent_url=parent_url,
            depth=depth,
            sub_link=sub_link,
        )
        chunk_summaries = build_chunk_summaries(chunks)
        composite_summaries = build_composite_summaries(
            chunk_summaries, group_size=self.composite_group_size
        )
        artifact = PageContentArtifact(
            page_id=page_id,
            url=url,
            title=item.get("title", ""),
            source_type="html",
            parent_url=parent_url,
            depth=depth,
            sections=sections,
            chunks=chunks,
            chunk_summaries=chunk_summaries,
            composite_summaries=composite_summaries,
        )
        self.artifacts.append(artifact)
        (self.output_dir / f"{page_id}.json").write_text(
            artifact.model_dump_json(indent=2), encoding="utf-8"
        )
        item.update({
            "page_id": page_id, "section_count": len(sections), "chunk_count": len(chunks),
            "chunk_summary_count": len(chunk_summaries),
            "composite_summary_count": len(composite_summaries),
            "chunking_stats": stats.model_dump(),
        })
        return item

    def close_spider(self, spider):
        index = [{
            "page_id": artifact.page_id, "url": artifact.url, "title": artifact.title,
            "parent_url": artifact.parent_url, "depth": artifact.depth,
            "source_type": artifact.source_type, "chunks": len(artifact.chunks),
            "chunk_summaries": len(artifact.chunk_summaries),
            "composite_summaries": len(artifact.composite_summaries),
        } for artifact in self.artifacts]
        (self.output_dir / "pages_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        ledger = getattr(spider, "crawl_ledger", None)
        if ledger:
            ledger.summary.total_chunks = sum(len(a.chunks) for a in self.artifacts)
            ledger.summary.total_chunk_summaries = sum(len(a.chunk_summaries) for a in self.artifacts)
            ledger.summary.total_composite_summaries = sum(
                len(a.composite_summaries) for a in self.artifacts
            )
        spider.logger.info("Content artifacts: %d pages written to %s", len(self.artifacts), self.output_dir)


class PdfDownloadPipeline:
    def __init__(self, pdf_dir: str, manifest_path: str):
        self.store = PdfDownloadStore(pdf_dir, manifest_path)
        self.download_count = 0

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            crawler.settings.get("PDF_DATA_DIR", "data/pdfs"),
            crawler.settings.get("PDF_MANIFEST_PATH", "data/pdfs/manifest.jsonl"),
        )

    def process_item(self, item, spider):
        if not isinstance(item, dict) or item.get("record_type") != "pdf_download":
            return item
        body = item.pop("_body", None)
        if body is None:
            return item
        metadata = item["pdf_meta"]
        record = self.store.save(
            source_url=metadata["source_url"],
            canonical_url=metadata["canonical_url"],
            body=body,
            parent_url=metadata.get("parent_url"),
            sub_link=metadata.get("sub_link"),
            depth=int(metadata.get("depth", 0)),
        )
        self.download_count += 1
        ledger = getattr(spider, "crawl_ledger", None)
        if ledger:
            ledger.record_pdf_downloaded()
        item["pdf_record"] = record.model_dump()
        spider.logger.info("PDF downloaded: %s -> %s", record.source_url, record.local_path)
        return item

    def close_spider(self, spider):
        spider.logger.info("PDF downloads this run: %d", self.download_count)
