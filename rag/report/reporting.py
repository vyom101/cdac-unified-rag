"""Structured run/build reporting with full pipeline metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from rag.config import AppConfig
from rag.crawl.pdf_downloader import PdfDownloadStore
from rag.graph.store import KnowledgeGraphStore
from rag.models import CrawlRunReport, PageContentArtifact


def load_crawl_report(path: str | Path) -> CrawlRunReport:
    return CrawlRunReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _scan_content_artifacts(extracted_dir: Path) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "html_pages": 0, "pdf_pages": 0, "total_sections": 0,
        "total_chunks": 0, "total_chunk_summaries": 0,
        "total_composite_summaries": 0,
    }
    pages = []
    for path in extracted_dir.glob("*.json"):
        if path.name == "pages_index.json":
            continue
        try:
            artifact = PageContentArtifact.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except ValueError:
            continue
        totals[
            "pdf_pages"
            if artifact.source_type in {"pdf", "uploaded_pdf", "crawled_pdf"}
            else "html_pages"
        ] += 1
        totals["total_sections"] += len(artifact.sections)
        totals["total_chunks"] += len(artifact.chunks)
        totals["total_chunk_summaries"] += len(artifact.chunk_summaries)
        totals["total_composite_summaries"] += len(artifact.composite_summaries)
        pages.append({
            "page_id": artifact.page_id, "url": artifact.url,
            "source_type": artifact.source_type, "depth": artifact.depth,
            "parent_url": artifact.parent_url, "chunks": len(artifact.chunks),
            "chunk_summaries": len(artifact.chunk_summaries),
            "composite_summaries": len(artifact.composite_summaries),
        })
    totals.update({"pages": pages, "pages_extracted": len(pages)})
    return totals


def _index_metrics(config: AppConfig) -> dict[str, Any]:
    metrics: dict[str, Any] = {"status": "unknown", "total_documents": 0}
    try:
        from rag.index.indexer import ElasticsearchIndexer

        indexer = ElasticsearchIndexer(config)
        if not indexer.index_exists():
            metrics["status"] = "missing"
            return metrics
        metrics.update({
            "status": "ready",
            "total_documents": int(indexer.client.count(index=indexer.index_name).get("count", 0)),
        })
        response = indexer.client.search(
            index=indexer.index_name,
            body={"size": 0, "aggs": {"by_kind": {"terms": {"field": "content_kind", "size": 20}}}},
        )
        buckets = response.get("aggregations", {}).get("by_kind", {}).get("buckets", [])
        metrics["content_kinds"] = {bucket["key"]: bucket["doc_count"] for bucket in buckets}
    except Exception as error:
        metrics.update({"status": "error", "error": str(error)})
    return metrics


def build_pipeline_report(config: AppConfig) -> Dict[str, Any]:
    report_path = Path(config.crawl_run_report_path)
    extracted_dir = Path(config.extracted_data_dir)
    kg_path = Path(config.kg_store_path)
    crawl_summary: dict[str, Any] = {"status": "missing"}
    failure_count = 0
    if report_path.exists():
        crawl = load_crawl_report(report_path)
        crawl_summary = crawl.summary.model_dump()
        crawl_summary["status"] = "ready"
        failure_count = len(crawl.failures)
    content = _scan_content_artifacts(extracted_dir) if extracted_dir.exists() else {"pages_extracted": 0}
    pdf_manifest = PdfDownloadStore(config.pdf_data_dir, config.pdf_manifest_path).list_downloaded()
    pdf_metrics = {
        "manifest_entries": len(pdf_manifest),
        "processed_artifacts": content.get("pdf_pages", 0),
        "pending_processing": max(0, len(pdf_manifest) - content.get("pdf_pages", 0)),
    }
    kg_summary = {"status": "missing"}
    if kg_path.exists():
        graph = KnowledgeGraphStore(str(kg_path))
        kg_summary = {
            "status": "ready", "nodes": len(graph.snapshot.nodes),
            "edges": len(graph.snapshot.edges),
            "schema_version": graph.snapshot.schema_version,
        }
    index_metrics = _index_metrics(config)
    build_metrics = {
        "page_count": crawl_summary.get("pages_fetched", 0),
        "pages_discovered": crawl_summary.get("pages_discovered", 0),
        "max_effective_depth": crawl_summary.get("max_effective_depth", 0),
        "depth_histogram": crawl_summary.get("depth_histogram", {}),
        "pdf_links_detected": crawl_summary.get("pdf_links_detected", 0),
        "pdf_downloaded": crawl_summary.get("pdf_downloaded", len(pdf_manifest)),
        "html_pages_extracted": content.get("html_pages", 0),
        "pdf_pages_processed": content.get("pdf_pages", 0),
        "total_sections": content.get("total_sections", 0),
        "total_chunks": content.get("total_chunks", 0),
        "total_chunk_summaries": content.get("total_chunk_summaries", 0),
        "total_composite_summaries": content.get("total_composite_summaries", 0),
        "embeddings_indexed": index_metrics.get("total_documents", 0),
        "http_failures": failure_count,
        "human_review_required": crawl_summary.get("human_review_required", 0),
        "http_retry_events": crawl_summary.get("http_retry_events", 0),
        "duration_seconds": crawl_summary.get("duration_seconds"),
        "termination_reason": crawl_summary.get("termination_reason"),
    }
    return {
        "crawler_tool": "Scrapy (cdac_spider)", "kb_source": config.kb_source,
        "build_metrics": build_metrics, "crawl": crawl_summary, "content": content,
        "pdf": pdf_metrics, "knowledge_graph": kg_summary,
        "elasticsearch": {"index": config.elasticsearch_index, **index_metrics},
    }


def write_pipeline_report(config: AppConfig, output_path: str | Path) -> Dict[str, Any]:
    report = build_pipeline_report(config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
