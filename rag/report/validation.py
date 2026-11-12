"""Page-to-knowledge validation and representative link checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from rag.config import AppConfig
from rag.crawl.pdf_downloader import PdfDownloadStore, pdf_id_from_url
from rag.crawl.pipelines import page_id_from_url
from rag.graph.store import KnowledgeGraphStore
from rag.index.indexer import ElasticsearchIndexer
from rag.models import CrawlRunReport, PageContentArtifact, PageRecord


def _load_crawl_report(config: AppConfig) -> CrawlRunReport | None:
    path = Path(config.crawl_run_report_path)
    return CrawlRunReport.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None


def _find_page_record(crawl: CrawlRunReport | None, page_url: str) -> PageRecord | None:
    if crawl:
        return next(
            (page for page in crawl.pages if page.canonical_url == page_url or page.raw_url == page_url),
            None,
        )
    return None


def _validate_lineage(artifact: PageContentArtifact, page_url: str) -> list[str]:
    issues = []
    for chunk in artifact.chunks:
        if chunk.source_url and chunk.source_url not in {page_url, artifact.url}:
            issues.append(f"Chunk {chunk.chunk_id} source_url mismatch")
        if not chunk.document_id:
            issues.append(f"Chunk {chunk.chunk_id} missing document_id")
        if artifact.source_type == "html" and artifact.depth > 0 and not chunk.parent_url:
            issues.append(f"Chunk {chunk.chunk_id} missing parent_url at depth {artifact.depth}")
    for summary in artifact.chunk_summaries:
        if summary.source_url and summary.source_url not in {page_url, artifact.url}:
            issues.append(f"Summary {summary.summary_id} source_url mismatch")
    return issues


def validate_page_trace(config: AppConfig, page_url: str) -> Dict[str, Any]:
    page_id = page_id_from_url(page_url)
    artifact_path = Path(config.extracted_data_dir) / f"{page_id}.json"
    result: dict[str, Any] = {
        "page_url": page_url, "page_id": page_id, "source_type": "html",
        "crawl_record": None, "parent_url": None, "sub_link": None, "depth": None,
        "sections": 0, "chunks": 0, "chunk_summaries": 0, "composite_summaries": 0,
        "elasticsearch_records": 0, "kg_entities": 0, "kg_edges": 0,
        "provenance_valid": False, "status": "partial", "issues": [],
    }
    page_record = _find_page_record(_load_crawl_report(config), page_url)
    if page_record:
        result.update({
            "crawl_record": page_record.model_dump(),
            "parent_url": page_record.parent_url, "depth": page_record.depth,
        })
    else:
        result["issues"].append("Page not found in crawl report")
    if artifact_path.exists():
        artifact = PageContentArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
        result.update({
            "source_type": artifact.source_type,
            "parent_url": artifact.parent_url or result["parent_url"],
            "depth": artifact.depth if artifact.depth else result["depth"],
            "sections": len(artifact.sections), "chunks": len(artifact.chunks),
            "chunk_summaries": len(artifact.chunk_summaries),
            "composite_summaries": len(artifact.composite_summaries),
            "sub_link": artifact.chunks[0].sub_link if artifact.chunks else None,
        })
        lineage_issues = _validate_lineage(artifact, page_url)
        result["issues"].extend(lineage_issues)
        result["provenance_valid"] = not lineage_issues
    else:
        result["issues"].append("Content artifact missing")
    try:
        indexer = ElasticsearchIndexer(config)
        if indexer.index_exists():
            response = indexer.client.count(
                index=indexer.index_name,
                body={"query": {"term": {"document_id": page_id}}},
            )
            result["elasticsearch_records"] = int(response.get("count", 0))
    except Exception as error:
        result["issues"].append(f"Elasticsearch check failed: {error}")
    kg_path = Path(config.kg_store_path)
    if kg_path.exists():
        graph = KnowledgeGraphStore(str(kg_path))
        entities = [
            node for node in graph.snapshot.nodes
            if any(provenance.source_url == page_url for provenance in node.provenance)
        ]
        identifiers = {entity.entity_id for entity in entities}
        result["kg_entities"] = len(entities)
        result["kg_edges"] = len([
            edge for edge in graph.snapshot.edges
            if edge.source_entity_id in identifiers or edge.target_entity_id in identifiers
        ])
    else:
        result["issues"].append("Knowledge graph store missing")
    non_optional = [issue for issue in result["issues"] if issue != "Knowledge graph store missing"]
    if result["provenance_valid"] and not non_optional and result["chunks"] and result["crawl_record"]:
        result["status"] = "complete"
    return result


def validate_pdf_trace(config: AppConfig, pdf_url: str) -> Dict[str, Any]:
    pdf_id = pdf_id_from_url(pdf_url)
    artifact_path = Path(config.extracted_data_dir) / f"pdf_{pdf_id}.json"
    result: dict[str, Any] = {
        "pdf_url": pdf_url, "pdf_id": pdf_id, "source_type": "pdf",
        "downloaded": False, "processed": False, "parent_url": None,
        "sub_link": None, "depth": None, "chunks": 0, "chunk_summaries": 0,
        "composite_summaries": 0, "provenance_valid": False,
        "status": "partial", "issues": [],
    }
    records = {
        record.source_url: record
        for record in PdfDownloadStore(config.pdf_data_dir, config.pdf_manifest_path).list_downloaded()
    }
    record = records.get(pdf_url)
    if record:
        result.update({
            "downloaded": True, "parent_url": record.parent_url,
            "sub_link": record.sub_link, "depth": record.depth,
        })
    else:
        result["issues"].append("PDF not found in download manifest")
    if artifact_path.exists():
        artifact = PageContentArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
        result.update({
            "processed": True, "chunks": len(artifact.chunks),
            "chunk_summaries": len(artifact.chunk_summaries),
            "composite_summaries": len(artifact.composite_summaries),
        })
        lineage_issues = _validate_lineage(artifact, pdf_url)
        result["issues"].extend(lineage_issues)
        result["provenance_valid"] = not lineage_issues
    else:
        result["issues"].append("PDF content artifact missing (run process-pdfs)")
    if result["downloaded"] and result["processed"] and result["provenance_valid"]:
        result["status"] = "complete"
    return result


def load_representative_links(path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("links", [])


def validate_representative_links(
    config: AppConfig, links_path: str | Path | None = None
) -> Dict[str, Any]:
    fixture = Path(links_path or config.validation_representative_links_path)
    links = load_representative_links(fixture)
    results = []
    complete = 0
    for entry in links:
        trace = (
            validate_pdf_trace(config, entry["url"])
            if entry.get("source_type", "html") == "pdf"
            else validate_page_trace(config, entry["url"])
        )
        expected_depth = entry.get("expected_depth")
        if expected_depth is not None and trace.get("depth") is not None and trace["depth"] != expected_depth:
            trace["issues"].append(f"Depth mismatch: expected {expected_depth}, got {trace['depth']}")
            trace["status"] = "partial"
        if entry.get("require_parent") and not trace.get("parent_url"):
            trace["issues"].append("Missing required parent_url")
            trace["status"] = "partial"
        trace.update({"label": entry.get("label"), "expected_depth": expected_depth})
        complete += int(trace.get("status") == "complete")
        results.append(trace)
    return {
        "fixture": str(fixture), "total_links": len(links), "complete": complete,
        "partial_or_missing": len(links) - complete,
        "success_pct": round(100.0 * complete / len(links), 2) if links else 0.0,
        "results": results,
    }
