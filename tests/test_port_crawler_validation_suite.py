"""Tests for representative link validation and build metrics."""

import json
from pathlib import Path

from rag.config import AppConfig
from rag.models import CrawlRunReport, PageContentArtifact, PageRecord, RunSummary
from rag.report.reporting import build_pipeline_report
from rag.report.validation import validate_page_trace, validate_representative_links


def _write_minimal_crawl(tmp_path: Path, config: AppConfig, pages: list[PageRecord]):
    report = CrawlRunReport(
        summary=RunSummary(
            run_id="test",
            seed_url="https://cdac.gov.in/",
            allowed_domain="cdac.gov.in",
            started_at="2026-01-01T00:00:00+00:00",
            pages_fetched=len(pages),
            max_effective_depth=max((p.depth for p in pages), default=0),
        ),
        pages=pages,
    )
    out = tmp_path / "crawl_run_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    config.crawl_run_report_path = str(out)


def _write_html_artifact(tmp_path: Path, config: AppConfig, url: str, depth: int, parent: str | None):
    from rag.index.chunker import SectionChunker
    from rag.models import SectionRecord
    from rag.crawl.pipelines import page_id_from_url
    from rag.index.summary_builder import build_chunk_summaries, build_composite_summaries

    page_id = page_id_from_url(url)
    section = SectionRecord(
        section_id=f"{page_id}_sec_000",
        page_id=page_id,
        heading_path=["Test"],
        text="C-DAC operates advanced computing programmes from Pune, India.",
        word_count=8,
        content_hash="abc",
        source_url=url,
    )
    chunks, _ = SectionChunker(min_chunk_size=20).process_sections(
        [section],
        document_name="Test",
        source_url=url,
        parent_url=parent,
        depth=depth,
        sub_link="/test",
    )
    summaries = build_chunk_summaries(chunks)
    composites = build_composite_summaries(summaries, group_size=10)
    artifact = PageContentArtifact(
        page_id=page_id,
        url=url,
        title="Test",
        source_type="html",
        parent_url=parent,
        depth=depth,
        sections=[section],
        chunks=chunks,
        chunk_summaries=summaries,
        composite_summaries=composites,
    )
    extracted = tmp_path / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    (extracted / f"{page_id}.json").write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    config.extracted_data_dir = str(extracted)


def test_validate_page_trace_with_artifact(tmp_path):
    config = AppConfig()
    url = "https://cdac.gov.in/"
    _write_minimal_crawl(
        tmp_path,
        config,
        [
            PageRecord(
                raw_url=url,
                canonical_url=url,
                depth=0,
                status=200,
                title="Home",
            )
        ],
    )
    _write_html_artifact(tmp_path, config, url, depth=0, parent=None)
    result = validate_page_trace(config, url)
    assert result["chunks"] >= 1
    assert result["provenance_valid"] is True
    assert result["depth"] == 0


def test_validate_representative_links_fixture(tmp_path):
    config = AppConfig()
    fixture = tmp_path / "links.json"
    fixture.write_text(
        json.dumps(
            [
                {
                    "label": "home",
                    "url": "https://cdac.gov.in/",
                    "source_type": "html",
                    "expected_depth": 0,
                    "require_parent": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    _write_minimal_crawl(
        tmp_path,
        config,
        [PageRecord(raw_url="https://cdac.gov.in/", canonical_url="https://cdac.gov.in/", depth=0, status=200)],
    )
    _write_html_artifact(tmp_path, config, "https://cdac.gov.in/", depth=0, parent=None)

    report = validate_representative_links(config, fixture)
    assert report["total_links"] == 1
    assert report["complete"] == 1


def test_build_pipeline_report_metrics(tmp_path):
    config = AppConfig()
    url = "https://cdac.gov.in/"
    _write_minimal_crawl(
        tmp_path,
        config,
        [PageRecord(raw_url=url, canonical_url=url, depth=0, status=200, title="Home")],
    )
    _write_html_artifact(tmp_path, config, url, depth=0, parent=None)
    config.pdf_data_dir = str(tmp_path / "pdfs")
    config.pdf_manifest_path = str(tmp_path / "pdfs" / "manifest.jsonl")
    config.kg_store_path = str(tmp_path / "kg" / "graph.json")

    report = build_pipeline_report(config)
    metrics = report["build_metrics"]
    assert metrics["page_count"] == 1
    assert metrics["total_chunks"] >= 1
    assert metrics["total_chunk_summaries"] >= 1
    assert report["crawler_tool"] == "Scrapy (cdac_spider)"

