"""Shared Pydantic v2 models for crawling, PDF processing, indexing, and retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class PageState(str, Enum):
    DISCOVERED = "discovered"
    QUEUED = "queued"
    FETCHED = "fetched"
    SKIPPED = "skipped"
    FAILED = "failed"
    DUPLICATE = "duplicate"


class SkipReason(str, Enum):
    EXTERNAL_DOMAIN = "external_domain"
    UNSUPPORTED_SCHEME = "unsupported_scheme"
    PRIVATE_NETWORK = "private_network"
    ROBOTS_DISALLOWED = "robots_disallowed"
    TRAP_PATTERN = "trap_pattern"
    URL_TOO_LONG = "url_too_long"
    NON_HTML = "non_html"
    IRRELEVANT_BINARY = "irrelevant_binary"
    SOFT_ERROR = "soft_error"
    SAFETY_CAP = "safety_cap"
    DUPLICATE = "duplicate"


class DiscoveryRecord(BaseModel):
    raw_url: str
    canonical_url: str
    parent_url: Optional[str] = None
    depth: int = 0
    state: PageState = PageState.DISCOVERED
    reason: Optional[str] = None
    discovered_at: str = Field(default_factory=utc_now_iso)


class PageRecord(BaseModel):
    raw_url: str
    canonical_url: str
    parent_url: Optional[str] = None
    depth: int = 0
    status: int = 0
    content_type: Optional[str] = None
    title: str = ""
    meta_description: str = ""
    headings: Dict[str, List[str]] = Field(default_factory=dict)
    internal_links_count: int = 0
    is_soft_error: bool = False
    fetch_duration_ms: Optional[float] = None
    redirect_chain: List[str] = Field(default_factory=list)
    fetched_at: str = Field(default_factory=utc_now_iso)


class FailureRecord(BaseModel):
    url: str
    canonical_url: Optional[str] = None
    parent_url: Optional[str] = None
    depth: int = 0
    error_type: str
    error_message: str
    retry_count: int = 0
    failure_kind: str = "http_request"
    needs_human_review: bool = False
    failed_at: str = Field(default_factory=utc_now_iso)


class RobotsObservation(BaseModel):
    url: str
    allowed: bool
    user_agent: str
    source: str = "robots.txt"


class RunSummary(BaseModel):
    run_id: str
    seed_url: str
    allowed_domain: str
    started_at: str
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    termination_reason: str = "in_progress"
    pages_discovered: int = 0
    pages_fetched: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    pages_duplicate: int = 0
    soft_error_pages: int = 0
    max_effective_depth: int = 0
    depth_histogram: Dict[str, int] = Field(default_factory=dict)
    robots_allowed: int = 0
    robots_disallowed: int = 0
    sitemap_urls_seeded: int = 0
    safety_caps_triggered: List[str] = Field(default_factory=list)
    pdf_links_detected: int = 0
    pdf_downloaded: int = 0
    total_chunks: int = 0
    total_chunk_summaries: int = 0
    total_composite_summaries: int = 0
    total_embeddings_indexed: int = 0
    http_retry_events: int = 0
    human_review_required: int = 0
    crawl_run_attempt: int = 1


class CrawlRunReport(BaseModel):
    summary: RunSummary
    pages: List[PageRecord] = Field(default_factory=list)
    failures: List[FailureRecord] = Field(default_factory=list)
    skipped: List[DiscoveryRecord] = Field(default_factory=list)
    robots_observations: List[RobotsObservation] = Field(default_factory=list)


class SectionRecord(BaseModel):
    section_id: str
    page_id: str
    heading_path: List[str] = Field(default_factory=list)
    text: str
    word_count: int = 0
    content_hash: str
    source_url: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkModel(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    section: Optional[str] = None
    page_numbers: List[int] = Field(default_factory=list)
    content: str
    source_url: Optional[str] = None
    parent_url: Optional[str] = None
    sub_link: Optional[str] = None
    depth: int = 0
    chunk_index: int = 0
    content_kind: str = "chunk"
    source_kind: str = "html"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SummaryModel(BaseModel):
    summary_id: str
    chunk_id: str
    document_id: str
    source_url: Optional[str] = None
    parent_url: Optional[str] = None
    sub_link: Optional[str] = None
    depth: int = 0
    summary_text: str
    content_hash: str
    content_kind: str = "chunk_summary"
    source_kind: str = "html"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompositeSummaryModel(BaseModel):
    composite_id: str
    document_id: str
    source_url: Optional[str] = None
    parent_url: Optional[str] = None
    depth: int = 0
    chunk_ids: List[str] = Field(default_factory=list)
    summary_ids: List[str] = Field(default_factory=list)
    composite_text: str
    group_index: int = 0
    group_size: int = 0
    content_hash: str
    content_kind: str = "composite_summary"
    source_kind: str = "html"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PdfRecord(BaseModel):
    pdf_id: str
    source_url: str
    canonical_url: str
    parent_url: Optional[str] = None
    sub_link: Optional[str] = None
    depth: int = 0
    local_path: str
    file_size: int = 0
    content_hash: str
    status: str = "downloaded"
    downloaded_at: str = Field(default_factory=utc_now_iso)


class PageContentArtifact(BaseModel):
    page_id: str
    url: str
    title: str
    source_type: str = "html"
    parent_url: Optional[str] = None
    depth: int = 0
    sections: List[SectionRecord] = Field(default_factory=list)
    chunks: List[ChunkModel] = Field(default_factory=list)
    chunk_summaries: List[SummaryModel] = Field(default_factory=list)
    composite_summaries: List[CompositeSummaryModel] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extracted_at: str = Field(default_factory=utc_now_iso)


class ExtractedImage(BaseModel):
    image_id: str
    page_number: int
    image_index: int
    file_path: str
    width: int = 0
    height: int = 0
    ocr_text: str = ""
    ocr_applied: bool = False


class BlockModel(BaseModel):
    block_id: str
    page_number: int
    text: str
    block_type: str = "paragraph"
    position_order: int = 0


class PageModel(BaseModel):
    page_number: int
    text: str = ""
    blocks: List[BlockModel] = Field(default_factory=list)
    images: List[ExtractedImage] = Field(default_factory=list)
    extraction_status: str = "SUCCESS"
    requires_ocr: bool = False
    ocr_applied: bool = False
    extraction_error: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentModel(BaseModel):
    document_id: str
    document_name: str
    source_path: str
    source_url: Optional[str] = None
    total_pages: int = 0
    successfully_extracted_pages: int = 0
    failed_pages: int = 0
    pages_requiring_ocr: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    pages: List[PageModel] = Field(default_factory=list)
    extraction_status: str = "SUCCESS"
    start_time: float = 0.0
    end_time: float = 0.0
    processing_duration_seconds: float = 0.0


class ChunkingStatistics(BaseModel):
    total_input_pages: int = 0
    total_input_blocks: int = 0
    total_generated_chunks: int = 0
    average_chunk_size: float = 0.0
    minimum_chunk_size: int = 0
    maximum_chunk_size: int = 0
    chunks_spanning_multiple_pages: int = 0
    rejected_chunks: int = 0
    duplicate_chunks_removed: int = 0
    processing_duration_seconds: float = 0.0
    llm_calls_made: int = 0
    merged_low_information_chunks: int = 0
    image_chunks_generated: int = 0
