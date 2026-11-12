"""Unified retrieval result types."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str = ""
    source_url: Optional[str] = None
    parent_url: Optional[str] = None
    sub_link: Optional[str] = None
    source_kind: str = "html"
    content_kind: str = "chunk"
    section: Optional[str] = None
    depth: int = 0
    page_numbers: list[int] = Field(default_factory=list)
    content: str
    score: float
    retrieval_method: str
    rank: int = 0
    chunk_type: str = "text"
    lineage: dict[str, Any] = Field(default_factory=dict)
    figure_refs: list[str] = Field(default_factory=list)
    table_refs: list[str] = Field(default_factory=list)
    equation_refs: list[str] = Field(default_factory=list)
    figure_pages: dict[str, list[int]] = Field(default_factory=dict)
    table_pages: dict[str, list[int]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HybridSearchResult(SearchResult):
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    bm25_score_normalized: Optional[float] = None
    vector_score_normalized: Optional[float] = None
    normalized_keyword_score: float = 0.0
    normalized_semantic_score: float = 0.0
    final_score: float = 0.0
    fusion_strategy: str = "min_max"
    reference_boosted: bool = False
    image_boosted: bool = False


def result_from_hit(hit: dict[str, Any], method: str, rank: int) -> SearchResult:
    source = hit.get("_source", {})
    metadata = source.get("metadata", {})
    return SearchResult(
        chunk_id=source.get("chunk_id") or source.get("record_id") or hit.get("_id", ""),
        document_id=source.get("document_id", ""), document_name=source.get("document_name", ""),
        source_url=source.get("source_url"), parent_url=source.get("parent_url"),
        sub_link=source.get("sub_link"), source_kind=source.get("source_kind", "html"),
        content_kind=source.get("content_kind", "chunk"), section=source.get("section"),
        depth=source.get("depth", 0), page_numbers=source.get("page_numbers") or [],
        content=source.get("content", ""), score=float(hit.get("_score") or 0),
        retrieval_method=method, rank=rank, chunk_type=source.get("chunk_type", "text"),
        lineage=source.get("lineage") or {}, figure_refs=source.get("figure_refs") or [],
        table_refs=source.get("table_refs") or [], equation_refs=source.get("equation_refs") or [],
        figure_pages=source.get("figure_pages") or {}, table_pages=source.get("table_pages") or {},
        metadata=metadata,
    )
