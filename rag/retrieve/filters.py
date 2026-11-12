"""Retrieval filter helpers with website domain and unified source filters."""

from __future__ import annotations

from typing import Any, Optional

SOURCE_KINDS = {
    "all",
    "html",
    "pdf",
    "uploaded_pdf",
    "crawled_pdf",
    "html_only",
    "pdf_only",
    "html_and_pdf",
    "uploaded_pdf_only",
}

SOURCE_FILTER_MAP = {
    "html_only": "html",
    "pdf_only": "pdf",
    "html_and_pdf": "all",
    "uploaded_pdf_only": "uploaded_pdf",
}


def resolve_domain(config, domain: str | None = None) -> str | None:
    """Return domain filter when enabled in config."""
    if domain is not None:
        return domain
    if getattr(config, "retrieval_domain_filter_enabled", True):
        return getattr(config, "crawl_allowed_domain", None)
    return None


def normalize_source_filter(source: str | None) -> str:
    if not source:
        return "all"
    value = source.lower()
    if value not in SOURCE_KINDS:
        raise ValueError(f"Unsupported source filter: {source}")
    return SOURCE_FILTER_MAP.get(value, value)


def _is_source_token(value: Any) -> bool:
    return isinstance(value, str) and value.lower() in SOURCE_KINDS


def domain_filter_clause(domain: str | None) -> dict[str, Any] | None:
    if not domain:
        return None
    return {"term": {"allowed_domain": domain}}


def source_domain_filter(
    source: str = "all", domain: Optional[str] = None
) -> Optional[dict[str, Any]]:
    source = normalize_source_filter(source)
    if source == "uploaded_pdf":
        return {"term": {"source_kind": "uploaded_pdf"}}
    if source in {"html", "crawled_pdf"}:
        clauses: list[dict[str, Any]] = [{"term": {"source_kind": source}}]
        if domain:
            clauses.append({"term": {"allowed_domain": domain}})
        return {"bool": {"filter": clauses}}
    if source == "pdf":
        if not domain:
            return {"terms": {"source_kind": ["uploaded_pdf", "crawled_pdf"]}}
        return {
            "bool": {
                "should": [
                    {"term": {"source_kind": "uploaded_pdf"}},
                    {
                        "bool": {
                            "filter": [
                                {"term": {"source_kind": "crawled_pdf"}},
                                {"term": {"allowed_domain": domain}},
                            ]
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }
    if not domain:
        return None
    return {
        "bool": {
            "should": [
                {"term": {"source_kind": "uploaded_pdf"}},
                {
                    "bool": {
                        "filter": [
                            {"terms": {"source_kind": ["html", "crawled_pdf"]}},
                            {"term": {"allowed_domain": domain}},
                        ]
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }


def bm25_query(
    query: str,
    source_or_domain: Any = None,
    domain: Optional[str] = None,
    should: Optional[list[dict[str, Any]]] = None,
    *,
    source: Optional[str] = None,
) -> dict[str, Any]:
    """Build a BM25 query.

    Compatible call shapes:
    - crawler tests: ``bm25_query(query, domain)``
    - unified retriever: ``bm25_query(query, source, domain, should)``
    """
    resolved_source = source
    resolved_domain = domain
    crawler_style = False
    if source_or_domain is not None:
        if _is_source_token(source_or_domain):
            resolved_source = source_or_domain
        else:
            resolved_domain = source_or_domain
            crawler_style = source is None and domain is None and not should

    if crawler_style:
        filt = domain_filter_clause(resolved_domain)
        if filt:
            return {
                "bool": {
                    "must": [{"match": {"content": query}}],
                    "filter": [filt],
                }
            }
        return {"match": {"content": query}}

    content_clause = {"match": {"content": {"query": query}}}
    filt = source_domain_filter(resolved_source or "all", resolved_domain)
    if filt is None and not should:
        return content_clause

    body: dict[str, Any] = {"must": [content_clause]}
    if filt:
        body["filter"] = [filt]
    if should:
        body["should"] = should
    return {"bool": body}


def knn_query(
    query_vector: list[float],
    *,
    k: int,
    num_candidates: int,
    domain: Optional[str] = None,
    source: Optional[str] = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": k,
        "num_candidates": num_candidates,
    }
    if source is None and domain:
        filt = domain_filter_clause(domain)
    else:
        filt = source_domain_filter(source or "all", domain)
    if filt:
        query["filter"] = filt
    return query
