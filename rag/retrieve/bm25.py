"""BM25 retrieval with optional source filters and PDF reference/image boosts."""

from __future__ import annotations

from typing import Any, List, Optional

from elasticsearch import Elasticsearch

from rag.config import AppConfig, load_config
from rag.extract.reference_extractor import extract_query_references, is_visual_query
from rag.retrieve import SearchResult, result_from_hit
from rag.retrieve.filters import bm25_query


class BM25Retriever:
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        client: Optional[Elasticsearch] = None,
        indexer: Any = None,
    ):
        self.config = config or load_config()
        self.index_name = self.config.elasticsearch_index
        self.indexer = indexer
        if client is not None:
            self.client = client
        elif indexer is not None:
            self.client = indexer.client
        else:
            username = self.config.elasticsearch_username
            if hasattr(username, "get_secret_value"):
                username = username.get_secret_value()
            password = (
                self.config.elasticsearch_password.get_secret_value()
                if self.config.elasticsearch_password
                else None
            )
            basic_auth = (username, password) if username and password else None
            self.client = Elasticsearch(
                self.config.elasticsearch_url,
                basic_auth=basic_auth,
                verify_certs=True,
                request_timeout=30.0,
            )

    def _validate_query(self, query: str) -> None:
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query)}")
        if not query or query.strip() == "":
            raise ValueError("Query text cannot be empty or whitespace-only.")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        source: Optional[str] = None,
        sources: Optional[str] = None,
        source_filter: Optional[str] = None,
        domain: Optional[str] = None,
        **_: Any,
    ) -> List[SearchResult]:
        self._validate_query(query)
        k = top_k if top_k is not None else self.config.top_k
        if k <= 0:
            raise ValueError(f"top_k must be a positive integer (got {k}).")

        try:
            if not self.client.indices.exists(index=self.index_name):
                return []
        except Exception:
            return []

        should_clauses: list[dict[str, Any]] = []
        if getattr(self.config, "enable_reference_metadata", True):
            boost = getattr(self.config, "figure_reference_boost", 4.0)
            query_refs = extract_query_references(query)
            for field in ("figure_refs", "table_refs", "equation_refs"):
                if query_refs.get(field):
                    should_clauses.append(
                        {"terms": {field: query_refs[field], "boost": boost}}
                    )
        if getattr(self.config, "enable_image_query_boost", True) and is_visual_query(query):
            image_boost = getattr(self.config, "image_query_boost", 2.0)
            if image_boost > 0:
                should_clauses.append(
                    {"term": {"chunk_type": {"value": "image", "boost": image_boost}}}
                )

        selected_source = source_filter or sources or source
        if selected_source or domain:
            query_body = bm25_query(
                query, selected_source, domain, should_clauses or None
            )
        elif should_clauses:
            query_body = {
                "bool": {
                    "must": [{"match": {"content": {"query": query}}}],
                    "should": should_clauses,
                }
            }
        else:
            query_body = {"match": {"content": {"query": query}}}

        response = self.client.search(
            index=self.index_name,
            body={"size": k, "query": query_body},
        )
        results = []
        for rank, hit in enumerate(response.get("hits", {}).get("hits", []), 1):
            try:
                results.append(result_from_hit(hit, "bm25", rank))
            except Exception:
                source_doc = hit.get("_source", {})
                results.append(
                    SearchResult(
                        chunk_id=source_doc["chunk_id"],
                        document_id=source_doc["document_id"],
                        document_name=source_doc["document_name"],
                        section=source_doc.get("section"),
                        page_numbers=source_doc.get("page_numbers") or [],
                        content=source_doc["content"],
                        score=float(hit["_score"]),
                        retrieval_method="bm25",
                        chunk_type=source_doc.get("chunk_type") or "text",
                        figure_refs=source_doc.get("figure_refs") or [],
                        table_refs=source_doc.get("table_refs") or [],
                        figure_pages=source_doc.get("figure_pages") or {},
                        table_pages=source_doc.get("table_pages") or {},
                    )
                )
        return results[:k]
