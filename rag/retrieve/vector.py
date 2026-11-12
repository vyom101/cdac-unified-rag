"""Dense-vector retrieval."""

from __future__ import annotations

from typing import Any, List, Optional

from elasticsearch import Elasticsearch

from rag.config import AppConfig, load_config
from rag.retrieve import SearchResult, result_from_hit
from rag.retrieve.filters import knn_query


class VectorRetriever:
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        indexer: Any = None,
        embedder: Any = None,
        embedding_provider: Any = None,
        client: Optional[Elasticsearch] = None,
    ):
        self.config = config or load_config()
        self.index_name = self.config.elasticsearch_index
        self.indexer = indexer
        self._embedding_provider = embedder or embedding_provider
        self.embedder = self._embedding_provider
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

    @property
    def embedding_provider(self):
        if self._embedding_provider is None:
            from rag.index.embedder import get_embedding_provider

            self._embedding_provider = get_embedding_provider(self.config)
            self.embedder = self._embedding_provider
        return self._embedding_provider

    @embedding_provider.setter
    def embedding_provider(self, value) -> None:
        self._embedding_provider = value
        self.embedder = value

    def _validate_query(self, query: str) -> None:
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query)}")
        if not query or query.strip() == "":
            raise ValueError("Query text cannot be empty or whitespace-only.")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        num_candidates: Optional[int] = None,
        *,
        source: Optional[str] = None,
        sources: Optional[str] = None,
        source_filter: Optional[str] = None,
        domain: Optional[str] = None,
        **_: Any,
    ) -> List[SearchResult]:
        self._validate_query(query)
        k = top_k if top_k is not None else self.config.top_k
        candidates = (
            num_candidates if num_candidates is not None else self.config.num_candidates
        )
        if k <= 0:
            raise ValueError(f"top_k must be a positive integer (got {k}).")
        if candidates <= 0:
            raise ValueError(
                f"num_candidates must be a positive integer (got {candidates})."
            )
        if candidates < k:
            raise ValueError(
                f"Validation constraint error: num_candidates ({candidates}) "
                f"must be greater than or equal to top_k ({k})."
            )

        try:
            exists = self.client.indices.exists(index=self.index_name)
        except Exception:
            return []
        if not exists:
            return []

        vector = self.embedding_provider.embed_text(query)
        selected_source = source_filter or sources or source
        knn = knn_query(
            vector,
            k=k,
            num_candidates=candidates,
            source=selected_source,
            domain=domain,
        )
        response = self.client.search(
            index=self.index_name,
            body={"size": k, "knn": knn},
        )
        results = []
        for rank, hit in enumerate(response.get("hits", {}).get("hits", []), 1):
            try:
                results.append(result_from_hit(hit, "vector", rank))
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
                        retrieval_method="vector",
                    )
                )
        return results[:k]
