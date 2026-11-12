"""Unified Elasticsearch schema and idempotent record indexing."""

from __future__ import annotations

import time
from typing import Any, Iterable, Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pydantic import BaseModel, Field, SecretStr

from rag.config import AppConfig, load_config
from rag.models import ChunkModel, CompositeSummaryModel, SummaryModel


class IndexableRecord(BaseModel):
    record_id: str
    document_id: str
    document_name: str = ""
    content: str
    source_kind: str
    content_kind: str = "chunk"
    source_url: Optional[str] = None
    parent_url: Optional[str] = None
    sub_link: Optional[str] = None
    allowed_domain: Optional[str] = None
    depth: int = 0
    section: Optional[str] = None
    page_numbers: list[int] = Field(default_factory=list)
    chunk_index: int = 0
    chunk_type: str = "text"
    lineage: dict[str, Any] = Field(default_factory=dict)
    figure_refs: list[str] = Field(default_factory=list)
    table_refs: list[str] = Field(default_factory=list)
    equation_refs: list[str] = Field(default_factory=list)
    figure_captions: list[str] = Field(default_factory=list)
    table_captions: list[str] = Field(default_factory=list)
    figure_pages: dict[str, list[int]] = Field(default_factory=dict)
    table_pages: dict[str, list[int]] = Field(default_factory=dict)
    image_path: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def chunk_to_indexable(
    chunk: ChunkModel, document_name: Optional[str] = None, *, source_kind: Optional[str] = None
) -> IndexableRecord:
    meta = chunk.metadata
    kind = source_kind or chunk.source_kind
    if source_kind is None and kind == "html" and chunk.page_numbers:
        kind = "uploaded_pdf"
    if kind == "uploaded_pdf" and chunk.source_url and chunk.parent_url:
        kind = "crawled_pdf"
    return IndexableRecord(
        record_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_name=document_name or chunk.document_name,
        content=chunk.content,
        source_kind=kind,
        content_kind=chunk.content_kind,
        source_url=chunk.source_url,
        parent_url=chunk.parent_url,
        sub_link=chunk.sub_link,
        depth=chunk.depth,
        section=chunk.section,
        page_numbers=chunk.page_numbers,
        chunk_index=chunk.chunk_index,
        chunk_type=meta.get("chunk_type", "text"),
        lineage=meta.get("lineage", {}),
        figure_refs=meta.get("figure_refs", []),
        table_refs=meta.get("table_refs", []),
        equation_refs=meta.get("equation_refs", []),
        figure_captions=meta.get("figure_captions", []),
        table_captions=meta.get("table_captions", []),
        figure_pages=meta.get("figure_pages", {}),
        table_pages=meta.get("table_pages", {}),
        image_path=meta.get("image_path"),
        metadata=meta,
    )


def summary_to_indexable(summary: SummaryModel, document_name: str = "") -> IndexableRecord:
    return IndexableRecord(
        record_id=summary.summary_id,
        document_id=summary.document_id,
        document_name=document_name,
        content=summary.summary_text,
        source_kind=summary.source_kind,
        content_kind=summary.content_kind,
        source_url=summary.source_url,
        parent_url=summary.parent_url,
        sub_link=summary.sub_link,
        depth=summary.depth,
        section="summary",
        lineage={**summary.metadata.get("lineage", {}), "chunk_id": summary.chunk_id},
        metadata=summary.metadata,
    )


def composite_to_indexable(
    composite: CompositeSummaryModel, document_name: str = ""
) -> IndexableRecord:
    return IndexableRecord(
        record_id=composite.composite_id,
        document_id=composite.document_id,
        document_name=document_name,
        content=composite.composite_text,
        source_kind=composite.source_kind,
        content_kind=composite.content_kind,
        source_url=composite.source_url,
        parent_url=composite.parent_url,
        depth=composite.depth,
        section="composite_summary",
        lineage={
            **composite.metadata.get("lineage", {}),
            "chunk_ids": composite.chunk_ids,
            "summary_ids": composite.summary_ids,
        },
        metadata=composite.metadata,
    )


class ElasticsearchIndexer:
    def __init__(self, config: Optional[AppConfig] = None, client: Any = None):
        self.config = config or load_config()
        self.url = self.config.elasticsearch_url
        self.index_name = self.config.elasticsearch_index
        username = self.config.elasticsearch_username
        if isinstance(username, SecretStr):
            username = username.get_secret_value()
        self.username = username
        self.password = (
            self.config.elasticsearch_password.get_secret_value()
            if self.config.elasticsearch_password
            else None
        )
        if client is None:
            basic_auth = (
                (self.username, self.password) if self.username and self.password else None
            )
            client = Elasticsearch(
                self.url,
                basic_auth=basic_auth,
                verify_certs=True,
                request_timeout=30.0,
            )
        self.client = client

    def ping(self) -> bool:
        try:
            if self.client.ping():
                return True
            self.client.info()
            return True
        except Exception:
            return False

    def check_connection(self) -> None:
        if not self.ping():
            masked_url = self.url
            if "@" in self.url:
                masked_url = f"http://***:***@{self.url.split('@')[-1]}"
            raise ConnectionError(f"Elasticsearch server is unreachable at {masked_url}")

    def index_exists(self) -> bool:
        try:
            return bool(self.client.indices.exists(index=self.index_name))
        except Exception:
            return False

    def _expected_mapping(self) -> dict[str, Any]:
        keyword = {"type": "keyword"}
        return {
            "mappings": {
                "properties": {
                    "chunk_id": keyword,
                    "record_id": keyword,
                    "document_id": keyword,
                    "document_name": {"type": "text", "fields": {"keyword": keyword}},
                    "source_kind": keyword,
                    "content_kind": keyword,
                    "source_url": keyword,
                    "parent_url": keyword,
                    "sub_link": keyword,
                    "allowed_domain": keyword,
                    "depth": {"type": "integer"},
                    "section": {"type": "text", "fields": {"keyword": keyword}},
                    "content": {"type": "text"},
                    "page_numbers": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "chunk_type": keyword,
                    "lineage": {"type": "object", "enabled": False},
                    "figure_refs": keyword,
                    "table_refs": keyword,
                    "equation_refs": keyword,
                    "figure_captions": {"type": "text"},
                    "table_captions": {"type": "text"},
                    "figure_pages": {"type": "object", "enabled": False},
                    "table_pages": {"type": "object", "enabled": False},
                    "image_path": keyword,
                    "embedding_model": keyword,
                    "embedding_dimension": {"type": "integer"},
                    "schema_version": keyword,
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.config.embedding_dimension,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        }

    def create_index_if_not_exists(self) -> bool:
        if self.index_exists():
            mapping = self.client.indices.get_mapping(index=self.index_name)
            if isinstance(mapping, dict) and self.index_name in mapping:
                props = mapping[self.index_name]["mappings"].get("properties", {})
                dimension = props.get("embedding", {}).get("dims")
                if dimension and dimension != self.config.embedding_dimension:
                    raise ValueError("Existing index embedding dimension is incompatible")
            return False
        self.client.indices.create(index=self.index_name, body=self._expected_mapping())
        return True

    def index_records(
        self, records: Iterable[IndexableRecord], embeddings: Iterable[list[float]]
    ) -> dict[str, Any]:
        records, embeddings = list(records), list(embeddings)
        if len(records) != len(embeddings):
            raise ValueError("records and embeddings length mismatch")
        started = time.monotonic()
        actions = []
        for record, embedding in zip(records, embeddings):
            if len(embedding) != self.config.embedding_dimension:
                raise ValueError(
                    f"Invalid embedding dimension for {record.record_id}: "
                    f"{len(embedding)} does not match configured dimension "
                    f"{self.config.embedding_dimension}."
                )
            source = record.model_dump()
            source.update(
                {
                    "chunk_id": record.record_id,
                    "record_id": record.record_id,
                    "allowed_domain": record.allowed_domain
                    or (
                        self.config.crawl_allowed_domain
                        if record.source_kind in {"html", "crawled_pdf"}
                        else None
                    ),
                    "embedding_model": self.config.embedding_model,
                    "embedding_dimension": self.config.embedding_dimension,
                    "schema_version": "v3",
                    "embedding": embedding,
                }
            )
            for key, value in record.metadata.items():
                source.setdefault(key, value)
            actions.append(
                {"_index": self.index_name, "_id": record.record_id, "_source": source}
            )
        success, errors = bulk(self.client, actions, raise_on_error=False)
        failed = len(errors) if isinstance(errors, list) else (errors or 0)
        return {
            "total_received": len(records),
            "successfully_indexed": success,
            "failed": failed,
            "duration_seconds": round(time.monotonic() - started, 3),
        }

    def index_chunks(
        self, chunks: list[ChunkModel], embeddings: list[list[float]]
    ) -> dict[str, Any]:
        if not chunks:
            return {
                "total_received": 0,
                "successfully_indexed": 0,
                "failed": 0,
                "duration_seconds": 0.0,
            }
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})."
            )
        return self.index_records(
            [chunk_to_indexable(chunk) for chunk in chunks], embeddings
        )

    def delete_document_chunks(self, document_id: str) -> int:
        response = self.client.delete_by_query(
            index=self.index_name,
            body={"query": {"term": {"document_id": document_id}}},
            refresh=True,
        )
        return int(response.get("deleted", 0))
