"""Retriever construction with mode and dependency overrides."""

from typing import Any, Optional

from rag.config import AppConfig
from rag.retrieve.bm25 import BM25Retriever
from rag.retrieve.hybrid import HybridRetriever
from rag.retrieve.vector import VectorRetriever


def get_retriever(
    config: AppConfig, mode: Optional[str] = None, *, client: Any = None,
    indexer: Any = None, embedder: Any = None, bm25_retriever: Any = None,
    vector_retriever: Any = None,
) -> Any:
    selected = (mode or config.retrieval_mode).lower()
    if selected == "bm25":
        return bm25_retriever or BM25Retriever(config, indexer=indexer, client=client)
    if selected == "vector":
        return vector_retriever or VectorRetriever(
            config, indexer=indexer, embedder=embedder, client=client
        )
    if selected == "hybrid":
        return HybridRetriever(
            config, bm25_retriever=bm25_retriever,
            vector_retriever=vector_retriever, indexer=indexer,
            embedder=embedder, client=client,
        )
    raise ValueError(f"Unsupported retrieval mode: {selected}")
