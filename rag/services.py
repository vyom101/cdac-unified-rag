"""Small injectable service functions for CLI, UI, and tests."""

from typing import Any, Optional

from rag.answer.pipeline import RAGAnswer, answer_question as _answer_question
from rag.config import AppConfig, load_config
from rag.retrieve.factory import get_retriever
from rag.retrieve.filters import resolve_domain


def search_corpus(query: str, config: Optional[AppConfig] = None, *,
                  retriever: Any = None, mode: Optional[str] = None,
                  sources: Optional[str] = None, domain: Optional[str] = None,
                  top_k: Optional[int] = None):
    config = config or load_config()
    retriever = retriever or get_retriever(config, mode=mode)
    kwargs = {
        "top_k": top_k or config.top_k,
        "source": sources or config.kb_source,
        "domain": domain if domain is not None else resolve_domain(config),
    }
    if (mode or config.retrieval_mode) in {"vector", "hybrid"}:
        kwargs["num_candidates"] = config.num_candidates
    return retriever.search(query, **kwargs)


def search_kg(query: str, config: Optional[AppConfig] = None, *,
              kg: Any = None, top_k: Optional[int] = None):
    config = config or load_config()
    if kg is None:
        from rag.graph.store import KnowledgeGraphStore
        kg = KnowledgeGraphStore(config.kg_store_path)
    return kg.search(query, top_k=top_k or config.top_k)


def answer_question(query: str, config: Optional[AppConfig] = None, **kwargs: Any) -> RAGAnswer:
    return _answer_question(query, config, **kwargs)
