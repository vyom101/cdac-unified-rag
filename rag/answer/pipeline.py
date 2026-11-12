"""Unified retrieval, context construction, grounding, and answer generation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from rag.answer.grounding import ABSTAIN_ANSWER, grounding_failure
from rag.config import AppConfig, load_config
from rag.retrieve import HybridSearchResult, SearchResult
from rag.retrieve.factory import get_retriever

RAG_SYSTEM_PROMPT = """You are a technical assistant for a PDF knowledge base.

Answer the user's question using ONLY the retrieved PDF context provided in the user message.

Rules:
1. Use only facts explicitly stated in the retrieved context.
2. Do not invent information or rely on outside knowledge.
3. Preserve technical terminology from the source documents.
4. If the retrieved context does not contain enough information to answer the question, respond with exactly:
   "The information is not available in the provided context."
5. When the context supports an answer, reference document names and page numbers where possible.
6. For HTML chunks, cite the source URL and crawl depth.
7. For PDF chunks, cite the document name, page numbers, and any figure or table references.
8. Knowledge-graph evidence is labeled separately; keep its node and edge provenance when you use it.
The same insufficient-context sentence applies to HTML, PDF, and knowledge-graph evidence.
"""


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    section: Optional[str] = None
    page_numbers: List[int] = Field(default_factory=list)
    content: str
    score: float
    retrieval_method: str
    source_kind: str = "uploaded_pdf"
    content_kind: str = "chunk"
    source_url: Optional[str] = None
    depth: int = 0
    figure_refs: List[str] = Field(default_factory=list)
    table_refs: List[str] = Field(default_factory=list)
    figure_pages: Dict[str, List[int]] = Field(default_factory=dict)
    table_pages: Dict[str, List[int]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    lineage: Dict[str, Any] = Field(default_factory=dict)


class SourceReference(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    source_kind: str = "uploaded_pdf"
    content_kind: str = "chunk"
    source_url: Optional[str] = None
    depth: int = 0
    section: Optional[str] = None
    page_numbers: List[int] = Field(default_factory=list)
    score: float
    retrieval_method: str
    figure_refs: List[str] = Field(default_factory=list)
    table_refs: List[str] = Field(default_factory=list)
    figure_pages: Dict[str, List[int]] = Field(default_factory=dict)
    table_pages: Dict[str, List[int]] = Field(default_factory=dict)
    lineage: Dict[str, Any] = Field(default_factory=dict)
    kg_node_provenance: List[Dict[str, Any]] = Field(default_factory=list)
    kg_edge_provenance: List[Dict[str, Any]] = Field(default_factory=list)


class RAGAnswer(BaseModel):
    query: str
    answer: str
    abstained: bool = False
    abstain_reason: Optional[str] = None
    sources: List[SourceReference] = Field(default_factory=list)
    retrieval_mode: str
    sources_filter: str = "all"
    chunks_retrieved: int
    include_kg: bool = False


def _score(result: Union[SearchResult, HybridSearchResult, RetrievedChunk]) -> float:
    return result.final_score if isinstance(result, HybridSearchResult) else result.score


def normalize_retrieval_results(
    results: List[Union[SearchResult, HybridSearchResult, RetrievedChunk]],
) -> List[RetrievedChunk]:
    normalized: List[RetrievedChunk] = []
    for result in results:
        if isinstance(result, RetrievedChunk):
            normalized.append(result)
            continue
        normalized.append(
            RetrievedChunk(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                document_name=result.document_name,
                section=result.section,
                page_numbers=result.page_numbers,
                content=result.content,
                score=_score(result),
                retrieval_method=result.retrieval_method,
                source_kind=getattr(result, "source_kind", "uploaded_pdf"),
                content_kind=getattr(result, "content_kind", "chunk"),
                source_url=getattr(result, "source_url", None),
                depth=getattr(result, "depth", 0),
                figure_refs=getattr(result, "figure_refs", []),
                table_refs=getattr(result, "table_refs", []),
                figure_pages=getattr(result, "figure_pages", {}),
                table_pages=getattr(result, "table_pages", {}),
                metadata=getattr(result, "metadata", {}),
                lineage=getattr(result, "lineage", {}),
            )
        )
    return normalized


def build_source_references(
    chunks: List[Union[RetrievedChunk, SearchResult]],
) -> List[SourceReference]:
    references = []
    for item in chunks:
        metadata = getattr(item, "metadata", {}) or {}
        references.append(
            SourceReference(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                document_name=item.document_name,
                source_kind=getattr(item, "source_kind", "uploaded_pdf"),
                content_kind=getattr(item, "content_kind", "chunk"),
                source_url=getattr(item, "source_url", None),
                depth=getattr(item, "depth", 0),
                section=item.section,
                page_numbers=item.page_numbers,
                score=_score(item),
                retrieval_method=item.retrieval_method,
                figure_refs=getattr(item, "figure_refs", []),
                table_refs=getattr(item, "table_refs", []),
                figure_pages=getattr(item, "figure_pages", {}),
                table_pages=getattr(item, "table_pages", {}),
                lineage=getattr(item, "lineage", {}),
                kg_node_provenance=metadata.get("node_provenance", []),
                kg_edge_provenance=metadata.get("edge_provenance", []),
            )
        )
    return references


def _format_page_numbers(page_numbers: List[int]) -> str:
    if not page_numbers:
        return "unknown"
    return ", ".join(str(page) for page in page_numbers)


def build_rag_context(
    chunks: List[Union[RetrievedChunk, SearchResult]],
) -> str:
    if not chunks:
        return "No relevant context was retrieved from the knowledge base."

    blocks: List[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source_kind = getattr(chunk, "source_kind", None)
        method = getattr(chunk, "retrieval_method", "")
        metadata = getattr(chunk, "metadata", {}) or {}
        lines = [f"[Source {index}]", f"Source kind: {source_kind or 'unknown'}"]
        if method == "knowledge_graph":
            provenance = getattr(chunk, "source_url", None) or metadata.get(
                "provenance", "unknown"
            )
            lines.extend(
                [
                    "Type: Knowledge Graph",
                    "Knowledge-graph evidence",
                    f"Provenance: {provenance}",
                ]
            )
            if metadata.get("node_provenance"):
                lines.append(f"Node provenance: {metadata['node_provenance']}")
            if metadata.get("edge_provenance"):
                lines.append(f"Edge provenance: {metadata['edge_provenance']}")
        elif source_kind == "html":
            lines.extend(
                [
                    f"URL: {getattr(chunk, 'source_url', None) or 'unknown'}",
                    f"Depth: {getattr(chunk, 'depth', 0)}",
                ]
            )
        else:
            lines.extend(
                [
                    f"Document: {chunk.document_name}",
                    f"Pages: {_format_page_numbers(chunk.page_numbers)}",
                ]
            )
            figure_refs = getattr(chunk, "figure_refs", [])
            table_refs = getattr(chunk, "table_refs", [])
            figure_pages = getattr(chunk, "figure_pages", {})
            table_pages = getattr(chunk, "table_pages", {})
            if figure_refs:
                lines.append(f"Figures: {', '.join(figure_refs)}")
            if table_refs:
                lines.append(f"Tables: {', '.join(table_refs)}")
            if figure_pages:
                lines.append(f"Figure pages: {figure_pages}")
            if table_pages:
                lines.append(f"Table pages: {table_pages}")
        lines.extend(
            [
                f"Section: {chunk.section or 'N/A'}",
                f"Chunk ID: {chunk.chunk_id}",
                f"Content:\n{chunk.content.strip()}",
            ]
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_rag_user_prompt(query: str, context: str) -> str:
    return (
        "Retrieved PDF Context:\n"
        f"{context}\n\n"
        "User Question:\n"
        f"{query.strip()}\n\n"
        "Answer the question using only the retrieved PDF context above."
    )


def build_insufficient_context_answer() -> str:
    return ABSTAIN_ANSWER


def _as_search_results(
    chunks: List[Union[SearchResult, HybridSearchResult, RetrievedChunk]],
) -> List[SearchResult]:
    results: List[SearchResult] = []
    for chunk in chunks:
        if isinstance(chunk, SearchResult):
            results.append(chunk)
            continue
        results.append(
            SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                source_kind=getattr(chunk, "source_kind", "uploaded_pdf"),
                content=chunk.content,
                score=_score(chunk),
                retrieval_method=chunk.retrieval_method,
                page_numbers=chunk.page_numbers,
                section=chunk.section,
                source_url=getattr(chunk, "source_url", None),
                figure_refs=getattr(chunk, "figure_refs", []),
                table_refs=getattr(chunk, "table_refs", []),
                figure_pages=getattr(chunk, "figure_pages", {}),
                table_pages=getattr(chunk, "table_pages", {}),
                metadata=getattr(chunk, "metadata", {}),
            )
        )
    return results


def answer_question(
    query: str,
    config: Optional[AppConfig] = None,
    *,
    retriever: Any = None,
    llm_client: Any = None,
    llm: Any = None,
    kg: Any = None,
    mode: Optional[str] = None,
    sources: Optional[str] = None,
    source_filter: Optional[str] = None,
    include_kg: bool = False,
    top_k: Optional[int] = None,
    domain: Optional[str] = None,
) -> RAGAnswer:
    config = config or load_config()
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query text cannot be empty or whitespace-only.")
    selected_mode = mode or config.retrieval_mode
    source_filter = source_filter or sources or config.kb_source
    retriever = retriever or get_retriever(config, mode=selected_mode)
    kwargs: dict[str, Any] = {
        "top_k": top_k or config.top_k,
        "source": source_filter,
        "domain": domain,
    }
    if selected_mode in {"vector", "hybrid"}:
        kwargs["num_candidates"] = config.num_candidates
    corpus_chunks = list(retriever.search(query, **kwargs))
    failure = grounding_failure(query, _as_search_results(corpus_chunks), config)
    if failure:
        return RAGAnswer(
            query=query.strip(),
            answer=ABSTAIN_ANSWER,
            abstained=True,
            abstain_reason=failure,
            sources=build_source_references(corpus_chunks),
            retrieval_mode=selected_mode,
            sources_filter=source_filter,
            chunks_retrieved=len(corpus_chunks),
            include_kg=include_kg,
        )

    chunks: List[Any] = list(corpus_chunks)
    if include_kg:
        if kg is None:
            from rag.graph.store import KnowledgeGraphStore

            kg = KnowledgeGraphStore(config.kg_store_path)
        chunks.extend(kg.search(query, top_k=top_k or config.top_k))

    llm_client = llm_client or llm
    if llm_client is None:
        from rag.answer.llm_client import get_llm_client

        llm_client = get_llm_client(config)
    context = build_rag_context(chunks)
    prompt = (
        "Retrieved Unified Context:\n"
        + context
        + "\n\nUser Question:\n"
        + query.strip()
        + "\n\nAnswer only from the context and cite its provenance."
    )
    answer = llm_client.generate_completion(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.0,
    )
    return RAGAnswer(
        query=query.strip(),
        answer=answer.strip(),
        abstained=False,
        sources=build_source_references(chunks),
        retrieval_mode=selected_mode,
        sources_filter=source_filter,
        chunks_retrieved=len(chunks),
        include_kg=include_kg,
    )


class RAGPipeline:
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        llm_client: Any = None,
        retriever: Any = None,
        kg: Any = None,
    ):
        self.config = config or load_config()
        if llm_client is None:
            from rag.answer.llm_client import get_llm_client

            llm_client = get_llm_client(self.config)
        self.llm_client = llm_client
        self.retriever = retriever or get_retriever(self.config)
        self.kg = kg

    def _validate_query(self, query: str) -> None:
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query)}")
        if not query or query.strip() == "":
            raise ValueError("Query text cannot be empty or whitespace-only.")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = top_k if top_k is not None else self.config.top_k
        if k <= 0:
            raise ValueError(f"top_k must be a positive integer (got {k}).")
        mode = self.config.retrieval_mode.lower()
        if mode in {"hybrid", "vector"}:
            raw_results = self.retriever.search(
                query,
                top_k=k,
                num_candidates=self.config.num_candidates,
            )
        else:
            raw_results = self.retriever.search(query, top_k=k)
        return normalize_retrieval_results(raw_results)

    def answer_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        *,
        sources: Optional[str] = None,
        include_kg: bool = False,
        domain: Optional[str] = None,
    ) -> RAGAnswer:
        self._validate_query(query)
        if sources is not None or include_kg or domain is not None:
            return answer_question(
                query,
                self.config,
                retriever=self.retriever,
                llm_client=self.llm_client,
                kg=self.kg,
                sources=sources,
                include_kg=include_kg,
                top_k=top_k,
                domain=domain,
            )
        chunks = self.retrieve(query, top_k=top_k)
        sources_out = build_source_references(chunks)
        if not chunks:
            return RAGAnswer(
                query=query.strip(),
                answer=build_insufficient_context_answer(),
                abstained=True,
                abstain_reason="no_retrieval_hits",
                sources=[],
                retrieval_mode=self.config.retrieval_mode,
                sources_filter=self.config.kb_source,
                chunks_retrieved=0,
            )
        context = build_rag_context(chunks)
        user_prompt = build_rag_user_prompt(query, context)
        answer = self.llm_client.generate_completion(
            system_prompt=RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        return RAGAnswer(
            query=query.strip(),
            answer=answer.strip(),
            sources=sources_out,
            retrieval_mode=self.config.retrieval_mode,
            sources_filter=self.config.kb_source,
            chunks_retrieved=len(chunks),
        )
