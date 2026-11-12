import pytest
from unittest.mock import MagicMock, patch

from rag.config import AppConfig, load_config
from rag.answer.llm_client import MockLLMClient
from rag.answer.pipeline import (
    RAGPipeline,
    RAG_SYSTEM_PROMPT,
    RetrievedChunk,
    build_insufficient_context_answer,
    build_rag_context,
    build_rag_user_prompt,
    build_source_references,
    normalize_retrieval_results,
)
from rag.retrieve import HybridSearchResult, SearchResult
from rag.retrieve.factory import get_retriever


def _search_result(chunk_id: str, method: str, score: float = 1.0) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        document_name=f"{chunk_id}.pdf",
        section="Intro",
        page_numbers=[1, 2],
        content=f"Content for {chunk_id}",
        score=score,
        retrieval_method=method,
    )


def _hybrid_result(chunk_id: str, score: float = 0.9) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        document_name=f"{chunk_id}.pdf",
        section="Methods",
        page_numbers=[3],
        content=f"Hybrid content for {chunk_id}",
        bm25_score=5.0,
        vector_score=0.8,
        normalized_keyword_score=0.5,
        normalized_semantic_score=0.7,
        final_score=score,
        score=score,
        retrieval_method="hybrid",
        fusion_strategy="min_max",
    )


def test_normalize_search_and_hybrid_results():
    search = normalize_retrieval_results([_search_result("a", "bm25", 4.2)])
    hybrid = normalize_retrieval_results([_hybrid_result("b", 0.75)])

    assert search[0].score == 4.2
    assert search[0].retrieval_method == "bm25"
    assert hybrid[0].score == 0.75
    assert hybrid[0].retrieval_method == "hybrid"


def test_build_rag_context_preserves_metadata():
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_001",
            document_id="doc_1",
            document_name="ML4 vyom.pdf",
            section="Decision Trees",
            page_numbers=[2, 3],
            content="Decision trees split data using feature thresholds.",
            score=0.91,
            retrieval_method="hybrid",
        )
    ]

    context = build_rag_context(chunks)
    assert "ML4 vyom.pdf" in context
    assert "Decision Trees" in context
    assert "2, 3" in context
    assert "Decision trees split data" in context
    assert "[Source 1]" in context


def test_build_rag_context_empty():
    assert "No relevant context was retrieved" in build_rag_context([])


def test_build_rag_user_prompt_contains_query_and_context():
    prompt = build_rag_user_prompt("What is HNSW?", "context block")
    assert "What is HNSW?" in prompt
    assert "context block" in prompt
    assert "Retrieved PDF Context" in prompt


def test_build_source_references():
    chunks = normalize_retrieval_results([_search_result("chunk_a", "vector", 0.88)])
    sources = build_source_references(chunks)
    assert len(sources) == 1
    assert sources[0].document_name == "chunk_a.pdf"
    assert sources[0].page_numbers == [1, 2]
    assert sources[0].retrieval_method == "vector"


def test_get_retriever_resolves_modes():
    bm25_config = AppConfig(retrieval_mode="bm25")
    vector_config = AppConfig(retrieval_mode="vector")
    hybrid_config = AppConfig(retrieval_mode="hybrid")

    with patch("rag.retrieve.factory.BM25Retriever") as mock_bm25:
        mock_bm25.return_value = MagicMock()
        get_retriever(bm25_config)
        mock_bm25.assert_called_once()

    with patch("rag.retrieve.factory.VectorRetriever") as mock_vector:
        mock_vector.return_value = MagicMock()
        get_retriever(vector_config)
        mock_vector.assert_called_once()

    with patch("rag.retrieve.factory.HybridRetriever") as mock_hybrid, patch(
        "rag.retrieve.factory.VectorRetriever"
    ) as mock_vector:
        mock_hybrid.return_value = MagicMock()
        mock_vector.return_value = MagicMock()
        get_retriever(hybrid_config)
        mock_hybrid.assert_called_once()


def test_rag_pipeline_answer_with_mocked_retriever_and_llm():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [_hybrid_result("chunk_001", 0.95)]

    mock_llm = MagicMock()
    mock_llm.generate_completion.return_value = (
        "Decision trees split data using feature thresholds (ML4 vyom.pdf, pages 3)."
    )

    config = load_config()
    config.retrieval_mode = "hybrid"
    pipeline = RAGPipeline(config=config, llm_client=mock_llm, retriever=mock_retriever)

    result = pipeline.answer_query("How do decision trees work?")

    assert result.query == "How do decision trees work?"
    assert "Decision trees split data" in result.answer
    assert result.retrieval_mode == "hybrid"
    assert result.chunks_retrieved == 1
    assert len(result.sources) == 1
    assert result.sources[0].document_name == "chunk_001.pdf"
    assert result.sources[0].section == "Methods"
    assert result.sources[0].page_numbers == [3]

    mock_retriever.search.assert_called_once()
    mock_llm.generate_completion.assert_called_once()
    system_prompt, user_prompt = (
        mock_llm.generate_completion.call_args.kwargs["system_prompt"],
        mock_llm.generate_completion.call_args.kwargs["user_prompt"],
    )
    assert system_prompt == RAG_SYSTEM_PROMPT
    assert "Hybrid content for chunk_001" in user_prompt
    assert "Do not invent information" not in user_prompt
    assert "only facts explicitly stated" in system_prompt.lower() or "ONLY" in system_prompt


def test_rag_pipeline_empty_retrieval_skips_llm():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    mock_llm = MagicMock()

    pipeline = RAGPipeline(llm_client=mock_llm, retriever=mock_retriever)
    result = pipeline.answer_query("Unknown topic?")

    assert result.answer == build_insufficient_context_answer()
    assert result.sources == []
    assert result.chunks_retrieved == 0
    mock_llm.generate_completion.assert_not_called()


def test_rag_pipeline_uses_configured_top_k_for_hybrid():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    config = load_config()
    config.retrieval_mode = "hybrid"
    config.top_k = 7
    config.num_candidates = 40

    pipeline = RAGPipeline(config=config, llm_client=MagicMock(), retriever=mock_retriever)
    pipeline.answer_query("test query")

    mock_retriever.search.assert_called_once_with(
        "test query",
        top_k=7,
        num_candidates=40,
    )


def test_rag_pipeline_bm25_mode_calls_bm25_search_signature():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [_search_result("bm25_chunk", "bm25")]
    mock_llm = MagicMock()
    mock_llm.generate_completion.return_value = "BM25-grounded answer."

    config = load_config()
    config.retrieval_mode = "bm25"
    pipeline = RAGPipeline(
        config=config,
        llm_client=mock_llm,
        retriever=mock_retriever,
    )
    pipeline.answer_query("keyword query")

    mock_retriever.search.assert_called_once_with("keyword query", top_k=config.top_k)


def test_rag_pipeline_vector_mode_passes_num_candidates():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [_search_result("vector_chunk", "vector")]
    mock_llm = MagicMock()
    mock_llm.generate_completion.return_value = "Vector-grounded answer."

    config = load_config()
    config.retrieval_mode = "vector"
    config.num_candidates = 25
    pipeline = RAGPipeline(
        config=config,
        llm_client=mock_llm,
        retriever=mock_retriever,
    )
    pipeline.answer_query("semantic query")

    mock_retriever.search.assert_called_once_with(
        "semantic query",
        top_k=config.top_k,
        num_candidates=25,
    )


def test_rag_pipeline_input_validation():
    pipeline = RAGPipeline(llm_client=MagicMock(), retriever=MagicMock())

    with pytest.raises(ValueError) as excinfo:
        pipeline.answer_query("   ")
    assert "cannot be empty or whitespace-only" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        pipeline.retrieve("valid", top_k=0)
    assert "top_k must be a positive integer" in str(excinfo.value)


def test_rag_pipeline_with_mock_llm_client_integration():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        _search_result("chunk_x", "bm25", score=3.3),
    ]

    pipeline = RAGPipeline(
        llm_client=MockLLMClient(),
        retriever=mock_retriever,
    )
    result = pipeline.answer_query("Explain the document content.")

    assert result.chunks_retrieved == 1
    assert result.sources[0].chunk_id == "chunk_x"
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0


def test_rag_system_prompt_requires_context_only_answers():
    assert "ONLY" in RAG_SYSTEM_PROMPT
    assert "not available in the provided context" in RAG_SYSTEM_PROMPT
    assert "Do not invent information" in RAG_SYSTEM_PROMPT

