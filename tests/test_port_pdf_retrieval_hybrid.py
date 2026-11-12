import pytest
from unittest.mock import MagicMock

from rag.config import load_config
from rag.retrieve import HybridSearchResult, SearchResult
from rag.retrieve.hybrid import (
    HybridRetriever,
    compute_rrf_scores,
    compute_weighted_min_max_scores,
    fuse_hybrid_results,
    merge_search_results,
    min_max_normalize,
    validate_fusion_weights,
)


def _make_result(
    chunk_id: str,
    score: float,
    method: str,
    *,
    content: str = "sample content",
    section: str = "Section A",
    page_numbers=None,
) -> SearchResult:
    if page_numbers is None:
        page_numbers = [1]
    return SearchResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        document_name=f"{chunk_id}.pdf",
        section=section,
        page_numbers=page_numbers,
        content=content,
        score=score,
        retrieval_method=method,
    )


def test_min_max_normalization():
    scores = {"a": 2.0, "b": 4.0, "c": 6.0}
    normalized = min_max_normalize(scores)
    assert normalized["a"] == pytest.approx(0.0)
    assert normalized["b"] == pytest.approx(0.5)
    assert normalized["c"] == pytest.approx(1.0)


def test_min_max_equal_scores():
    scores = {"a": 5.0, "b": 5.0, "c": 5.0}
    normalized = min_max_normalize(scores)
    assert normalized == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_min_max_empty_scores():
    assert min_max_normalize({}) == {}


def test_weight_calculation():
    bm25_scores = {"shared": 10.0, "bm25_only": 5.0}
    vector_scores = {"shared": 0.2, "vector_only": 0.8}
    fused = compute_weighted_min_max_scores(bm25_scores, vector_scores, 0.3, 0.7)

    assert fused["shared"][0] == pytest.approx(1.0)
    assert fused["shared"][1] == pytest.approx(0.0)
    assert fused["shared"][2] == pytest.approx(0.3)

    assert fused["bm25_only"][0] == pytest.approx(0.0)
    assert fused["bm25_only"][1] == pytest.approx(0.0)
    assert fused["bm25_only"][2] == pytest.approx(0.0)

    assert fused["vector_only"][0] == pytest.approx(0.0)
    assert fused["vector_only"][1] == pytest.approx(1.0)
    assert fused["vector_only"][2] == pytest.approx(0.7)


def test_invalid_weights_rejected():
    with pytest.raises(ValueError) as excinfo:
        validate_fusion_weights(-0.1, 1.1)
    assert "non-negative" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        validate_fusion_weights(0.4, 0.7)
    assert "must equal 1.0" in str(excinfo.value)


def test_merge_same_chunk_from_both_retrievers():
    bm25_results = [_make_result("chunk_001", 12.0, "bm25")]
    vector_results = [_make_result("chunk_001", 0.91, "vector")]

    merged = merge_search_results(bm25_results, vector_results)
    assert len(merged) == 1
    assert merged["chunk_001"].bm25_score == 12.0
    assert merged["chunk_001"].vector_score == 0.91
    assert merged["chunk_001"].document_name == "chunk_001.pdf"


def test_fuse_both_retrievers_return_same_chunk():
    bm25_results = [_make_result("chunk_001", 10.0, "bm25", content="shared chunk")]
    vector_results = [_make_result("chunk_001", 0.5, "vector", content="shared chunk")]

    results = fuse_hybrid_results(
        bm25_results,
        vector_results,
        keyword_weight=0.3,
        semantic_weight=0.7,
        top_k=5,
    )

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, HybridSearchResult)
    assert result.chunk_id == "chunk_001"
    assert result.bm25_score == 10.0
    assert result.vector_score == 0.5
    assert result.normalized_keyword_score == 1.0
    assert result.normalized_semantic_score == 1.0
    assert result.final_score == pytest.approx(1.0)
    assert result.score == result.final_score
    assert result.retrieval_method == "hybrid"
    assert result.fusion_strategy == "min_max"


def test_fuse_bm25_only_chunk():
    bm25_results = [_make_result("bm25_only", 8.0, "bm25")]
    vector_results = []

    results = fuse_hybrid_results(bm25_results, vector_results, top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "bm25_only"
    assert results[0].bm25_score == 8.0
    assert results[0].vector_score is None
    assert results[0].normalized_keyword_score == 1.0
    assert results[0].normalized_semantic_score == 0.0
    assert results[0].final_score == pytest.approx(0.3)


def test_fuse_vector_only_chunk():
    bm25_results = []
    vector_results = [_make_result("vector_only", 0.75, "vector")]

    results = fuse_hybrid_results(bm25_results, vector_results, top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == "vector_only"
    assert results[0].bm25_score is None
    assert results[0].vector_score == 0.75
    assert results[0].normalized_keyword_score == 0.0
    assert results[0].normalized_semantic_score == 1.0
    assert results[0].final_score == pytest.approx(0.7)


def test_fuse_empty_bm25():
    bm25_results = []
    vector_results = [
        _make_result("vector_a", 0.9, "vector"),
        _make_result("vector_b", 0.5, "vector"),
    ]

    results = fuse_hybrid_results(bm25_results, vector_results, top_k=5)
    assert len(results) == 2
    assert results[0].chunk_id == "vector_a"
    assert results[0].final_score == pytest.approx(0.7)


def test_fuse_empty_vector():
    bm25_results = [
        _make_result("bm25_a", 12.0, "bm25"),
        _make_result("bm25_b", 6.0, "bm25"),
    ]
    vector_results = []

    results = fuse_hybrid_results(bm25_results, vector_results, top_k=5)
    assert len(results) == 2
    assert results[0].chunk_id == "bm25_a"
    assert results[0].final_score == pytest.approx(0.3)


def test_fuse_both_empty():
    assert fuse_hybrid_results([], [], top_k=5) == []


def test_rrf_fusion():
    bm25_results = [
        _make_result("chunk_a", 10.0, "bm25"),
        _make_result("chunk_b", 8.0, "bm25"),
    ]
    vector_results = [
        _make_result("chunk_b", 0.9, "vector"),
        _make_result("chunk_c", 0.8, "vector"),
    ]

    results = fuse_hybrid_results(
        bm25_results,
        vector_results,
        fusion_strategy="rrf",
        rrf_k=60,
        top_k=5,
    )

    expected = compute_rrf_scores(
        ["chunk_a", "chunk_b"],
        ["chunk_b", "chunk_c"],
        60,
    )
    assert len(results) == 3
    assert results[0].chunk_id == "chunk_b"
    assert results[0].final_score == pytest.approx(expected["chunk_b"])
    assert results[0].fusion_strategy == "rrf"
    assert results[0].normalized_keyword_score == 0.0
    assert results[0].normalized_semantic_score == 0.0


def test_rrf_invalid_k():
    with pytest.raises(ValueError) as excinfo:
        compute_rrf_scores(["a"], ["b"], 0)
    assert "RRF_K must be a positive integer" in str(excinfo.value)


def test_top_k_limit():
    bm25_results = [
        _make_result("chunk_a", 10.0, "bm25"),
        _make_result("chunk_b", 8.0, "bm25"),
        _make_result("chunk_c", 6.0, "bm25"),
    ]
    vector_results = [
        _make_result("chunk_a", 0.9, "vector"),
        _make_result("chunk_b", 0.8, "vector"),
        _make_result("chunk_c", 0.7, "vector"),
    ]

    results = fuse_hybrid_results(bm25_results, vector_results, top_k=2)
    assert len(results) == 2


def test_duplicate_merging_preserves_metadata():
    bm25_results = [
        _make_result(
            "dup_chunk",
            10.0,
            "bm25",
            content="bm25 content",
            section="Intro",
            page_numbers=[1, 2],
        )
    ]
    vector_results = [
        _make_result(
            "dup_chunk",
            0.95,
            "vector",
            content="vector content",
            section="Other",
            page_numbers=[99],
        )
    ]

    results = fuse_hybrid_results(bm25_results, vector_results, top_k=1)
    assert len(results) == 1
    assert results[0].content == "bm25 content"
    assert results[0].section == "Intro"
    assert results[0].page_numbers == [1, 2]
    assert results[0].document_id == "doc_dup_chunk"


def test_deterministic_ranking_on_tie():
    bm25_results = [
        _make_result("chunk_z", 5.0, "bm25"),
        _make_result("chunk_a", 5.0, "bm25"),
    ]
    vector_results = [
        _make_result("chunk_z", 0.5, "vector"),
        _make_result("chunk_a", 0.5, "vector"),
    ]

    first = fuse_hybrid_results(bm25_results, vector_results, top_k=5)
    second = fuse_hybrid_results(bm25_results, vector_results, top_k=5)
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert [item.chunk_id for item in first] == ["chunk_a", "chunk_z"]


def test_metadata_preservation():
    bm25_results = [_make_result("meta_chunk", 9.0, "bm25", section="Methods", page_numbers=[3, 4])]
    vector_results = [_make_result("meta_chunk", 0.88, "vector", section="Methods", page_numbers=[3, 4])]

    result = fuse_hybrid_results(bm25_results, vector_results, top_k=1)[0]
    assert result.document_id == "doc_meta_chunk"
    assert result.document_name == "meta_chunk.pdf"
    assert result.section == "Methods"
    assert result.page_numbers == [3, 4]
    assert result.content == "sample content"
    assert result.bm25_score == 9.0
    assert result.vector_score == 0.88


def test_hybrid_retriever_search_with_mocked_dependencies():
    config = load_config()
    mock_bm25 = MagicMock()
    mock_vector = MagicMock()

    mock_bm25.search.return_value = [_make_result("chunk_a", 10.0, "bm25")]
    mock_vector.search.return_value = [_make_result("chunk_b", 0.9, "vector")]

    retriever = HybridRetriever(
        config=config,
        bm25_retriever=mock_bm25,
        vector_retriever=mock_vector,
    )

    results = retriever.search("hybrid query", top_k=3, num_candidates=20)
    assert len(results) == 2
    assert {item.chunk_id for item in results} == {"chunk_a", "chunk_b"}

    # Hybrid fuses over a larger candidate pool (max(top_k, hybrid_candidate_pool))
    # before truncating to top_k, so sub-retrievers are queried with the pool size.
    expected_pool = max(3, config.hybrid_candidate_pool)
    expected_pool_candidates = max(20, expected_pool)
    mock_bm25.search.assert_called_once_with("hybrid query", top_k=expected_pool)
    mock_vector.search.assert_called_once_with(
        "hybrid query", top_k=expected_pool, num_candidates=expected_pool_candidates
    )


def test_hybrid_retriever_input_validation():
    retriever = HybridRetriever(
        bm25_retriever=MagicMock(),
        vector_retriever=MagicMock(),
    )

    with pytest.raises(ValueError) as excinfo:
        retriever.search("   ")
    assert "cannot be empty or whitespace-only" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        retriever.search("valid query", top_k=0)
    assert "top_k must be a positive integer" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        retriever.search("valid query", top_k=10, num_candidates=5)
    assert "num_candidates" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        retriever.search("valid query", keyword_weight=0.2, semantic_weight=0.2)
    assert "must equal 1.0" in str(excinfo.value)


def test_hybrid_retriever_both_empty_from_dependencies():
    mock_bm25 = MagicMock()
    mock_vector = MagicMock()
    mock_bm25.search.return_value = []
    mock_vector.search.return_value = []

    retriever = HybridRetriever(
        bm25_retriever=mock_bm25,
        vector_retriever=mock_vector,
    )
    assert retriever.search("empty query") == []

