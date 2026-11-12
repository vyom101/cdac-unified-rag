import pytest
from unittest.mock import MagicMock, patch

from rag.config import AppConfig, load_config
from rag.index.embedder import BaseEmbeddingProvider
from rag.retrieve.vector import VectorRetriever
from rag.retrieve import SearchResult


# ==============================================================================
# UNIT TESTS FOR VECTOR RETRIEVER (100% Offline)
# ==============================================================================

@patch("rag.retrieve.vector.Elasticsearch")
def test_vector_retriever_initialization_and_provider_injection(mock_es_class):
    """
    Verifies that the VectorRetriever initializes correctly and holds the injected EmbeddingProvider.
    """
    config = load_config()
    config.elasticsearch_index = "cdac_vector_index"
    
    mock_provider = MagicMock(spec=BaseEmbeddingProvider)
    
    retriever = VectorRetriever(config=config, embedding_provider=mock_provider)
    
    assert retriever.index_name == "cdac_vector_index"
    assert retriever.embedding_provider == mock_provider


@patch("rag.retrieve.vector.Elasticsearch")
def test_vector_query_construction_and_parsing(mock_es_class):
    """
    Verifies HNSW kNN query payload structure, embedding generation call, and result scoring parsing.
    """
    mock_client = MagicMock()
    mock_client.indices.exists.return_value = True

    # Configure mock search response with HNSW vector scores
    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_score": 0.897,
                    "_source": {
                        "chunk_id": "doc_1_chunk_004",
                        "document_id": "doc_1",
                        "document_name": "ML_CDAC.pdf",
                        "section": "HNSW Algorithm",
                        "page_numbers": [4, 5],
                        "content": "Graph-based HNSW vector proximity search details."
                    }
                }
            ]
        }
    }
    mock_es_class.return_value = mock_client

    # Mock embedding provider
    mock_provider = MagicMock(spec=BaseEmbeddingProvider)
    query_vector = [0.1] * 256
    mock_provider.embed_text.return_value = query_vector

    config = load_config()
    config.elasticsearch_index = "cdac_knn_index"
    
    retriever = VectorRetriever(
        config=config,
        embedding_provider=mock_provider,
        client=mock_client
    )

    # Execute search
    results = retriever.search("approximate nearest neighbors", top_k=5, num_candidates=50)

    assert len(results) == 1
    res = results[0]
    assert isinstance(res, SearchResult)
    assert res.chunk_id == "doc_1_chunk_004"
    assert res.score == 0.897
    assert res.retrieval_method == "vector"
    assert res.page_numbers == [4, 5]

    # Check that the injected embedding provider was called to embed the query
    mock_provider.embed_text.assert_called_once_with("approximate nearest neighbors")

    # Verify Elasticsearch search arguments (matches ES 8.x knn structure)
    mock_client.search.assert_called_once()
    call_args = mock_client.search.call_args
    assert call_args[1]["index"] == "cdac_knn_index"
    
    body = call_args[1]["body"]
    assert body["size"] == 5
    
    knn = body["knn"]
    assert knn["field"] == "embedding"
    assert knn["query_vector"] == query_vector
    assert knn["k"] == 5
    assert knn["num_candidates"] == 50


@patch("rag.retrieve.vector.Elasticsearch")
def test_num_candidates_constraint_validation(mock_es_class):
    """
    Enforces the validation constraint: num_candidates must be >= top_k.
    """
    mock_provider = MagicMock(spec=BaseEmbeddingProvider)
    retriever = VectorRetriever(embedding_provider=mock_provider)

    # Invalid: num_candidates (5) is less than top_k (10)
    with pytest.raises(ValueError) as excinfo:
        retriever.search("test query", top_k=10, num_candidates=5)
    assert "num_candidates" in str(excinfo.value)
    assert "must be greater than or equal to top_k" in str(excinfo.value)


@patch("rag.retrieve.vector.Elasticsearch")
def test_vector_input_validation(mock_es_class):
    """
    Verifies query validation for empty and invalid query parameters.
    """
    mock_provider = MagicMock(spec=BaseEmbeddingProvider)
    retriever = VectorRetriever(embedding_provider=mock_provider)

    # Empty query
    with pytest.raises(ValueError) as excinfo:
        retriever.search("   ")
    assert "cannot be empty or whitespace-only" in str(excinfo.value)

    # Invalid top_k
    with pytest.raises(ValueError) as excinfo:
        retriever.search("valid text", top_k=-5)
    assert "top_k must be a positive integer" in str(excinfo.value)

    # Invalid num_candidates
    with pytest.raises(ValueError) as excinfo:
        retriever.search("valid text", top_k=5, num_candidates=-10)
    assert "num_candidates" in str(excinfo.value)

