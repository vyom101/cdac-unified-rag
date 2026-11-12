import pytest
from unittest.mock import MagicMock, patch

from rag.config import AppConfig, load_config
from rag.retrieve.bm25 import BM25Retriever
from rag.retrieve import SearchResult


# ==============================================================================
# UNIT TESTS FOR BM25 RETRIEVER (100% Offline)
# ==============================================================================

@patch("rag.retrieve.bm25.Elasticsearch")
def test_bm25_retriever_initialization(mock_es_class):
    """
    Checks that the BM25 retriever correctly connects to the index from config.
    """
    config = load_config()
    config.elasticsearch_url = "http://localhost:9200"
    config.elasticsearch_index = "cdac_bm25_index"

    retriever = BM25Retriever(config)
    assert retriever.index_name == "cdac_bm25_index"
    mock_es_class.assert_called_once_with(
        "http://localhost:9200",
        basic_auth=None,
        verify_certs=True,
        request_timeout=30.0
    )


@patch("rag.retrieve.bm25.Elasticsearch")
def test_bm25_query_construction_and_parsing(mock_es_class):
    """
    Verifies that the search method builds match content queries and parses raw results correctly.
    """
    mock_client = MagicMock()
    mock_client.indices.exists.return_value = True

    # Configure mock search hits response
    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_score": 12.34,
                    "_source": {
                        "chunk_id": "doc_1_chunk_001",
                        "document_id": "doc_1",
                        "document_name": "ML_CDAC.pdf",
                        "section": "Decision Trees",
                        "page_numbers": [2, 3],
                        "content": "BM25 keyword match content here."
                    }
                }
            ]
        }
    }
    mock_es_class.return_value = mock_client

    config = load_config()
    config.elasticsearch_index = "cdac_test_idx"
    retriever = BM25Retriever(config)

    # Run search
    results = retriever.search("decision trees", top_k=5)

    assert len(results) == 1
    res = results[0]
    assert isinstance(res, SearchResult)
    assert res.chunk_id == "doc_1_chunk_001"
    assert res.score == 12.34
    assert res.retrieval_method == "bm25"
    assert res.page_numbers == [2, 3]

    # Verify Elasticsearch search parameters
    mock_client.search.assert_called_once()
    call_args = mock_client.search.call_args
    assert call_args[1]["index"] == "cdac_test_idx"
    
    body = call_args[1]["body"]
    assert body["size"] == 5
    assert body["query"]["match"]["content"]["query"] == "decision trees"


@patch("rag.retrieve.bm25.Elasticsearch")
def test_bm25_input_validation(mock_es_class):
    """
    Verifies that empty, invalid, and out of bounds top_k inputs throw appropriate exceptions.
    """
    retriever = BM25Retriever()

    # Empty query
    with pytest.raises(ValueError) as excinfo:
        retriever.search("   ")
    assert "cannot be empty or whitespace-only" in str(excinfo.value)

    # Invalid top_k
    with pytest.raises(ValueError) as excinfo:
        retriever.search("valid query", top_k=0)
    assert "top_k must be a positive integer" in str(excinfo.value)

    with pytest.raises(TypeError):
        retriever.search(12345)  # type: ignore


@patch("rag.retrieve.bm25.Elasticsearch")
def test_bm25_empty_search_results(mock_es_class):
    """
    Verifies empty list output when search returns zero matches.
    """
    mock_client = MagicMock()
    mock_client.indices.exists.return_value = True
    mock_client.search.return_value = {"hits": {"hits": []}}
    mock_es_class.return_value = mock_client

    retriever = BM25Retriever()
    res = retriever.search("query with no matches")
    assert res == []

