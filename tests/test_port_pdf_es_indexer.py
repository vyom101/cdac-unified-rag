import pytest
from unittest.mock import MagicMock, patch
from pydantic import SecretStr, ValidationError

from rag.config import AppConfig, load_config
from rag.index.pdf_chunker import ChunkModel
from rag.index.indexer import ElasticsearchIndexer


# ==============================================================================
# UNIT TESTS FOR ELASTICSEARCH INDEXER (100% Offline)
# ==============================================================================

@patch("rag.index.indexer.Elasticsearch")
def test_indexer_initialization_and_auth(mock_es_class):
    """
    Verifies that the indexer correctly initializes, reads configurations, and maps auth parameters.
    """
    config = load_config()
    config.elasticsearch_url = "http://localhost:9200"
    config.elasticsearch_username = "cdac_user"
    config.elasticsearch_password = SecretStr("cdac_pass_123")
    config.elasticsearch_index = "cdac_test_index"

    indexer = ElasticsearchIndexer(config)

    assert indexer.url == "http://localhost:9200"
    assert indexer.index_name == "cdac_test_index"
    assert indexer.username == "cdac_user"
    assert indexer.password == "cdac_pass_123"

    mock_es_class.assert_called_once_with(
        "http://localhost:9200",
        basic_auth=("cdac_user", "cdac_pass_123"),
        verify_certs=True,
        request_timeout=30.0
    )


@patch("rag.index.indexer.Elasticsearch")
def test_indexer_ping_falls_back_to_info(mock_es_class):
    """
    Ensures connectivity succeeds when HEAD ping fails but cluster info is reachable.
    """
    mock_client = MagicMock()
    mock_client.ping.return_value = False
    mock_client.info.return_value = {"version": {"number": "8.12.0"}}
    mock_es_class.return_value = mock_client

    indexer = ElasticsearchIndexer()
    assert indexer.ping() is True


@patch("rag.index.indexer.Elasticsearch")
def test_indexer_connection_failure_handling(mock_es_class):
    """
    Ensures connection validation throws ConnectionError when Elasticsearch is unreachable.
    """
    mock_client = MagicMock()
    mock_client.ping.return_value = False
    mock_client.info.side_effect = ConnectionError("Elasticsearch unreachable")
    mock_es_class.return_value = mock_client

    indexer = ElasticsearchIndexer()
    
    with pytest.raises(ConnectionError) as excinfo:
        indexer.check_connection()
    assert "Elasticsearch server is unreachable" in str(excinfo.value)


@patch("rag.index.indexer.Elasticsearch")
def test_index_creation_mapping_and_dimension(mock_es_class):
    """
    Verifies index creation payloads, Dense-Vector mapping, and dimension validation.
    """
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.indices.exists.return_value = False  # Simulate index doesn't exist yet
    mock_es_class.return_value = mock_client

    config = load_config()
    config.embedding_dimension = 512
    config.elasticsearch_index = "cdac_dim_index"

    indexer = ElasticsearchIndexer(config)
    created = indexer.create_index_if_not_exists()

    assert created is True
    mock_client.indices.exists.assert_called_once_with(index="cdac_dim_index")
    mock_client.indices.create.assert_called_once()
    
    # Check body arguments inside created index mapping
    call_kwargs = mock_client.indices.create.call_args[1]
    assert call_kwargs["index"] == "cdac_dim_index"
    
    properties = call_kwargs["body"]["mappings"]["properties"]
    assert properties["content"]["type"] == "text"
    assert properties["embedding"]["type"] == "dense_vector"
    assert properties["embedding"]["dims"] == 512  # Dynamic dimension check!
    assert properties["embedding"]["index"] is True
    assert properties["embedding"]["similarity"] == "cosine"


@patch("rag.index.indexer.Elasticsearch")
@patch("rag.index.indexer.bulk")
def test_idempotent_bulk_indexing(mock_bulk_helper, mock_es_class):
    """
    Verifies that the indexer formats document sources correctly and enforces idempotent document updates.
    """
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.indices.exists.return_value = True  # Index already exists
    mock_es_class.return_value = mock_client

    # Configure mock return value for helper.bulk
    mock_bulk_helper.return_value = (5, [])  # (success_count, failed_items)

    config = load_config()
    config.embedding_dimension = 384
    config.elasticsearch_index = "cdac_bulk_index"
    indexer = ElasticsearchIndexer(config)

    # Construct two test ChunkModels
    chunks = [
        ChunkModel(
            chunk_id="doc_1_chunk_001",
            document_id="doc_1",
            document_name="doc1.pdf",
            section="Intro",
            page_numbers=[1],
            content="Sample text content for chunk one.",
            chunk_index=1
        ),
        ChunkModel(
            chunk_id="doc_1_chunk_002",
            document_id="doc_1",
            document_name="doc1.pdf",
            section="Intro",
            page_numbers=[2],
            content="Sample text content for chunk two.",
            chunk_index=2
        )
    ]
    embeddings = [
        [0.1] * 384,
        [0.2] * 384
    ]

    stats = indexer.index_chunks(chunks, embeddings)

    # Check stats return format
    assert stats["total_received"] == 2
    assert stats["successfully_indexed"] == 5
    assert stats["failed"] == 0
    assert isinstance(stats["duration_seconds"], float)

    # Verify formatting payload passed to bulk helper
    mock_bulk_helper.assert_called_once()
    bulk_actions = mock_bulk_helper.call_args[0][1]
    
    assert len(bulk_actions) == 2
    
    # 1. Idempotency Check: action must set the explicit document _id
    assert bulk_actions[0]["_id"] == "doc_1_chunk_001"
    assert bulk_actions[1]["_id"] == "doc_1_chunk_002"
    
    # 2. Payload Check
    assert bulk_actions[0]["_index"] == "cdac_bulk_index"
    assert bulk_actions[0]["_source"]["chunk_id"] == "doc_1_chunk_001"
    assert bulk_actions[0]["_source"]["content"] == "Sample text content for chunk one."
    assert bulk_actions[0]["_source"]["embedding"] == embeddings[0]


@patch("rag.index.indexer.Elasticsearch")
def test_indexing_mismatched_dimensions(mock_es_class):
    """
    Verifies that trying to index vectors with dimensions different from config raises ValueError.
    """
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_es_class.return_value = mock_client

    config = load_config()
    config.embedding_dimension = 384
    indexer = ElasticsearchIndexer(config)

    chunks = [
        ChunkModel(
            chunk_id="chunk_1",
            document_id="doc_1",
            document_name="test.pdf",
            page_numbers=[1],
            content="Short test content.",
            chunk_index=1
        )
    ]
    
    # Embedding with 128 dimension instead of 384
    invalid_embeddings = [[0.5] * 128]

    with pytest.raises(ValueError) as excinfo:
        indexer.index_chunks(chunks, invalid_embeddings)
    assert "does not match configured dimension" in str(excinfo.value)

