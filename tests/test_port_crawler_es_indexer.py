"""Tests for mock embedder and ES indexer mapping (offline)."""

from unittest.mock import MagicMock, patch

from rag.config import AppConfig
from rag.index.embedder import MockEmbeddingProvider, get_embedding_provider
from rag.index.indexer import ElasticsearchIndexer
from rag.models import ChunkModel


def test_mock_embedder_dimension():
    config = AppConfig(embedding_provider="mock", embedding_dimension=384)
    provider = get_embedding_provider(config)
    vec = provider.embed_text("C-DAC website chunk")
    assert len(vec) == 384


def test_index_mapping_contains_dense_vector():
    config = AppConfig(embedding_dimension=384)
    indexer = ElasticsearchIndexer(config)
    mapping = indexer._expected_mapping()
    assert mapping["mappings"]["properties"]["embedding"]["dims"] == 384


@patch("rag.index.indexer.Elasticsearch")
def test_index_chunks_validates_dimensions(mock_es):
    config = AppConfig(embedding_dimension=384)
    client = MagicMock()
    mock_es.return_value = client
    indexer = ElasticsearchIndexer(config)
    chunk = ChunkModel(
        chunk_id="doc_ab_chunk_000",
        document_id="ab",
        document_name="About",
        content="C-DAC content for indexing test " * 5,
        source_url="https://cdac.gov.in/about",
        chunk_index=0,
    )
    embeddings = [MockEmbeddingProvider(config).embed_text(chunk.content)]
    stats = indexer.index_chunks([chunk], embeddings)
    assert stats["total_received"] == 1

