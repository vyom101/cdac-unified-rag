from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from rag.answer.llm_client import CustomAPILLMClient, MockLLMClient
from rag.config import load_config
from rag.index.embedder import (
    CustomAPIEmbeddingProvider,
    LocalEmbeddingProvider,
    MockEmbeddingProvider,
    get_embedding_provider,
)


def test_mock_embedding_is_deterministic_and_validates_input():
    config = load_config()
    config.embedding_dimension = 128
    provider = MockEmbeddingProvider(config)
    first = provider.embed_text("Machine Learning at CDAC")
    assert len(first) == 128
    assert first == provider.embed_text("Machine Learning at CDAC")
    assert first != provider.embed_text("Different text content")
    with pytest.raises(ValueError, match="cannot be empty"):
        provider.embed_text(" ")
    with pytest.raises(TypeError):
        provider.embed_documents("not a list")


@patch("sentence_transformers.SentenceTransformer")
def test_local_embedding_provider(mock_transformer):
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = 384
    model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 384)
    mock_transformer.return_value = model
    config = load_config()
    config.embedding_model = "all-MiniLM-L6-v2"
    config.embedding_dimension = 384
    provider = LocalEmbeddingProvider(config)
    assert provider.get_dimension() == 384
    assert provider.embed_text("Local text") == [0.1] * 384
    model.encode.assert_called_with("Local text", convert_to_numpy=True)


@patch("requests.post")
def test_custom_api_embedding_provider(mock_post):
    response = MagicMock()
    response.json.return_value = {
        "data": [
            {"embedding": [0.12] * 256, "index": 0},
            {"embedding": [0.34] * 256, "index": 1},
        ]
    }
    mock_post.return_value = response
    config = load_config()
    config.embedding_endpoint = "https://mentor-api.cdac.in/v1/embeddings"
    config.embedding_model = "cdac-embed-large"
    config.embedding_api_key = SecretStr("secret_mentor_key")
    config.embedding_dimension = 256
    vectors = CustomAPIEmbeddingProvider(config).embed_documents(
        ["Page text one", "Page text two"], batch_size=2
    )
    assert vectors[0] == [0.12] * 256
    assert vectors[1] == [0.34] * 256
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == (
        "Bearer secret_mentor_key"
    )


def test_embedding_factory_fallback():
    config = load_config()
    config.embedding_provider = "unsupported_provider_xyz"
    assert isinstance(get_embedding_provider(config), MockEmbeddingProvider)


def test_mock_llm_client_contract():
    client = MockLLMClient()
    assert "Mock LLM text completion" in client.generate_completion("", "")
    response = client.generate_structured_completion("", "blocks mentioning HNSW")
    assert len(response["chunks"]) == 2
    assert response["chunks"][0]["source_pages"] == [1, 2]


@patch("requests.post")
def test_custom_api_llm_client_parses_openai_and_structured_json(mock_post):
    response = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"content": '```json\n{"chunks": []}\n```'}}]
    }
    mock_post.return_value = response
    config = load_config()
    config.llm_endpoint = "https://example.test/v1/chat"
    config.llm_model = "test-model"
    config.llm_api_key = SecretStr("secret")
    client = CustomAPILLMClient(config)
    assert client.generate_structured_completion("system", "user") == {"chunks": []}
    assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer secret"
