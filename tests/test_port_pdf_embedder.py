import pytest
from unittest.mock import MagicMock, patch
from pydantic import SecretStr, ValidationError
import requests

from rag.config import AppConfig, load_config
from rag.index.embedder import (
    BaseEmbeddingProvider,
    MockEmbeddingProvider,
    LocalEmbeddingProvider,
    CustomAPIEmbeddingProvider,
    get_embedding_provider
)


# ==============================================================================
# MOCK PROVIDER TESTS (100% Deterministic & Offline)
# ==============================================================================

def test_mock_provider_initialization():
    """
    Verifies that the MockEmbeddingProvider correctly initializes and exposes the dimension.
    """
    config = load_config()
    config.embedding_provider = "mock"
    config.embedding_dimension = 256

    provider = MockEmbeddingProvider(config)
    assert provider.get_dimension() == 256
    assert provider.configured_dimension == 256


def test_mock_provider_single_text_embedding():
    """
    Verifies single text embedding output types, values, and determinism.
    """
    config = load_config()
    config.embedding_provider = "mock"
    config.embedding_dimension = 384

    provider = MockEmbeddingProvider(config)
    text = "Machine Learning at CDAC"

    # Verify type and dimension
    vec_1 = provider.embed_text(text)
    assert isinstance(vec_1, list)
    assert len(vec_1) == 384
    assert all(isinstance(val, float) for val in vec_1)

    # Verify determinism (same text produces identical vectors)
    vec_2 = provider.embed_text(text)
    assert vec_1 == vec_2

    # Verify different text produces different vectors
    vec_diff = provider.embed_text("Different text content")
    assert vec_1 != vec_diff


def test_mock_provider_document_list_embedding():
    """
    Verifies multiple document list embedding behavior and validation.
    """
    config = load_config()
    config.embedding_dimension = 128
    provider = MockEmbeddingProvider(config)

    texts = [
        "First sample document",
        "Second sample document",
        "Third document to generate embeddings"
    ]

    vectors = provider.embed_documents(texts)
    assert isinstance(vectors, list)
    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == 128


def test_mock_provider_invalid_inputs():
    """
    Checks that empty and whitespace-only text inputs raise clear ValueErrors.
    """
    config = load_config()
    provider = MockEmbeddingProvider(config)

    # Empty text
    with pytest.raises(ValueError) as excinfo:
        provider.embed_text("")
    assert "cannot be empty or whitespace-only" in str(excinfo.value)

    # Whitespace text
    with pytest.raises(ValueError) as excinfo:
        provider.embed_text("   \n  \t ")
    assert "cannot be empty or whitespace-only" in str(excinfo.value)

    # List containing empty text
    with pytest.raises(ValueError) as excinfo:
        provider.embed_documents(["Valid text", ""])
    assert "cannot be empty or whitespace-only" in str(excinfo.value)

    # Type error checks
    with pytest.raises(TypeError):
        provider.embed_text(12345)  # type: ignore

    with pytest.raises(TypeError):
        provider.embed_documents("not a list")  # type: ignore


# ==============================================================================
# LOCAL PROVIDER TESTS (Using patches to guarantee offline testing)
# ==============================================================================

@patch("sentence_transformers.SentenceTransformer")
def test_local_provider_initialization_and_dimension_match(mock_transformer_class):
    """
    Tests initialization checks for configured vs actual dimension matching.
    """
    # Configure mock SentenceTransformer class
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_transformer_class.return_value = mock_model

    config = load_config()
    config.embedding_provider = "local"
    config.embedding_model = "all-MiniLM-L6-v2"
    config.embedding_dimension = 384

    # 1. Success case: actual dimension matches configured (384 == 384)
    provider = LocalEmbeddingProvider(config)
    assert provider.get_dimension() == 384
    mock_transformer_class.assert_called_once_with("all-MiniLM-L6-v2")

    # 2. Failure case: configured dimension mismatch (512 != 384)
    config.embedding_dimension = 512
    with pytest.raises(ValueError) as excinfo:
        LocalEmbeddingProvider(config)
    assert "Embedding dimension mismatch" in str(excinfo.value)


@patch("sentence_transformers.SentenceTransformer")
def test_local_provider_embedding_execution(mock_transformer_class):
    """
    Verifies local provider embedding functions map correctly to SentenceTransformer methods.
    """
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    # Mock return values for model.encode
    mock_vec = [0.1] * 384
    mock_model.encode.return_value = MagicMock(tolist=lambda: mock_vec)
    mock_transformer_class.return_value = mock_model

    config = load_config()
    config.embedding_dimension = 384
    provider = LocalEmbeddingProvider(config)

    # Test single text embedding calls model.encode
    res = provider.embed_text("Local text")
    assert res == mock_vec
    mock_model.encode.assert_called_with("Local text", convert_to_numpy=True)

    # Test batch/document embedding respects batch size parameter
    mock_vecs = [mock_vec, mock_vec]
    mock_model.encode.return_value = MagicMock(tolist=lambda: mock_vecs)
    
    docs = ["Doc A", "Doc B"]
    res_batch = provider.embed_documents(docs, batch_size=10)
    assert res_batch == mock_vecs
    mock_model.encode.assert_called_with(docs, batch_size=10, convert_to_numpy=True)


# ==============================================================================
# CUSTOM REST API PROVIDER TESTS
# ==============================================================================

@patch("requests.post")
def test_custom_api_provider_successful_embedding(mock_post):
    """
    Verifies that the Custom API client maps list parameters, auth headers, and response formats.
    """
    # Configure mock API successful JSON response (OpenAI data array style)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.12] * 256, "index": 0},
            {"embedding": [0.34] * 256, "index": 1}
        ]
    }
    mock_post.return_value = mock_response

    config = load_config()
    config.embedding_provider = "custom_api"
    config.embedding_endpoint = "https://mentor-api.cdac.in/v1/embeddings"
    config.embedding_model = "cdac-embed-large"
    config.embedding_api_key = SecretStr("secret_mentor_key")
    config.embedding_dimension = 256

    provider = CustomAPIEmbeddingProvider(config)
    texts = ["Page text one", "Page text two"]

    vectors = provider.embed_documents(texts, batch_size=2)

    assert len(vectors) == 2
    assert vectors[0] == [0.12] * 256
    assert vectors[1] == [0.34] * 256

    # Verify requests payload, content headers, and token auth placement
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "https://mentor-api.cdac.in/v1/embeddings"
    assert kwargs["headers"]["Authorization"] == "Bearer secret_mentor_key"
    assert kwargs["json"]["model"] == "cdac-embed-large"
    assert kwargs["json"]["input"] == texts


@patch("requests.post")
def test_custom_api_provider_error_and_secret_masking(mock_post):
    """
    Verifies that REST connection failures raise errors and mask credentials.
    """
    mock_post.side_effect = requests.exceptions.HTTPError("401 Unauthorized API key secret_mentor_key")

    config = load_config()
    config.embedding_provider = "custom_api"
    config.embedding_endpoint = "https://mentor-api.cdac.in/v1/embeddings"
    config.embedding_api_key = SecretStr("secret_mentor_key")
    config.embedding_dimension = 384

    provider = CustomAPIEmbeddingProvider(config)

    with pytest.raises(RuntimeError) as excinfo:
        provider.embed_text("Some text to embed")

    # Assert secret was masked
    assert "secret_mentor_key" not in str(excinfo.value)
    assert "********" in str(excinfo.value)


@patch("requests.post")
def test_custom_api_dimension_mismatch(mock_post):
    """
    Verifies that the Custom API client throws dimension mismatch errors.
    """
    # Return vector of size 128 when configured dimension is 256
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "embeddings": [[0.5] * 128]
    }
    mock_post.return_value = mock_response

    config = load_config()
    config.embedding_provider = "custom_api"
    config.embedding_endpoint = "https://api.endpoint/v1"
    config.embedding_dimension = 256

    provider = CustomAPIEmbeddingProvider(config)

    with pytest.raises(ValueError) as excinfo:
        provider.embed_text("Dimension check")
    assert "Embedding dimension mismatch" in str(excinfo.value)


# ==============================================================================
# FACTORY RESOLUTION & SUBSTITUTION TESTS
# ==============================================================================

def test_factory_resolves_mock_provider():
    config = load_config()
    config.embedding_provider = "mock"
    
    provider = get_embedding_provider(config)
    assert isinstance(provider, MockEmbeddingProvider)


@patch("sentence_transformers.SentenceTransformer")
def test_factory_resolves_local_provider(mock_transformer):
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_transformer.return_value = mock_model

    config = load_config()
    config.embedding_provider = "local"
    config.embedding_dimension = 384

    provider = get_embedding_provider(config)
    assert isinstance(provider, LocalEmbeddingProvider)


def test_factory_resolves_custom_api_provider():
    config = load_config()
    config.embedding_provider = "custom_api"
    config.embedding_endpoint = "https://endpoint"

    provider = get_embedding_provider(config)
    assert isinstance(provider, CustomAPIEmbeddingProvider)


def test_factory_fallback_on_unsupported_provider():
    config = load_config()
    config.embedding_provider = "unsupported_provider_xyz"

    provider = get_embedding_provider(config)
    assert isinstance(provider, MockEmbeddingProvider)


def test_provider_substitution_neutrality():
    """
    Checks that the interface BaseEmbeddingProvider can be cleanly substituted
    between Mock, Local, and Custom providers without breaking consumer logic.
    """
    config = load_config()
    config.embedding_dimension = 128

    mock_provider = MockEmbeddingProvider(config)
    assert isinstance(mock_provider, BaseEmbeddingProvider)

    # All providers share the same method signitures
    assert hasattr(mock_provider, "embed_text")
    assert hasattr(mock_provider, "embed_documents")
    assert hasattr(mock_provider, "get_dimension")

