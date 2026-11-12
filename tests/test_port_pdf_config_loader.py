import pytest
import os
from pathlib import Path
from pydantic import ValidationError, SecretStr
from rag.config import load_config, AppConfig

@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """
    Cleans any existing configuration environment variables before each test
    to guarantee test isolation.
    """
    keys_to_clear = [
        "PROJECT_NAME",
        "ELASTICSEARCH_URL",
        "ELASTICSEARCH_INDEX",
        "ELASTICSEARCH_USERNAME",
        "ELASTICSEARCH_PASSWORD",
        "LLM_PROVIDER",
        "LLM_ENDPOINT",
        "LLM_MODEL",
        "LLM_API_KEY",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_ENDPOINT",
        "EMBEDDING_DIMENSION",
        "RETRIEVAL_MODE",
        "TOP_K",
        "NUM_CANDIDATES",
        "KEYWORD_WEIGHT",
        "SEMANTIC_WEIGHT",
        "FUSION_STRATEGY",
        "RRF_K",
        "PDF_DATA_DIR",
        "EXTRACTED_DATA_DIR",
        "MAX_CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "LLM_BATCH_SIZE",
    ]
    for key in keys_to_clear:
        monkeypatch.delenv(key, raising=False)


def test_valid_default_config():
    """
    Validates that a default config object can be parsed and holds the expected default values.
    """
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.project_name == "C-DAC Unified RAG"
    assert config.elasticsearch_url == "http://localhost:9200"
    assert config.elasticsearch_index == "rag_corpus_v1"
    assert config.retrieval_mode == "hybrid"
    assert config.top_k == 5
    assert config.num_candidates == 50
    assert config.keyword_weight == 0.3
    assert config.semantic_weight == 0.7
    assert config.fusion_strategy == "min_max"
    assert config.rrf_k == 60
    assert config.max_chunk_size == 1200
    assert config.chunk_overlap == 100
    assert config.html_chunk_overlap == 150
    assert config.pdf_chunk_overlap == 100
    assert config.pdf_min_chunk_size == 300


def test_missing_required_values(monkeypatch):
    """
    Validates that clear, descriptive errors are raised when required configurations are empty.
    """
    # Override ELASTICSEARCH_URL with empty string in env
    monkeypatch.setenv("ELASTICSEARCH_URL", " ")
    with pytest.raises(ValidationError) as excinfo:
        load_config()
    assert "Required configuration field 'ELASTICSEARCH_URL'" in str(excinfo.value)


def test_invalid_retrieval_mode(monkeypatch):
    """
    Validates that retrieval mode must be one of: bm25, vector, hybrid.
    """
    monkeypatch.setenv("RETRIEVAL_MODE", "invalid_mode")
    with pytest.raises(ValidationError) as excinfo:
        load_config()
    assert "RETRIEVAL_MODE must be one of" in str(excinfo.value)


def test_invalid_top_k(monkeypatch):
    """
    Validates that TOP_K must be a positive integer.
    """
    # Test zero
    monkeypatch.setenv("TOP_K", "0")
    with pytest.raises(ValidationError) as excinfo:
        load_config()
    assert "TOP_K must be a positive integer" in str(excinfo.value)

    # Test negative
    monkeypatch.setenv("TOP_K", "-5")
    with pytest.raises(ValidationError) as excinfo:
        load_config()
    assert "TOP_K must be a positive integer" in str(excinfo.value)


def test_invalid_num_candidates(monkeypatch):
    """
    Validates that NUM_CANDIDATES must be a positive integer.
    """
    # Test negative
    monkeypatch.setenv("NUM_CANDIDATES", "-1")
    with pytest.raises(ValidationError) as excinfo:
        load_config()
    assert "NUM_CANDIDATES must be a positive integer" in str(excinfo.value)


def test_invalid_fusion_strategy(monkeypatch):
    """
    Validates that fusion strategy must be one of: min_max, rrf.
    """
    monkeypatch.setenv("FUSION_STRATEGY", "invalid_strategy")
    with pytest.raises(ValidationError) as excinfo:
        load_config()
    assert "FUSION_STRATEGY must be one of" in str(excinfo.value)


def test_invalid_rrf_k(monkeypatch):
    """
    Validates that RRF_K must be a positive integer.
    """
    monkeypatch.setenv("RRF_K", "0")
    with pytest.raises(ValidationError) as excinfo:
        load_config()
    assert "RRF_K must be a positive integer" in str(excinfo.value)


def test_invalid_negative_weights(monkeypatch):
    """
    Validates that retrieval weights must be non-negative.
    """
    monkeypatch.setenv("KEYWORD_WEIGHT", "-0.1")
    monkeypatch.setenv("SEMANTIC_WEIGHT", "1.1")
    with pytest.raises(ValidationError) as excinfo:
        load_config()
    assert "non-negative" in str(excinfo.value)


def test_invalid_hybrid_weights(monkeypatch):
    """
    Validates that hybrid weights must sum to exactly 1.0 (within float tolerance).
    """
    # Sum is 1.1 (invalid)
    monkeypatch.setenv("KEYWORD_WEIGHT", "0.4")
    monkeypatch.setenv("SEMANTIC_WEIGHT", "0.7")
    with pytest.raises(ValidationError) as excinfo:
        load_config()
    assert "weights KEYWORD_WEIGHT + SEMANTIC_WEIGHT must equal 1.0" in str(excinfo.value)

    # Sum is 0.999 (close enough within float tolerance 1e-5)
    monkeypatch.setenv("KEYWORD_WEIGHT", "0.300001")
    monkeypatch.setenv("SEMANTIC_WEIGHT", "0.699999")
    config = load_config()
    assert abs((config.keyword_weight + config.semantic_weight) - 1.0) <= 1e-5


def test_env_variable_overrides(monkeypatch):
    """
    Validates that environment variables correctly override the settings.yaml configuration values.
    """
    monkeypatch.setenv("PROJECT_NAME", "Override CDAC RAG")
    monkeypatch.setenv("ELASTICSEARCH_URL", "http://prod-es:9200")
    monkeypatch.setenv("ELASTICSEARCH_INDEX", "prod_index")
    monkeypatch.setenv("LLM_PROVIDER", "custom_api")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    monkeypatch.setenv("TOP_K", "10")
    monkeypatch.setenv("MAX_CHUNK_SIZE", "1500")

    config = load_config()
    
    assert config.project_name == "Override CDAC RAG"
    assert config.elasticsearch_url == "http://prod-es:9200"
    assert config.elasticsearch_index == "prod_index"
    assert config.llm_provider == "custom_api"
    assert config.embedding_provider == "openai"
    assert config.retrieval_mode == "bm25"
    assert config.top_k == 10
    assert config.max_chunk_size == 1500


def test_secrets_masking(monkeypatch):
    """
    Validates that sensitive values are stored securely in SecretStr structures
    and are masked during standard logging exports.
    """
    monkeypatch.setenv("ELASTICSEARCH_PASSWORD", "super_secret_es_pass")
    monkeypatch.setenv("LLM_API_KEY", "super_secret_llm_api_key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "super_secret_embedding_key")

    config = load_config()
    
    # Assert type
    assert isinstance(config.elasticsearch_password, SecretStr)
    assert isinstance(config.llm_api_key, SecretStr)
    assert isinstance(config.embedding_api_key, SecretStr)

    # Assert raw secrets can be read with get_secret_value()
    assert config.elasticsearch_password.get_secret_value() == "super_secret_es_pass"
    assert config.llm_api_key.get_secret_value() == "super_secret_llm_api_key"
    assert config.embedding_api_key.get_secret_value() == "super_secret_embedding_key"

    # Assert they are masked when calling get_masked_dict() or repr()
    masked_dict = config.get_masked_dict()
    assert masked_dict["elasticsearch_password"] == "********"
    assert masked_dict["llm_api_key"] == "********"
    assert masked_dict["embedding_api_key"] == "********"
    
    # Assert that string representations do not leak the secrets
    assert "super_secret_es_pass" not in str(config.elasticsearch_password)
    assert "********" in str(config.elasticsearch_password)


def test_llm_batch_size_defaults_and_overrides(monkeypatch):
    """
    Verifies default, valid env overrides, and error bounds validation for LLM_BATCH_SIZE.
    """
    # 1. Default value when environment variable is not set
    config = load_config()
    assert config.llm_batch_size == 2

    # 2. Valid overrides: batch size of 1
    monkeypatch.setenv("LLM_BATCH_SIZE", "1")
    config_1 = load_config()
    assert config_1.llm_batch_size == 1

    # 3. Valid overrides: batch size of 5
    monkeypatch.setenv("LLM_BATCH_SIZE", "5")
    config_5 = load_config()
    assert config_5.llm_batch_size == 5

    # 4. Invalid override: zero
    monkeypatch.setenv("LLM_BATCH_SIZE", "0")
    with pytest.raises(ValidationError) as excinfo:
        load_config()
    assert "LLM_BATCH_SIZE must be a positive integer" in str(excinfo.value)

    # 5. Invalid override: negative
    monkeypatch.setenv("LLM_BATCH_SIZE", "-2")
    with pytest.raises(ValidationError) as excinfo:
        load_config()
    assert "LLM_BATCH_SIZE must be a positive integer" in str(excinfo.value)

    # 6. Invalid override: non-integer string
    monkeypatch.setenv("LLM_BATCH_SIZE", "invalid_int")
    with pytest.raises(ValidationError) as excinfo:
        load_config()
    assert any(err["type"] == "int_parsing" for err in excinfo.value.errors())

