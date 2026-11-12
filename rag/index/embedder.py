"""Swappable mock, local, and custom API embedding providers."""

import abc
import hashlib
import logging
import random
from typing import Dict, List, Optional

import requests
from pydantic import SecretStr

from rag.config import AppConfig, load_config

logger = logging.getLogger("embedder")


class BaseEmbeddingProvider(abc.ABC):
    def __init__(self, config: AppConfig):
        self.config = config
        self.configured_dimension = config.embedding_dimension

    def _validate_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text)}")
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only.")

    def _validate_vector_dimension(self, vector: List[float]) -> None:
        if len(vector) != self.configured_dimension:
            raise ValueError(
                "Embedding dimension mismatch: Configured dimension is "
                f"{self.configured_dimension}, but generated vector has dimension "
                f"{len(vector)}."
            )

    @abc.abstractmethod
    def get_dimension(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError

    @abc.abstractmethod
    def embed_documents(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        raise NotImplementedError


class MockEmbeddingProvider(BaseEmbeddingProvider):
    def get_dimension(self) -> int:
        return self.configured_dimension

    def embed_text(self, text: str) -> List[float]:
        self._validate_text(text)
        randomizer = random.Random(hashlib.sha256(text.encode("utf-8")).digest())
        vector = [
            randomizer.uniform(-1.0, 1.0)
            for _ in range(self.configured_dimension)
        ]
        self._validate_vector_dimension(vector)
        return vector

    def embed_documents(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        if not isinstance(texts, list):
            raise TypeError(f"Input texts must be a list of strings, got {type(texts)}")
        for text in texts:
            self._validate_text(text)
        return [self.embed_text(text) for text in texts]


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.model_name = config.embedding_model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers library is not installed."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local model '{self.model_name}': {exc}"
            ) from exc
        actual_dimension = self.get_dimension()
        if actual_dimension != self.configured_dimension:
            raise ValueError(
                "Embedding dimension mismatch: Configured dimension in settings is "
                f"{self.configured_dimension}, but model '{self.model_name}' produces "
                f"vectors of dimension {actual_dimension}."
            )

    def get_dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_text(self, text: str) -> List[float]:
        self._validate_text(text)
        try:
            vector = self.model.encode(text, convert_to_numpy=True).tolist()
            self._validate_vector_dimension(vector)
            return vector
        except Exception as exc:
            raise RuntimeError(f"Local embedding failed: {exc}") from exc

    def embed_documents(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        if not isinstance(texts, list):
            raise TypeError(f"Input texts must be a list of strings, got {type(texts)}")
        for text in texts:
            self._validate_text(text)
        try:
            vectors = self.model.encode(
                texts,
                batch_size=batch_size if batch_size is not None else 32,
                convert_to_numpy=True,
            ).tolist()
            for vector in vectors:
                self._validate_vector_dimension(vector)
            return vectors
        except Exception as exc:
            raise RuntimeError(f"Local batch embedding failed: {exc}") from exc


class CustomAPIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.endpoint = config.embedding_endpoint or ""
        self.model_name = config.embedding_model
        self.api_key: Optional[SecretStr] = config.embedding_api_key

    def get_dimension(self) -> int:
        return self.configured_dimension

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key.get_secret_value():
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        return headers

    def embed_text(self, text: str) -> List[float]:
        self._validate_text(text)
        return self.embed_documents([text])[0]

    def embed_documents(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> List[List[float]]:
        if not self.endpoint:
            raise ValueError(
                "EMBEDDING_ENDPOINT is not configured in Custom API embedding provider."
            )
        if not isinstance(texts, list):
            raise TypeError(f"Input texts must be a list of strings, got {type(texts)}")
        for text in texts:
            self._validate_text(text)
        size = batch_size if batch_size is not None else 32
        all_vectors = []
        for offset in range(0, len(texts), size):
            batch = texts[offset:offset + size]
            try:
                response = requests.post(
                    self.endpoint,
                    json={"model": self.model_name, "input": batch},
                    headers=self._get_headers(),
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list):
                    vectors = data
                elif isinstance(data.get("data"), list):
                    ordered = sorted(data["data"], key=lambda item: item.get("index", 0))
                    vectors = [item["embedding"] for item in ordered]
                elif isinstance(data.get("embeddings"), list):
                    vectors = data["embeddings"]
                elif isinstance(data.get("embedding"), list):
                    vectors = [data["embedding"]]
                else:
                    raise ValueError(
                        f"Unrecognized response format from custom API endpoint: {data}"
                    )
                if len(vectors) != len(batch):
                    raise ValueError(
                        f"Custom API returned {len(vectors)} vectors, but requested "
                        f"batch size was {len(batch)}."
                    )
                for vector in vectors:
                    self._validate_vector_dimension(vector)
                    all_vectors.append(vector)
            except ValueError:
                raise
            except Exception as exc:
                message = str(exc)
                if self.api_key and self.api_key.get_secret_value():
                    message = message.replace(
                        self.api_key.get_secret_value(), "********"
                    )
                raise RuntimeError(
                    f"Custom API embedding call failed: {message}"
                ) from exc
        return all_vectors


def get_embedding_provider(
    config: Optional[AppConfig] = None,
) -> BaseEmbeddingProvider:
    config = config or load_config()
    provider = config.embedding_provider.lower()
    if provider == "mock":
        return MockEmbeddingProvider(config)
    if provider == "local":
        return LocalEmbeddingProvider(config)
    if provider in {"custom_api", "openai"}:
        return CustomAPIEmbeddingProvider(config)
    return MockEmbeddingProvider(config)
