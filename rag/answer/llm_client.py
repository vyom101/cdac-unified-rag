"""Provider-neutral mock and custom REST LLM clients."""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests
from pydantic import SecretStr

from rag.config import AppConfig, load_config

logger = logging.getLogger("llm_client")


class BaseLLMClient(ABC):
    @abstractmethod
    def generate_completion(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.0
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_structured_completion(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.0
    ) -> Dict[str, Any]:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    def generate_completion(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.0
    ) -> str:
        return "Mock LLM text completion. The input context was successfully received."

    def generate_structured_completion(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.0
    ) -> Dict[str, Any]:
        try:
            if "blocks" in user_prompt:
                if "HNSW" in user_prompt or "hnsw" in user_prompt:
                    chunks = [
                        {
                            "section": "HNSW Algorithm",
                            "content": (
                                "HNSW (Hierarchical Navigable Small World) is a "
                                "state-of-the-art graph-based algorithm for "
                                "approximate nearest neighbor search."
                            ),
                            "source_pages": [1, 2],
                        },
                        {
                            "section": "HNSW Algorithm",
                            "content": (
                                "The search process in HNSW proceeds by navigating "
                                "a series of layered graphs from top to bottom."
                            ),
                            "source_pages": [2],
                        },
                    ]
                else:
                    chunks = [{
                        "section": "Introduction",
                        "content": (
                            "This is a mock chunk extracted from the supplied local "
                            "document block context. It represents structured, "
                            "context-preserving information."
                        ),
                        "source_pages": [1],
                    }]
            else:
                chunks = [{
                    "section": "General Content",
                    "content": "Default mock LLM-structured chunk.",
                    "source_pages": [1],
                }]
        except Exception:
            chunks = [{
                "section": "Error",
                "content": "Fallback mock chunk due to parsing error.",
                "source_pages": [1],
            }]
        return {"chunks": chunks}


class CustomAPILLMClient(BaseLLMClient):
    def __init__(self, config: AppConfig):
        self.endpoint = config.llm_endpoint or ""
        self.model = config.llm_model
        self.api_key: Optional[SecretStr] = config.llm_api_key

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key.get_secret_value():
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        return headers

    def generate_completion(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.0
    ) -> str:
        if not self.endpoint:
            raise ValueError("LLM_ENDPOINT is not configured in custom LLM client.")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        try:
            response = requests.post(
                self.endpoint, json=payload, headers=self._get_headers(), timeout=180.0
            )
            response.raise_for_status()
            data = response.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            if "completion" in data:
                return data["completion"]
            if "text" in data:
                return data["text"]
            return str(data)
        except Exception as exc:
            raise RuntimeError(f"LLM API call failed: {exc}") from exc

    def generate_structured_completion(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.0
    ) -> Dict[str, Any]:
        raw = self.generate_completion(
            system_prompt
            + "\n\nYou MUST return a valid JSON object. Do not wrap in markdown or backticks.",
            user_prompt,
            temperature,
        )
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(json)?\n", "", cleaned)
            cleaned = re.sub(r"\n```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except Exception as exc:
            raise ValueError(
                f"LLM did not return a valid JSON structure: {exc}"
            ) from exc


def get_llm_client(config: Optional[AppConfig] = None) -> BaseLLMClient:
    config = config or load_config()
    provider = config.llm_provider.lower()
    if provider == "mock":
        return MockLLMClient()
    if provider in {"custom_api", "openai"}:
        return CustomAPILLMClient(config)
    logger.warning("Unknown LLM provider '%s'; using mock.", provider)
    return MockLLMClient()
