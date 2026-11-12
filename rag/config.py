"""Unified, environment-overridable application configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, ValidationInfo, field_validator, model_validator


class AppConfig(BaseModel):
    project_name: str = "C-DAC Unified RAG"
    extracted_data_dir: str = "data/extracted"
    raw_data_dir: str = "data/raw"
    pdf_data_dir: str = "data/pdfs"
    pdf_manifest_path: str = "data/pdfs/manifest.jsonl"
    crawl_run_report_path: str = "data/reports/crawl_run_report.json"

    crawl_start_url: str = "https://cdac.gov.in/"
    crawl_allowed_domain: Optional[str] = "cdac.gov.in"
    crawl_max_depth: Optional[int] = None
    crawl_max_urls: int = 500
    crawl_max_duration_seconds: int = 3600
    crawl_max_response_bytes: int = 5_000_000
    crawl_max_redirects: int = 5
    crawl_max_url_length: int = 2048
    crawl_user_agent: str = "CdacResearchBot/1.0 (Educational)"
    crawl_download_delay: float = 1.5
    crawl_seed_sitemaps: bool = True
    crawl_retry_max_attempts: int = 3
    crawl_max_retry_attempts: int = 3
    crawl_retry_interval_seconds: float = 2.0
    crawl_whole_run_max_attempts: int = 1
    crawl_whole_run_retry_interval_seconds: float = 30.0
    crawl_state_path: str = "data/crawl/state.json"
    crawl_success_log_path: str = "data/logs/success.log"
    crawl_error_log_path: str = "data/logs/error.log"

    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "rag_corpus_v1"
    elasticsearch_username: Optional[str] = None
    elasticsearch_password: Optional[SecretStr] = None

    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_endpoint: Optional[str] = None
    embedding_api_key: Optional[SecretStr] = None
    embedding_dimension: int = 384

    llm_provider: str = "mock"
    llm_endpoint: Optional[str] = None
    llm_model: str = "mentor-provided-model"
    llm_api_key: Optional[SecretStr] = None
    llm_batch_size: int = 2

    retrieval_mode: str = "hybrid"
    top_k: int = 5
    num_candidates: int = 50
    hybrid_candidate_pool: int = 20
    keyword_weight: float = 0.3
    semantic_weight: float = 0.7
    fusion_strategy: str = "min_max"
    rrf_k: int = 60
    retrieval_domain_filter_enabled: bool = True
    enable_reference_metadata: bool = True
    figure_reference_boost: float = 4.0
    enable_image_query_boost: bool = True
    image_query_boost: float = 2.0
    hybrid_eval_keyword_weight: float = 0.3
    hybrid_eval_semantic_weight: float = 0.7
    hnsw_benchmark_sizes: list[int] = Field(default_factory=lambda: [1000, 5000, 10000])

    kb_source: str = "all"
    grounding_min_score: float = 0.35
    grounding_abstain_on_role_mismatch: bool = True
    kg_store_path: str = "data/kg/graph_v1.json"
    validation_representative_links_path: str = "evaluation/representative_links.json"

    html_max_chunk_size: int = 1200
    html_chunk_overlap: int = 150
    html_min_chunk_size: int = 80
    composite_group_size: int = 10
    pdf_max_chunk_size: int = 1200
    pdf_chunk_overlap: int = 100
    pdf_min_chunk_size: int = 300
    duplicate_similarity_threshold: float = 0.92
    max_chunk_size: int = 1200
    chunk_overlap: int = 100
    min_chunk_size: int = 300
    use_llm_assisted_organization: bool = False

    ocr_enabled: bool = True
    ocr_dpi: int = 300
    ocr_language: str = "eng"
    ocr_tesseract_cmd: Optional[str] = None
    image_extraction_enabled: bool = True
    image_output_dir: str = "data/extracted/images"
    image_min_dimension: int = 50
    min_image_chunk_chars: int = 15

    @field_validator(
        "project_name",
        "elasticsearch_url",
        "elasticsearch_index",
        "pdf_data_dir",
        "extracted_data_dir",
        mode="before",
    )
    @classmethod
    def validate_non_empty_strings(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            field_name = info.field_name.upper() if info.field_name else "FIELD"
            raise ValueError(
                f"Required configuration field '{field_name}' cannot be empty or missing."
            )
        return value

    @field_validator("top_k")
    @classmethod
    def validate_top_k_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"TOP_K must be a positive integer (got {value}).")
        return value

    @field_validator("num_candidates")
    @classmethod
    def validate_num_candidates_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"NUM_CANDIDATES must be a positive integer (got {value}).")
        return value

    @field_validator("hybrid_candidate_pool")
    @classmethod
    def validate_hybrid_candidate_pool_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(
                f"HYBRID_CANDIDATE_POOL must be a positive integer (got {value})."
            )
        return value

    @field_validator("max_chunk_size")
    @classmethod
    def validate_max_chunk_size_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"MAX_CHUNK_SIZE must be a positive integer (got {value}).")
        return value

    @field_validator("llm_batch_size")
    @classmethod
    def validate_llm_batch_size_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"LLM_BATCH_SIZE must be a positive integer (got {value}).")
        return value

    @field_validator("retrieval_mode")
    @classmethod
    def validate_retrieval_mode(cls, value: str) -> str:
        allowed = {"bm25", "vector", "hybrid"}
        if value.lower() not in allowed:
            raise ValueError(f"RETRIEVAL_MODE must be one of {allowed} (got '{value}').")
        return value.lower()

    @field_validator("fusion_strategy")
    @classmethod
    def validate_fusion_strategy(cls, value: str) -> str:
        allowed = {"min_max", "rrf"}
        if value.lower() not in allowed:
            raise ValueError(f"FUSION_STRATEGY must be one of {allowed} (got '{value}').")
        return value.lower()

    @field_validator("rrf_k")
    @classmethod
    def validate_rrf_k_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"RRF_K must be a positive integer (got {value}).")
        return value

    @field_validator("crawl_max_urls", "crawl_max_retry_attempts")
    @classmethod
    def validate_crawl_positive_integers(cls, value: int, info: ValidationInfo) -> int:
        if value <= 0:
            field_name = info.field_name.upper() if info.field_name else "FIELD"
            raise ValueError(f"{field_name} must be a positive integer (got {value}).")
        return value

    @field_validator("keyword_weight", "semantic_weight")
    @classmethod
    def validate_weights_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError(f"Retrieval weight must be non-negative (got {value}).")
        return value

    @field_validator("duplicate_similarity_threshold")
    @classmethod
    def validate_duplicate_similarity_threshold(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError(
                f"DUPLICATE_SIMILARITY_THRESHOLD must be within (0.0, 1.0] (got {value})."
            )
        return value

    @field_validator("kb_source")
    @classmethod
    def valid_source(cls, value: str) -> str:
        value = value.lower()
        allowed = {
            "html_only",
            "pdf_only",
            "html_and_pdf",
            "uploaded_pdf_only",
            "all",
        }
        if value not in allowed:
            raise ValueError(
                "kb_source must be exactly one of html_only, pdf_only, "
                "html_and_pdf, uploaded_pdf_only, or all"
            )
        return value

    @model_validator(mode="after")
    def validate_hybrid_weights(self) -> "AppConfig":
        total_weight = self.keyword_weight + self.semantic_weight
        if abs(total_weight - 1.0) > 1e-5:
            raise ValueError(
                f"Hybrid weights KEYWORD_WEIGHT + SEMANTIC_WEIGHT must equal 1.0 "
                f"(got KEYWORD_WEIGHT={self.keyword_weight} + SEMANTIC_WEIGHT="
                f"{self.semantic_weight} = {total_weight})."
            )
        return self

    def get_masked_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        for key in ("elasticsearch_password", "llm_api_key", "embedding_api_key"):
            data[key] = "********" if getattr(self, key) else None
        return data


_SECTIONS = {
    "crawler": "crawl",
    "elasticsearch": "elasticsearch",
    "embedding": "embedding",
    "llm": "llm",
    "retrieval": "",
    "grounding": "grounding",
    "knowledge_graph": "kg",
    "kb": "kb",
    "paths": "",
    "chunking": "",
    "html_chunking": "html",
    "pdf_chunking": "pdf",
    "ocr": "ocr",
    "image_extraction": "image",
}
_RENAMES = {
    "weights_keyword": "keyword_weight",
    "weights_semantic": "semantic_weight",
    "index": "elasticsearch_index",
    "url": "elasticsearch_url",
    "mode": "retrieval_mode",
    "retrieval_top_k": "top_k",
    "retrieval_num_candidates": "num_candidates",
    "retrieval_mode": "retrieval_mode",
    "kb_source": "kb_source",
    "domain_filter_enabled": "retrieval_domain_filter_enabled",
    "image_enabled": "image_extraction_enabled",
    "image_output_dir": "image_output_dir",
    "image_min_dimension_px": "image_min_dimension",
    "image_min_ocr_text_chars": "min_image_chunk_chars",
    "store_path": "kg_store_path",
}


def _flatten(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section, value in data.items():
        if not isinstance(value, dict):
            result[section.lower()] = value
            continue
        prefix = _SECTIONS.get(section.lower(), section.lower())
        for key, item in value.items():
            if isinstance(item, dict):
                for child, child_value in item.items():
                    name = "_".join(part for part in (prefix, key, child) if part)
                    result[_RENAMES.get(name, name)] = child_value
            else:
                name = "_".join(part for part in (prefix, key) if part)
                result[_RENAMES.get(name, name)] = item
    return result


def load_config(
    config_path: str | Path | None = None, env_path: str | Path | None = None
) -> AppConfig:
    root = Path(__file__).resolve().parent.parent
    load_dotenv(Path(env_path) if env_path else root / ".env")
    path = Path(config_path) if config_path else root / "config" / "settings.yaml"
    values: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values.update(_flatten(raw))
    if "pdf_max_chunk_size" in values and "max_chunk_size" not in values:
        values["max_chunk_size"] = values["pdf_max_chunk_size"]
    if "pdf_chunk_overlap" in values and "chunk_overlap" not in values:
        values["chunk_overlap"] = values["pdf_chunk_overlap"]
    if "pdf_min_chunk_size" in values and "min_chunk_size" not in values:
        values["min_chunk_size"] = values["pdf_min_chunk_size"]
    for field in AppConfig.model_fields:
        env_key = field.upper()
        if env_key in os.environ:
            values[field] = os.environ[env_key]
    if os.environ.get("LLM_ENDPOINT") and "LLM_PROVIDER" not in os.environ:
        values["llm_provider"] = "custom_api"
    return AppConfig(**values)
