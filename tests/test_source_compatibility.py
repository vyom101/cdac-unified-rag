"""Source tests retained during the unified Phase 1 port."""

from unittest.mock import MagicMock, patch

from rag.config import AppConfig
from rag.crawl.pdf_downloader import PdfDownloadStore
from rag.crawl.url_utils import is_irrelevant_asset_url, is_pdf_url, is_processable_url
from rag.index.embedder import MockEmbeddingProvider, get_embedding_provider
from rag.index.indexer import ElasticsearchIndexer
from rag.index.summary_builder import (
    build_chunk_summaries,
    build_composite_summaries,
    extractive_summary,
)
from rag.models import ChunkModel, FailureRecord


def _chunk(idx: int, text: str) -> ChunkModel:
    return ChunkModel(
        chunk_id=f"doc_ab_chunk_{idx:03d}",
        document_id="ab",
        document_name="About",
        content=text,
        source_url="https://cdac.gov.in/about",
        parent_url="https://cdac.gov.in/",
        sub_link="/about",
        depth=1,
        chunk_index=idx,
    )


def test_extractive_summary_shortens_text():
    text = (
        "C-DAC is a premier R&D organization. "
        "It operates from Pune, India. "
        "The centre develops advanced computing systems."
    )
    summary = extractive_summary(text, max_sentences=2)
    assert len(summary) < len(text)
    assert "C-DAC" in summary or "Pune" in summary


def test_build_chunk_summaries_for_each_chunk():
    chunks = [_chunk(0, "C-DAC develops PARAM supercomputers in Pune."),
              _chunk(1, "Advanced computing research continues.")]
    summaries = build_chunk_summaries(chunks)
    assert len(summaries) == 2
    assert summaries[0].chunk_id == chunks[0].chunk_id
    assert summaries[0].parent_url == "https://cdac.gov.in/"
    assert summaries[0].depth == 1


def test_build_composite_summaries_groups_by_ten():
    chunks = [_chunk(i, f"Chunk content number {i} about C-DAC research topics.")
              for i in range(12)]
    summaries = build_chunk_summaries(chunks)
    composites = build_composite_summaries(summaries, group_size=10)
    assert len(composites) == 2
    assert composites[0].group_size == 10
    assert composites[1].group_size == 2
    assert len(composites[0].chunk_ids) == 10


def test_source_mock_embedder_dimension():
    config = AppConfig(embedding_provider="mock", embedding_dimension=384)
    provider = get_embedding_provider(config)
    vec = provider.embed_text("C-DAC website chunk")
    assert len(vec) == 384


def test_source_index_mapping_contains_dense_vector():
    config = AppConfig(embedding_dimension=384)
    indexer = ElasticsearchIndexer(config, client=MagicMock())
    mapping = indexer._expected_mapping()
    assert mapping["mappings"]["properties"]["embedding"]["dims"] == 384


@patch("elasticsearch.helpers.bulk")
def test_source_index_chunks_validates_dimensions(mock_bulk):
    config = AppConfig(embedding_dimension=384)
    mock_bulk.return_value = (1, [])
    indexer = ElasticsearchIndexer(config, client=MagicMock())
    chunk = _chunk(0, "C-DAC content for indexing test " * 5)
    embeddings = [MockEmbeddingProvider(config).embed_text(chunk.content)]
    stats = indexer.index_chunks([chunk], embeddings)
    assert stats["total_received"] == 1


def test_source_config_retry_defaults():
    config = AppConfig()
    assert config.crawl_retry_max_attempts == 3
    assert config.crawl_retry_interval_seconds == 2.0


def test_source_failure_record_human_review_flag():
    failure = FailureRecord(
        url="https://cdac.gov.in/missing",
        error_type="TimeoutError",
        error_message="timed out",
        retry_count=3,
        failure_kind="http_request",
        needs_human_review=True,
    )
    assert failure.needs_human_review is True
    assert failure.failure_kind == "http_request"


def test_source_pdf_url_contract():
    assert is_pdf_url("https://cdac.gov.in/docs/report.pdf")
    assert not is_pdf_url("https://cdac.gov.in/about")
    assert is_irrelevant_asset_url("https://cdac.gov.in/static/style.css")
    assert not is_irrelevant_asset_url("https://cdac.gov.in/docs/report.pdf")
    assert is_processable_url("https://cdac.gov.in/index.aspx")
    assert not is_processable_url("https://cdac.gov.in/app.js")


