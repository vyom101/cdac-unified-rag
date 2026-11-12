import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.config import load_config
from rag.answer.llm_client import MockLLMClient
from rag.pdf_rag_pipeline import DEFAULT_SAMPLE_PDF, PDFRAGPipeline
from rag.answer.pipeline import build_insufficient_context_answer
from rag.retrieve import SearchResult


ML4_VYOM_PAGE_TEXT = (
    "Experiment No. 4\n"
    "Aim: To implement the CART algorithm.\n\n"
    "Theory:\n"
    "Decision Tree\n"
    "A Decision Tree is a supervised machine learning algorithm used for both classification and "
    "regression tasks.\n"
    "Gini Index\n"
    "The Gini Index measures how impure or mixed a dataset is."
)


@pytest.fixture
def ml4_vyom_pdf_path():
    """
    Creates a temporary PDF file named like the project sample document.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b"%PDF-1.4 ML4 vyom sample bytes")
        pdf_path = Path(tmp.name)

    named_path = pdf_path.parent / "ML4 vyom.pdf"
    pdf_path.rename(named_path)
    yield named_path
    if named_path.exists():
        named_path.unlink()


def _search_hit(chunk_id: str, content: str, score: float, method: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc_ml4",
        document_name="ML4 vyom.pdf",
        section="Introduction",
        page_numbers=[1],
        content=content,
        score=score,
        retrieval_method=method,
    )


@patch("rag.pdf_rag_pipeline.ElasticsearchIndexer")
@patch("rag.pdf_rag_pipeline.get_embedding_provider")
@patch("rag.pdf_rag_pipeline.Chunker")
@patch("rag.pdf_rag_pipeline.PDFProcessor")
def test_end_to_end_pipeline_with_mocks(
    mock_pdf_processor_class,
    mock_chunker_class,
    mock_embedder_factory,
    mock_indexer_class,
    ml4_vyom_pdf_path,
):
    """
    Verifies the full PDF RAG flow using offline mocks:
    PDF -> extraction -> chunks -> embeddings -> optional indexing -> retrieval -> LLM answer.
    """
    from rag.index.pdf_chunker import ChunkModel, ChunkingStatistics
    from rag.extract.pdf_processor import DocumentModel, PageModel, BlockModel

    mock_document = DocumentModel(
        document_id="doc_ml4_hash",
        document_name="ML4 vyom.pdf",
        source_path=str(ml4_vyom_pdf_path),
        total_pages=1,
        successfully_extracted_pages=1,
        failed_pages=0,
        pages_requiring_ocr=0,
        metadata={"Title": "ML4"},
        pages=[
            PageModel(
                page_number=1,
                text=ML4_VYOM_PAGE_TEXT,
                blocks=[
                    BlockModel(
                        block_id="b1",
                        page_number=1,
                        text=ML4_VYOM_PAGE_TEXT,
                        block_type="paragraph",
                        position_order=1,
                    )
                ],
                requires_ocr=False,
                extraction_status="SUCCESS",
            )
        ],
        extraction_status="SUCCESS",
        start_time=0.0,
        end_time=1.0,
        processing_duration_seconds=1.0,
    )

    sample_chunk = ChunkModel(
        chunk_id="doc_ml4_hash_chunk_001",
        document_id="doc_ml4_hash",
        document_name="ML4 vyom.pdf",
        section="Introduction",
        page_numbers=[1],
        content=ML4_VYOM_PAGE_TEXT,
        chunk_index=1,
    )
    chunk_stats = ChunkingStatistics(
        total_input_pages=1,
        total_input_blocks=1,
        total_generated_chunks=1,
        average_chunk_size=float(len(ML4_VYOM_PAGE_TEXT)),
        minimum_chunk_size=len(ML4_VYOM_PAGE_TEXT),
        maximum_chunk_size=len(ML4_VYOM_PAGE_TEXT),
    )

    mock_pdf_processor = MagicMock()
    mock_pdf_processor.process_pdf.return_value = mock_document
    mock_pdf_processor.save_extraction.return_value = Path("data/extracted/ML4 vyom.json")
    mock_pdf_processor_class.return_value = mock_pdf_processor

    mock_chunker = MagicMock()
    mock_chunker.process_document.return_value = ([sample_chunk], chunk_stats)
    mock_chunker.save_chunks.return_value = Path("data/extracted/ML4 vyom_chunks.json")
    mock_chunker_class.return_value = mock_chunker

    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [[0.1] * 384]
    mock_embedder_factory.return_value = mock_embedder

    mock_indexer = MagicMock()
    mock_indexer.ping.return_value = True
    mock_indexer.index_chunks.return_value = {
        "total_received": 1,
        "successfully_indexed": 1,
        "failed": 0,
        "duration_seconds": 0.1,
    }
    mock_indexer.client = MagicMock()
    mock_indexer_class.return_value = mock_indexer

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        _search_hit(
            "doc_ml4_hash_chunk_001",
            ML4_VYOM_PAGE_TEXT,
            0.92,
            "hybrid",
        )
    ]

    mock_llm = MagicMock()
    mock_llm.generate_completion.return_value = (
        "The CART algorithm uses decision trees and the Gini Index (ML4 vyom.pdf, page 1)."
    )

    config = load_config()
    config.retrieval_mode = "hybrid"

    from rag.answer.pipeline import RAGPipeline

    pipeline = PDFRAGPipeline(
        config=config,
        pdf_processor=mock_pdf_processor,
        chunker=mock_chunker,
        embedding_provider=mock_embedder,
        indexer=mock_indexer,
        llm_client=mock_llm,
        rag_pipeline=RAGPipeline(config=config, llm_client=mock_llm, retriever=mock_retriever),
    )

    result = pipeline.run_end_to_end(
        ml4_vyom_pdf_path,
        "What is the aim of Experiment No. 4?",
        verify_retrieval=False,
    )

    assert result.ingest.document_name == "ML4 vyom.pdf"
    assert result.ingest.extraction_status == "SUCCESS"
    assert result.ingest.chunks_generated == 1
    assert result.ingest.embeddings_generated == 1
    assert result.ingest.elasticsearch_status.available is True
    assert result.ingest.indexed is True
    assert result.answer.chunks_retrieved == 1
    assert result.answer.sources[0].document_name == "ML4 vyom.pdf"
    assert result.answer.sources[0].page_numbers == [1]
    assert "CART algorithm" in result.answer.answer

    mock_pdf_processor.process_pdf.assert_called_once()
    mock_chunker.process_document.assert_called_once()
    mock_embedder.embed_documents.assert_called_once()
    mock_indexer.index_chunks.assert_called_once()
    mock_retriever.search.assert_called_once()
    mock_llm.generate_completion.assert_called_once()


@patch("rag.pdf_rag_pipeline.ElasticsearchIndexer")
@patch("rag.pdf_rag_pipeline.get_embedding_provider")
@patch("rag.pdf_rag_pipeline.Chunker")
@patch("rag.pdf_rag_pipeline.PDFProcessor")
def test_pipeline_skips_indexing_when_elasticsearch_unavailable(
    mock_pdf_processor_class,
    mock_chunker_class,
    mock_embedder_factory,
    mock_indexer_class,
    ml4_vyom_pdf_path,
):
    from rag.index.pdf_chunker import ChunkModel, ChunkingStatistics
    from rag.extract.pdf_processor import DocumentModel, PageModel

    mock_document = DocumentModel(
        document_id="doc_ml4_hash",
        document_name="ML4 vyom.pdf",
        source_path=str(ml4_vyom_pdf_path),
        total_pages=1,
        successfully_extracted_pages=1,
        failed_pages=0,
        pages_requiring_ocr=0,
        metadata={},
        pages=[
            PageModel(
                page_number=1,
                text=ML4_VYOM_PAGE_TEXT,
                blocks=[],
                requires_ocr=False,
                extraction_status="SUCCESS",
            )
        ],
        extraction_status="SUCCESS",
        start_time=0.0,
        end_time=1.0,
        processing_duration_seconds=1.0,
    )
    sample_chunk = ChunkModel(
        chunk_id="doc_ml4_hash_chunk_001",
        document_id="doc_ml4_hash",
        document_name="ML4 vyom.pdf",
        section="Introduction",
        page_numbers=[1],
        content=ML4_VYOM_PAGE_TEXT,
        chunk_index=1,
    )

    mock_pdf_processor = MagicMock()
    mock_pdf_processor.process_pdf.return_value = mock_document
    mock_pdf_processor.save_extraction.return_value = Path("data/extracted/ML4 vyom.json")
    mock_pdf_processor_class.return_value = mock_pdf_processor

    mock_chunker = MagicMock()
    mock_chunker.process_document.return_value = (
        [sample_chunk],
        ChunkingStatistics(total_generated_chunks=1),
    )
    mock_chunker.save_chunks.return_value = Path("data/extracted/ML4 vyom_chunks.json")
    mock_chunker_class.return_value = mock_chunker

    mock_embedder = MagicMock()
    mock_embedder.embed_documents.return_value = [[0.2] * 384]
    mock_embedder_factory.return_value = mock_embedder

    mock_indexer = MagicMock()
    mock_indexer.ping.return_value = False
    mock_indexer_class.return_value = mock_indexer

    pipeline = PDFRAGPipeline(
        pdf_processor=mock_pdf_processor,
        chunker=mock_chunker,
        embedding_provider=mock_embedder,
        indexer=mock_indexer,
        llm_client=MockLLMClient(),
        rag_pipeline=MagicMock(),
    )

    ingest = pipeline.ingest_pdf(ml4_vyom_pdf_path, save_artifacts=False, index_if_available=True)

    assert ingest.elasticsearch_status.available is False
    assert ingest.indexed is False
    assert ingest.indexing_result is None
    assert "unavailable" in ingest.elasticsearch_status.message.lower()
    mock_indexer.index_chunks.assert_not_called()


def test_default_sample_pdf_path_points_to_ml4_vyom():
    assert DEFAULT_SAMPLE_PDF.name == "ML4 vyom.pdf"


def test_load_document_from_extracted_json_artifact():
    extracted_json = Path("data/extracted/ML4 vyom.json")
    if not extracted_json.exists():
        pytest.skip("Extracted ML4 vyom.json artifact is not available in this workspace.")

    document = PDFRAGPipeline.load_document_from_extracted_json(extracted_json)
    assert document.document_name == "ML4 vyom.pdf"
    assert document.total_pages == 14
    assert document.successfully_extracted_pages == 14


@patch("rag.extract.pdf_processor.PdfReader")
def test_live_pdf_processing_when_sample_pdf_exists(mock_reader_class):
    """
    Processes ML4 vyom.pdf locally when the file is present.
    """
    pdf_path = DEFAULT_SAMPLE_PDF
    if not pdf_path.exists():
        pytest.skip(f"Sample PDF not found at {pdf_path}. Offline mocked pipeline tests still apply.")

    page = MagicMock()
    page.extract_text.return_value = ML4_VYOM_PAGE_TEXT
    mock_reader = MagicMock()
    mock_reader.pages = [page]
    mock_reader.metadata = {"Title": "ML4"}
    mock_reader_class.return_value = mock_reader

    pipeline = PDFRAGPipeline()
    ingest = pipeline.ingest_pdf(pdf_path, save_artifacts=False, index_if_available=False)

    assert ingest.document_name == "ML4 vyom.pdf"
    assert ingest.chunks_generated > 0
    assert ingest.embeddings_generated == ingest.chunks_generated


def test_live_elasticsearch_integration_optional():
    """
    Attempts a live Elasticsearch ping without faking unavailable clusters.
    """
    pipeline = PDFRAGPipeline()
    status = pipeline.check_elasticsearch()

    if not status.available:
        assert "unavailable" in status.message.lower()
        pytest.skip("Elasticsearch unavailable: live indexing/retrieval not executed.")
    else:
        assert status.available is True


@patch("rag.answer.pipeline.get_retriever")
def test_end_to_end_query_without_retrieved_chunks(mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    mock_get_retriever.return_value = mock_retriever

    pipeline = PDFRAGPipeline(
        llm_client=MagicMock(),
        rag_pipeline=__import__("rag.answer.pipeline", fromlist=["RAGPipeline"]).RAGPipeline(
            llm_client=MagicMock(),
            retriever=mock_retriever,
        ),
    )

    answer = pipeline.answer_question("Unknown question about quantum physics?")
    assert answer.answer == build_insufficient_context_answer()
    assert answer.chunks_retrieved == 0


def test_discover_pdf_paths_returns_sorted_pdfs(tmp_path):
    (tmp_path / "b_doc.pdf").write_bytes(b"%PDF-1.4 b")
    (tmp_path / "a_doc.pdf").write_bytes(b"%PDF-1.4 a")
    (tmp_path / "notes.txt").write_text("not a pdf")

    from rag.pdf_rag_pipeline import discover_pdf_paths

    discovered = discover_pdf_paths(tmp_path)
    assert [path.name for path in discovered] == ["a_doc.pdf", "b_doc.pdf"]


@patch("rag.pdf_rag_pipeline.PDFRAGPipeline.ingest_pdf")
def test_ingest_all_pdfs_processes_every_discovered_pdf(mock_ingest_pdf, tmp_path):
    from rag.index.pdf_chunker import ChunkingStatistics
    from rag.config import AppConfig
    from rag.pdf_rag_pipeline import ElasticsearchStatus, IngestResult, PDFRAGPipeline

    pdf_a = tmp_path / "alpha.pdf"
    pdf_b = tmp_path / "beta.pdf"
    pdf_a.write_bytes(b"%PDF-1.4 alpha")
    pdf_b.write_bytes(b"%PDF-1.4 beta")

    es_status = ElasticsearchStatus(
        available=True,
        url="http://localhost:9200",
        index_name="test_index",
        message="ok",
    )

    def _make_result(path: Path) -> IngestResult:
        return IngestResult(
            pdf_path=str(path),
            document_id=f"id_{path.stem}",
            document_name=path.name,
            extraction_status="SUCCESS",
            total_pages=1,
            chunks_generated=2,
            embeddings_generated=2,
            chunking_stats=ChunkingStatistics(total_generated_chunks=2),
            elasticsearch_status=es_status,
            indexed=True,
        )

    mock_ingest_pdf.side_effect = lambda path, **kwargs: _make_result(Path(path))

    config = AppConfig(pdf_data_dir=str(tmp_path))
    pipeline = PDFRAGPipeline(config=config)
    batch = pipeline.ingest_all_pdfs(tmp_path)

    assert batch.total_pdfs_found == 2
    assert batch.successfully_ingested == 2
    assert batch.failed == 0
    assert mock_ingest_pdf.call_count == 2
    assert [result.document_name for result in batch.results] == ["alpha.pdf", "beta.pdf"]


@patch("rag.pdf_rag_pipeline.PDFRAGPipeline.ingest_pdf")
def test_ingest_all_pdfs_continues_after_failure(mock_ingest_pdf, tmp_path):
    from rag.index.pdf_chunker import ChunkingStatistics
    from rag.config import AppConfig
    from rag.pdf_rag_pipeline import ElasticsearchStatus, IngestResult, PDFRAGPipeline

    good_pdf = tmp_path / "good.pdf"
    bad_pdf = tmp_path / "bad.pdf"
    good_pdf.write_bytes(b"%PDF-1.4 good")
    bad_pdf.write_bytes(b"%PDF-1.4 bad")

    es_status = ElasticsearchStatus(
        available=True,
        url="http://localhost:9200",
        index_name="test_index",
        message="ok",
    )

    def _ingest_side_effect(path, **kwargs):
        path = Path(path)
        if path.name == "bad.pdf":
            raise ValueError("corrupted pdf")
        return IngestResult(
            pdf_path=str(path),
            document_id="good_id",
            document_name=path.name,
            extraction_status="SUCCESS",
            total_pages=1,
            chunks_generated=1,
            embeddings_generated=1,
            chunking_stats=ChunkingStatistics(total_generated_chunks=1),
            elasticsearch_status=es_status,
            indexed=True,
        )

    mock_ingest_pdf.side_effect = _ingest_side_effect

    config = AppConfig(pdf_data_dir=str(tmp_path))
    pipeline = PDFRAGPipeline(config=config)
    batch = pipeline.ingest_all_pdfs(tmp_path)

    assert batch.total_pdfs_found == 2
    assert batch.successfully_ingested == 1
    assert batch.failed == 1
    assert batch.failures[0].pdf_path.endswith("bad.pdf")
    assert "corrupted pdf" in batch.failures[0].error

