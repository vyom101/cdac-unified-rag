import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from rag.answer.llm_client import BaseLLMClient, get_llm_client
from rag.answer.pipeline import RAGAnswer, RAGPipeline
from rag.config import AppConfig, load_config
from rag.extract.pdf_processor import DocumentModel, PDFProcessor
from rag.index.embedder import BaseEmbeddingProvider, get_embedding_provider
from rag.index.indexer import ElasticsearchIndexer
from rag.index.pdf_chunker import Chunker, ChunkingStatistics
from rag.retrieve.bm25 import BM25Retriever
from rag.retrieve.hybrid import HybridRetriever
from rag.retrieve.vector import VectorRetriever

logger = logging.getLogger("pdf_rag_pipeline")

DEFAULT_SAMPLE_PDF = Path("data/raw/ML4 vyom.pdf")


class ElasticsearchStatus(BaseModel):
    available: bool
    url: str
    index_name: str
    message: str


class IngestResult(BaseModel):
    pdf_path: str
    document_id: str
    document_name: str
    extraction_status: str
    total_pages: int
    chunks_generated: int
    embeddings_generated: int
    chunking_stats: ChunkingStatistics
    extraction_json_path: Optional[str] = None
    chunks_json_path: Optional[str] = None
    elasticsearch_status: ElasticsearchStatus
    indexed: bool = False
    indexing_result: Optional[Dict[str, Any]] = None


class RetrievalVerification(BaseModel):
    bm25_results: int = 0
    vector_results: int = 0
    hybrid_results: int = 0


class EndToEndResult(BaseModel):
    ingest: IngestResult
    query: str
    answer: RAGAnswer
    retrieval_verification: Optional[RetrievalVerification] = None


class BatchIngestFailure(BaseModel):
    pdf_path: str
    error: str


class BatchIngestResult(BaseModel):
    pdf_directory: str
    total_pdfs_found: int
    successfully_ingested: int
    failed: int
    results: List[IngestResult] = Field(default_factory=list)
    failures: List[BatchIngestFailure] = Field(default_factory=list)


def discover_pdf_paths(
    pdf_dir: Optional[Path] = None, config: Optional[AppConfig] = None
) -> List[Path]:
    cfg = config or load_config()
    directory = Path(pdf_dir) if pdf_dir is not None else Path(cfg.pdf_data_dir)
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.pdf") if path.is_file())


class PDFRAGPipeline:
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        pdf_processor: Optional[PDFProcessor] = None,
        chunker: Optional[Chunker] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        indexer: Optional[ElasticsearchIndexer] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self.config = config or load_config()
        self.pdf_processor = pdf_processor or PDFProcessor(self.config)
        self.chunker = chunker or Chunker(self.config)
        self.embedding_provider = embedding_provider or get_embedding_provider(self.config)
        self.llm_client = llm_client or get_llm_client(self.config)
        self.indexer = indexer or ElasticsearchIndexer(self.config)
        self.rag_pipeline = rag_pipeline or RAGPipeline(
            config=self.config,
            llm_client=self.llm_client,
        )

    def check_elasticsearch(self) -> ElasticsearchStatus:
        available = self.indexer.ping()
        message = (
            "Elasticsearch is reachable."
            if available
            else "Elasticsearch is unavailable. Indexing and live retrieval were not executed."
        )
        return ElasticsearchStatus(
            available=available,
            url=self.config.elasticsearch_url,
            index_name=self.config.elasticsearch_index,
            message=message,
        )

    def ingest_pdf(
        self,
        pdf_path: Path,
        *,
        save_artifacts: bool = True,
        index_if_available: bool = True,
        source_url: Optional[str] = None,
    ) -> IngestResult:
        pdf_path = Path(pdf_path)
        logger.info("PDFRAGPipeline: ingesting PDF '%s'", pdf_path)

        document = self.pdf_processor.process_pdf(pdf_path, source_url=source_url)
        extraction_json_path = None
        if save_artifacts:
            saved_extraction = self.pdf_processor.save_extraction(document)
            extraction_json_path = str(saved_extraction)

        chunks, chunking_stats = self.chunker.process_document(document)
        for chunk in chunks:
            chunk.source_kind = "uploaded_pdf"
            chunk.source_url = source_url
        chunks_json_path = None
        if save_artifacts and chunks:
            saved_chunks = self.chunker.save_chunks(chunks)
            chunks_json_path = str(saved_chunks)

        embeddings = self.embedding_provider.embed_documents(
            [chunk.content for chunk in chunks]
        )

        es_status = self.check_elasticsearch()
        indexed = False
        indexing_result = None

        if index_if_available and es_status.available and chunks:
            indexing_result = self.indexer.index_chunks(chunks, embeddings)
            indexed = indexing_result.get("successfully_indexed", 0) > 0
        elif index_if_available and not es_status.available:
            logger.warning(
                "PDFRAGPipeline: skipping Elasticsearch indexing because the cluster is unavailable."
            )

        return IngestResult(
            pdf_path=str(pdf_path),
            document_id=document.document_id,
            document_name=document.document_name,
            extraction_status=document.extraction_status,
            total_pages=document.total_pages,
            chunks_generated=len(chunks),
            embeddings_generated=len(embeddings),
            chunking_stats=chunking_stats,
            extraction_json_path=extraction_json_path,
            chunks_json_path=chunks_json_path,
            elasticsearch_status=es_status,
            indexed=indexed,
            indexing_result=indexing_result,
        )

    def ingest_all_pdfs(
        self,
        pdf_dir: Optional[Path] = None,
        *,
        save_artifacts: bool = True,
        index_if_available: bool = True,
    ) -> BatchIngestResult:
        directory = (
            Path(pdf_dir)
            if pdf_dir is not None
            else Path(self.config.pdf_data_dir)
        )
        pdf_paths = discover_pdf_paths(directory, self.config)
        logger.info(
            "PDFRAGPipeline: batch ingest found %s PDF(s) in '%s'",
            len(pdf_paths),
            directory,
        )

        results: List[IngestResult] = []
        failures: List[BatchIngestFailure] = []

        for pdf_path in pdf_paths:
            try:
                results.append(
                    self.ingest_pdf(
                        pdf_path,
                        save_artifacts=save_artifacts,
                        index_if_available=index_if_available,
                    )
                )
            except Exception as exc:
                logger.error("PDFRAGPipeline: failed to ingest '%s': %s", pdf_path, exc)
                failures.append(
                    BatchIngestFailure(pdf_path=str(pdf_path), error=str(exc))
                )

        return BatchIngestResult(
            pdf_directory=str(directory),
            total_pdfs_found=len(pdf_paths),
            successfully_ingested=len(results),
            failed=len(failures),
            results=results,
            failures=failures,
        )

    def answer_question(self, query: str, top_k: Optional[int] = None) -> RAGAnswer:
        return self.rag_pipeline.answer_query(query, top_k=top_k)

    def verify_retrieval_modes(
        self, query: str, top_k: Optional[int] = None
    ) -> RetrievalVerification:
        k = top_k if top_k is not None else self.config.top_k
        bm25 = BM25Retriever(self.config, client=self.indexer.client)
        vector = VectorRetriever(
            self.config,
            embedding_provider=self.embedding_provider,
            client=self.indexer.client,
        )
        hybrid = HybridRetriever(
            config=self.config,
            bm25_retriever=bm25,
            vector_retriever=vector,
        )
        return RetrievalVerification(
            bm25_results=len(bm25.search(query, top_k=k)),
            vector_results=len(
                vector.search(query, top_k=k, num_candidates=self.config.num_candidates)
            ),
            hybrid_results=len(
                hybrid.search(query, top_k=k, num_candidates=self.config.num_candidates)
            ),
        )

    def run_end_to_end(
        self,
        pdf_path: Path,
        query: str,
        *,
        save_artifacts: bool = True,
        index_if_available: bool = True,
        verify_retrieval: bool = False,
        top_k: Optional[int] = None,
    ) -> EndToEndResult:
        ingest_result = self.ingest_pdf(
            pdf_path,
            save_artifacts=save_artifacts,
            index_if_available=index_if_available,
        )
        answer = self.answer_question(query, top_k=top_k)
        retrieval_verification = None
        if (
            verify_retrieval
            and ingest_result.elasticsearch_status.available
            and ingest_result.indexed
        ):
            retrieval_verification = self.verify_retrieval_modes(query, top_k=top_k)
        return EndToEndResult(
            ingest=ingest_result,
            query=query,
            answer=answer,
            retrieval_verification=retrieval_verification,
        )

    @staticmethod
    def load_document_from_extracted_json(json_path: Path) -> DocumentModel:
        with open(json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return DocumentModel(**payload)
