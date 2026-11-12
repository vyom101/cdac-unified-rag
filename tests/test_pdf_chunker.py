from rag.answer.llm_client import MockLLMClient
from rag.config import load_config
from rag.index.pdf_chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_MIN_CHUNK_SIZE,
    Chunker,
    split_text_with_overlap,
)
from rag.models import BlockModel, DocumentModel, PageModel


def make_document(pages_data):
    pages = []
    document_id = "synthetic_doc_hash_123"
    for page_index, block_data in enumerate(pages_data, start=1):
        blocks = [
            BlockModel(
                block_id=f"doc_{document_id}_p{page_index}_b{block_index}",
                page_number=page_index,
                text=text,
                block_type=block_type,
                position_order=block_index,
            )
            for block_index, (block_type, text) in enumerate(block_data, start=1)
        ]
        pages.append(PageModel(
            page_number=page_index,
            text="\n\n".join(text for _, text in block_data),
            blocks=blocks,
            extraction_status="SUCCESS",
        ))
    return DocumentModel(
        document_id=document_id,
        document_name="synthetic.pdf",
        source_path="data/raw/synthetic.pdf",
        total_pages=len(pages),
        successfully_extracted_pages=len(pages),
        failed_pages=0,
        pages_requiring_ocr=0,
        pages=pages,
        extraction_status="SUCCESS",
        start_time=100.0,
        end_time=101.0,
        processing_duration_seconds=1.0,
    )


def config_without_minimum_merge():
    config = load_config()
    config.min_chunk_size = 0
    return config


def test_pdf_default_chunking_contract():
    assert DEFAULT_MAX_CHUNK_SIZE == 1200
    assert DEFAULT_CHUNK_OVERLAP == 100
    assert DEFAULT_MIN_CHUNK_SIZE == 300


def test_split_text_with_overlap():
    text = (
        "This is paragraph one.\n\nThis is paragraph two. It is very long.\n\n"
        "This is paragraph three."
    )
    chunks = split_text_with_overlap(text, max_size=40, overlap=10)
    assert len(chunks) > 1
    assert "This is paragraph one." in chunks[0]
    assert split_text_with_overlap("Short text", 100, 10) == ["Short text"]


def test_multipage_chunking_and_provenance():
    document = make_document([
        [("heading", "1. Section One"), ("paragraph", "Content on page 1.")],
        [("paragraph", "Content on page 2 under section one.")],
        [("heading", "2. Section Two"), ("paragraph", "Content on page 3.")],
    ])
    chunks, statistics = Chunker(config_without_minimum_merge()).process_document(
        document
    )
    assert statistics.total_generated_chunks == 2
    assert statistics.chunks_spanning_multiple_pages == 1
    assert chunks[0].section == "1. Section One"
    assert chunks[0].page_numbers == [1, 2]
    assert chunks[1].page_numbers == [3]


def test_exact_and_near_duplicate_removal():
    document = make_document([[
        ("heading", "Section One"),
        ("paragraph", "This is a very long unique text segment with useful content."),
        ("heading", "Section Two"),
        ("paragraph", "This is a very long unique text segment with useful content."),
    ]])
    chunks, statistics = Chunker(config_without_minimum_merge()).process_document(
        document
    )
    assert len(chunks) == 1
    assert statistics.duplicate_chunks_removed == 1


def test_minimum_size_merges_low_information_chunks():
    config = load_config()
    config.min_chunk_size = 300
    document = make_document([[
        ("heading", "One"),
        ("paragraph", "First short paragraph with enough characters to validate."),
        ("heading", "One"),
        ("paragraph", "Second short paragraph with enough characters to validate."),
    ]])
    chunks, statistics = Chunker(config).process_document(document)
    assert len(chunks) == 1
    assert statistics.merged_low_information_chunks == 1
    assert "First short paragraph" in chunks[0].content
    assert "Second short paragraph" in chunks[0].content


def test_mock_llm_assisted_chunking():
    config = config_without_minimum_merge()
    config.use_llm_assisted_organization = True
    config.llm_provider = "mock"
    document = make_document([
        [("heading", "HNSW Algorithm"), ("paragraph", "HNSW details on page 1.")],
        [("paragraph", "HNSW details on page 2.")],
    ])
    chunks, statistics = Chunker(config, MockLLMClient()).process_document(document)
    assert statistics.llm_calls_made == 1
    assert statistics.total_generated_chunks == 2
    assert chunks[0].section == "HNSW Algorithm"
    assert chunks[0].page_numbers == [1, 2]
