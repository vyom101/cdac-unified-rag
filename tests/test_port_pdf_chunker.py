import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from rag.extract.pdf_processor import DocumentModel, PageModel, BlockModel
from rag.index.pdf_chunker import Chunker, split_text_with_overlap, ChunkModel, ChunkingStatistics
from rag.answer.llm_client import MockLLMClient, BaseLLMClient
from rag.config import load_config

# ==============================================================================
# HELPER TO CONSTRUCT SYNTHETIC DOCUMENTS FOR CHUNKING TESTS
# ==============================================================================

def create_synthetic_document(pages_data: list[list[tuple[str, str]]]) -> DocumentModel:
    """
    Constructs a valid Pydantic DocumentModel using simple layout data.
    Input structure: list of pages, where each page is a list of (block_type, text) tuples.
    """
    pages_list = []
    doc_id = "synthetic_doc_hash_123"
    
    for i, page_blocks in enumerate(pages_data):
        page_num = i + 1
        blocks_list = []
        for idx, (b_type, b_text) in enumerate(page_blocks):
            blocks_list.append(BlockModel(
                block_id=f"doc_{doc_id}_p{page_num}_b{idx+1}",
                page_number=page_num,
                text=b_text,
                block_type=b_type,
                position_order=idx + 1
            ))
            
        # Join text for page
        full_text = "\n\n".join(b_text for _, b_text in page_blocks)
        
        pages_list.append(PageModel(
            page_number=page_num,
            text=full_text,
            blocks=blocks_list,
            extraction_status="SUCCESS"
        ))
        
    return DocumentModel(
        document_id=doc_id,
        document_name="synthetic.pdf",
        source_path="data/raw/synthetic.pdf",
        total_pages=len(pages_data),
        successfully_extracted_pages=len(pages_data),
        failed_pages=0,
        pages_requiring_ocr=0,
        pages=pages_list,
        extraction_status="SUCCESS",
        start_time=100.0,
        end_time=101.0,
        processing_duration_seconds=1.0
    )


# ==============================================================================
# UNIT TESTS
# ==============================================================================

def test_split_text_with_overlap():
    """
    Validates sentence/paragraph boundary safety split and overlap size bounds.
    """
    text = "This is paragraph one.\n\nThis is paragraph two. It is very long and has details.\n\nThis is paragraph three."
    
    # 1. Split with small max size focusing on paragraph boundaries (\n\n)
    chunks = split_text_with_overlap(text, max_size=40, overlap=10)
    assert len(chunks) > 1
    assert "This is paragraph one." in chunks[0]
    
    # 2. Check no crash for short text
    assert split_text_with_overlap("Short text", 100, 10) == ["Short text"]


def test_single_page_document_chunking():
    """
    Verifies rule-based chunking on a basic, single-page document.
    """
    pages_data = [
        [
            ("heading", "1. Introduction"),
            ("paragraph", "This is the first paragraph introducing the topic."),
            ("paragraph", "This is the second paragraph under the same header.")
        ]
    ]
    doc = create_synthetic_document(pages_data)
    
    processor = Chunker()
    chunks, stats = processor.process_document(doc)
    
    assert isinstance(stats, ChunkingStatistics)
    assert stats.total_input_pages == 1
    assert stats.total_generated_chunks == 1
    
    assert len(chunks) == 1
    c1 = chunks[0]
    assert c1.chunk_id == f"doc_{doc.document_id}_chunk_001"
    assert c1.section == "1. Introduction"
    assert c1.page_numbers == [1]
    assert "introducing the topic" in c1.content


def test_multipage_document_chunking_and_provenance():
    """
    Verifies multi-page document parsing, sequential heading tracking,
    and page range provenance aggregation.
    """
    pages_data = [
        # Page 1
        [
            ("heading", "1. Section One"),
            ("paragraph", "Content on page 1.")
        ],
        # Page 2
        [
            ("paragraph", "Content on page 2 under section one.")
        ],
        # Page 3
        [
            ("heading", "2. Section Two"),
            ("paragraph", "Content on page 3 belonging to section two.")
        ]
    ]
    doc = create_synthetic_document(pages_data)
    
    processor = Chunker()
    chunks, stats = processor.process_document(doc)
    
    # We expect 2 chunks (one for Section One spanning pages 1-2, one for Section Two on page 3)
    assert stats.total_generated_chunks == 2
    assert stats.chunks_spanning_multiple_pages == 1
    
    # Section One check
    c1 = chunks[0]
    assert c1.section == "1. Section One"
    assert c1.page_numbers == [1, 2]  # correctly aggregates pages 1 and 2!
    assert "Content on page 2" in c1.content
    
    # Section Two check
    c2 = chunks[1]
    assert c2.section == "2. Section Two"
    assert c2.page_numbers == [3]


def test_deterministic_chunk_ids():
    """
    Confirms that chunking identical documents with identical setups produces matching chunk IDs.
    """
    pages_data = [
        [
            ("heading", "Title"),
            ("paragraph", "Sample paragraph text.")
        ]
    ]
    doc_1 = create_synthetic_document(pages_data)
    doc_2 = create_synthetic_document(pages_data)
    
    processor = Chunker()
    chunks_1, _ = processor.process_document(doc_1)
    chunks_2, _ = processor.process_document(doc_2)
    
    assert chunks_1[0].chunk_id == chunks_2[0].chunk_id
    assert chunks_1[0].chunk_id == f"doc_{doc_1.document_id}_chunk_001"


def test_size_boundaries_splitting_and_overlap():
    """
    Validates that adding paragraph blocks exceeding MAX_CHUNK_SIZE
    triggers split boundaries cleanly with correct character overlap.
    """
    pages_data = [
        [
            ("paragraph", "A" * 500),
            ("paragraph", "B" * 600)  # Total 1100 exceeds MAX_CHUNK_SIZE (800)
        ]
    ]
    doc = create_synthetic_document(pages_data)
    
    # Load custom tight bounds
    config = load_config()
    config.max_chunk_size = 800
    config.chunk_overlap = 100
    
    processor = Chunker(config=config)
    chunks, _ = processor.process_document(doc)
    
    assert len(chunks) == 2
    assert len(chunks[0].content) <= 800
    assert len(chunks[1].content) <= 800


def test_empty_blocks_ignored():
    """
    Confirms that empty or blank paragraphs do not result in redundant or empty chunks.
    """
    pages_data = [
        [
            ("paragraph", "Valid block with plenty of characters to extract cleanly."),
            ("paragraph", "   "),  # Empty whitespace
            ("paragraph", "")      # Empty string
        ]
    ]
    doc = create_synthetic_document(pages_data)
    
    processor = Chunker()
    chunks, _ = processor.process_document(doc)
    
    assert len(chunks) == 1
    assert "Valid block" in chunks[0].content


def test_duplicate_chunk_removal():
    """
    Verifies that identical duplicated text blocks are safely filtered
    under conservative deduplication rules.
    """
    pages_data = [
        [
            ("heading", "Section One"),
            ("paragraph", "This is a very long, unique text segment that has plenty of characters to be useful."),
            ("heading", "Section Two"),
            ("paragraph", "This is a very long, unique text segment that has plenty of characters to be useful.")  # Exact duplicate
        ]
    ]
    doc = create_synthetic_document(pages_data)
    
    processor = Chunker()
    chunks, stats = processor.process_document(doc)
    
    assert len(chunks) == 1
    assert stats.duplicate_chunks_removed == 1


def test_chunk_validation_and_rejected_references():
    """
    Checks that invalid page references or empty contents are correctly flagged
    and rejected from final indexing outputs.
    """
    processor = Chunker()
    
    # Create invalid page chunk
    invalid_chunk = ChunkModel(
        chunk_id="doc_123_chunk_001",
        document_id="doc_123",
        document_name="test.pdf",
        section="Title",
        page_numbers=[99],  # Hallucinated page number (not in source pages)
        content="This is some valid text.",
        chunk_index=1
    )
    
    source_pages = {1, 2}
    existing_hashes = set()
    
    rejection = processor._validate_chunk(invalid_chunk, source_pages, existing_hashes)
    assert rejection is not None
    assert "Hallucinated page reference" in rejection


# ==============================================================================
# OFFLINE MOCK LLM AND BATCH CHUNKING TESTS
# ==============================================================================

def test_mock_llm_assisted_chunking():
    """
    Verifies that enabling use_llm_assisted_organization executes the LLM Client pathway
    cleanly offline using MockLLMClient structures, and yields correctly formatted chunks.
    """
    pages_data = [
        [
            ("heading", "HNSW Algorithm"),
            ("paragraph", "First piece of HNSW details on page 1.")
        ],
        [
            ("paragraph", "Second piece of HNSW details on page 2.")
        ]
    ]
    doc = create_synthetic_document(pages_data)
    
    # Configure LLM Client
    config = load_config()
    config.use_llm_assisted_organization = True
    config.llm_provider = "mock"
    
    mock_llm = MockLLMClient()
    processor = Chunker(config=config, llm_client=mock_llm)
    chunks, stats = processor.process_document(doc)
    
    # Mock client returns structured HNSW blocks
    assert stats.llm_calls_made == 1
    assert stats.total_generated_chunks == 2
    assert chunks[0].section == "HNSW Algorithm"
    assert chunks[0].page_numbers == [1, 2]  # Aggregated correctly


def test_invalid_llm_structure_fallback_recovery():
    """
    Ensures that if the LLM client returns corrupted JSON, invalid formats, or crashes,
    the pipeline gracefully intercepts the error, logs it, and falls back
    to robust structural rule-based chunking for that batch, preventing document crashes.
    """
    pages_data = [
        [
            ("heading", "Fallback Test"),
            ("paragraph", "This paragraph must survive and be chunked structurally if the LLM crashes.")
        ]
    ]
    doc = create_synthetic_document(pages_data)
    
    config = load_config()
    config.use_llm_assisted_organization = True
    
    # Create a crashing LLM mock
    broken_llm = MagicMock(spec=BaseLLMClient)
    broken_llm.generate_structured_completion.side_effect = ValueError("Mangled LLM response json")
    
    processor = Chunker(config=config, llm_client=broken_llm)
    
    # Ensure it processes successfully without throwing, falling back behind the scenes!
    chunks, stats = processor.process_document(doc)
    
    assert len(chunks) == 1
    assert chunks[0].section == "Fallback Test"
    assert "This paragraph must survive" in chunks[0].content


# ==============================================================================
# INTEGRATION TESTING OF LOCAL EXTRACTION AND DEBUG REPS
# ==============================================================================

def test_json_saving_and_debug_printing():
    """
    Validates local chunk saving to JSON and verifies human-readable representation strings.
    """
    pages_data = [
        [
            ("heading", "Intro"),
            ("paragraph", "This is a long content segment with plenty of characters to be valid.")
        ]
    ]
    doc = create_synthetic_document(pages_data)
    processor = Chunker()
    chunks, _ = processor.process_document(doc)
    
    # Test debug printing
    debug_str = processor.get_debug_representation(chunks)
    assert "CHUNK 001" in debug_str
    assert "Document : synthetic.pdf" in debug_str
    assert "Section  : Intro" in debug_str
    assert "This is a long content segment" in debug_str

    # Test local JSON file saving
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "synthetic_chunks.json"
        saved_path = processor.save_chunks(chunks, out_path)
        
        assert saved_path.exists()
        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["chunk_id"] == chunks[0].chunk_id


def test_llm_batch_size_usage():
    """
    Verifies that the Chunker correctly respects the llm_batch_size configuration
    by grouping the specified number of pages into each LLM request.
    """
    # Create a synthetic document with 4 pages
    pages_data = [
        [("paragraph", f"Content of page {i+1}")] for i in range(4)
    ]
    doc = create_synthetic_document(pages_data)
    
    # Setup base config
    base_config = load_config()
    base_config.use_llm_assisted_organization = True
    
    # --- SCENARIO 1: llm_batch_size = 1 ---
    base_config.llm_batch_size = 1
    mock_llm_1 = MagicMock(spec=BaseLLMClient)
    # Return empty chunks to avoid validation errors, but we care about the call count/content
    mock_llm_1.generate_structured_completion.return_value = {"chunks": []}
    
    processor_1 = Chunker(config=base_config, llm_client=mock_llm_1)
    processor_1.process_document(doc)
    
    # With 4 pages and batch_size=1, we expect exactly 4 calls
    assert mock_llm_1.generate_structured_completion.call_count == 4
    
    # Check that each call contained exactly one page's content
    for i in range(4):
        call_args = mock_llm_1.generate_structured_completion.call_args_list[i]
        user_prompt = call_args[0][1] # second positional arg is user_prompt
        assert f"Content of page {i+1}" in user_prompt
        # Ensure next page is NOT in this batch
        if i < 3:
            assert f"Content of page {i+2}" not in user_prompt

    # --- SCENARIO 2: llm_batch_size = 2 ---
    base_config.llm_batch_size = 2
    mock_llm_2 = MagicMock(spec=BaseLLMClient)
    mock_llm_2.generate_structured_completion.return_value = {"chunks": []}
    
    processor_2 = Chunker(config=base_config, llm_client=mock_llm_2)
    processor_2.process_document(doc)
    
    # With 4 pages and batch_size=2, we expect exactly 2 calls
    assert mock_llm_2.generate_structured_completion.call_count == 2
    
    # First batch should have pages 1 and 2
    prompt_1 = mock_llm_2.generate_structured_completion.call_args_list[0][0][1]
    assert "Content of page 1" in prompt_1
    assert "Content of page 2" in prompt_1
    assert "Content of page 3" not in prompt_1
    
    # Second batch should have pages 3 and 4
    prompt_2 = mock_llm_2.generate_structured_completion.call_args_list[1][0][1]
    assert "Content of page 3" in prompt_2
    assert "Content of page 4" in prompt_2
    assert "Content of page 1" not in prompt_2


# ==============================================================================
# SCATTERED CONTENT INTEGRATION TEST (REQUIREMENT 25)
# ==============================================================================

def test_scattered_content_grouping_and_isolation():
    """
    REQUIREMENT 25:
    Page 1:
      Heading: 'HNSW'
      Paragraph: definition of HNSW.
    Page 2:
      Paragraph: explanation of HNSW search (should inherit 'HNSW' section and merge with above).
    Page 3:
      Heading: 'Other Algorithm'
      Paragraph: unrelated content (must remain separate under 'Other Algorithm').
      
    Verifies that:
    1. 'HNSW'-related content is consolidated together, tracking pages 1-2.
    2. Unrelated 'Other Algorithm' details on Page 3 are cleanly isolated, and NOT merged.
    3. Page provenance tracks perfectly.
    """
    pages_data = [
        # Page 1
        [
            ("heading", "HNSW"),
            ("paragraph", "HNSW is a graph-based algorithm for vector search.")
        ],
        # Page 2
        [
            ("paragraph", "Search in HNSW proceeds by walking layered proximity graphs.")
        ],
        # Page 3
        [
            ("heading", "Other Algorithm"),
            ("paragraph", "LSH partitions vectors using random projection hyperplanes.")
        ]
    ]
    doc = create_synthetic_document(pages_data)
    
    processor = Chunker()
    chunks, stats = processor.process_document(doc)
    
    # Assertions
    assert stats.total_generated_chunks == 2
    
    # Chunk 1 (HNSW Definition + Explanation)
    c1 = chunks[0]
    assert c1.section == "HNSW"
    assert c1.page_numbers == [1, 2]  # Preserves multi-page provenance spanning 1 and 2
    assert "vector search" in c1.content
    assert "proximity graphs" in c1.content  # Merged successfully!
    
    # Chunk 2 (Other Algorithm)
    c2 = chunks[1]
    assert c2.section == "Other Algorithm"
    assert c2.page_numbers == [3]  # Isolated cleanly to page 3
    assert "LSH partitions" in c2.content
    assert "vector search" not in c2.content  # Complete semantic isolation!

