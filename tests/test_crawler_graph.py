from rag.graph.extractor import extract_from_chunk, normalize_name
from rag.graph.resolver import EntityResolver
from rag.graph.store import KnowledgeGraphStore
from rag.models import ChunkModel


def _chunk():
    return ChunkModel(
        chunk_id="doc_x_chunk_000", document_id="x",
        document_name="About C-DAC",
        content="C-DAC operates from Pune, India. PARAM Siddhi-AI is a supercomputer.",
        source_url="https://cdac.gov.in/about", chunk_index=0,
    )


def test_extract_creates_organization_and_location(tmp_path):
    entities, edges = extract_from_chunk(_chunk())
    types = {entity.entity_type for entity in entities}
    assert "Organization" in types
    assert "Location" in types
    assert edges


def test_resolver_merges_aliases():
    resolver = EntityResolver()
    assert normalize_name("C-DAC") == normalize_name("cdac")


def test_kg_search_returns_results(tmp_path):
    store = KnowledgeGraphStore(str(tmp_path / "graph.json"))
    store.ingest_chunks([_chunk()])
    results = store.search("C-DAC Pune", top_k=3)
    assert results
    assert results[0].retrieval_method == "knowledge_graph"
