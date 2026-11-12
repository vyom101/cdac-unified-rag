from pathlib import Path

from rag.answer.grounding import ABSTAIN_ANSWER
from rag.answer.pipeline import answer_question, build_rag_context
from rag.config import AppConfig, load_config
from rag.index.indexer import ElasticsearchIndexer, chunk_to_indexable
from rag.models import ChunkModel
from rag.retrieve import SearchResult
from rag.retrieve.filters import source_domain_filter
from rag.retrieve.hybrid import HybridRetriever


def _result(chunk_id="one", score=0.9, **changes):
    data = dict(
        chunk_id=chunk_id, document_id="doc", document_name="manual.pdf",
        source_kind="uploaded_pdf", content="supported evidence", score=score,
        retrieval_method="vector", page_numbers=[2],
    )
    data.update(changes)
    return SearchResult(**data)


def test_unified_config_defaults_and_endpoint_provider(tmp_path, monkeypatch):
    settings = tmp_path / "settings.yaml"
    settings.write_text("retrieval:\n  top_k: 7\n", encoding="utf-8")
    monkeypatch.setenv("LLM_ENDPOINT", "https://llm.invalid/v1")
    config = load_config(settings, tmp_path / "missing.env")
    assert config.elasticsearch_index == "rag_corpus_v1"
    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.embedding_dimension == 384
    assert config.top_k == 7
    assert config.llm_provider == "custom_api"
    assert (config.html_chunk_overlap, config.pdf_chunk_overlap) == (150, 100)


def test_pdf_filter_keeps_uploaded_pdfs_outside_domain():
    clause = source_domain_filter("pdf", "cdac.gov.in")
    should = clause["bool"]["should"]
    assert should[0] == {"term": {"source_kind": "uploaded_pdf"}}
    assert {"term": {"allowed_domain": "cdac.gov.in"}} in should[1]["bool"]["filter"]
    assert clause["bool"]["minimum_should_match"] == 1


def test_unified_mapping_and_pdf_converter_preserve_metadata():
    class Indices:
        pass
    class Client:
        indices = Indices()
    config = AppConfig()
    mapping = ElasticsearchIndexer(config, client=Client())._expected_mapping()
    props = mapping["mappings"]["properties"]
    for field in ("source_kind", "content_kind", "lineage", "page_numbers",
                  "figure_pages", "table_pages", "embedding"):
        assert field in props
    record = chunk_to_indexable(ChunkModel(
        chunk_id="c", document_id="d", document_name="x.pdf", content="text",
        page_numbers=[3], metadata={"figure_refs": ["1.2"], "figure_pages": {"1.2": [3]}},
    ))
    assert record.source_kind == "uploaded_pdf"
    assert record.page_numbers == [3]
    assert record.figure_pages == {"1.2": [3]}


def test_hybrid_uses_candidate_pool_before_top_k():
    class Stub:
        def __init__(self, results):
            self.results, self.calls = results, []
        def search(self, query, **kwargs):
            self.calls.append(kwargs)
            return self.results
    bm25 = Stub([_result("a", 2.0, retrieval_method="bm25")])
    vector = Stub([_result("b", 0.8)])
    retriever = HybridRetriever(
        AppConfig(top_k=1, hybrid_candidate_pool=20), bm25_retriever=bm25,
        vector_retriever=vector,
    )
    assert len(retriever.search("query")) == 1
    assert bm25.calls[0]["top_k"] == 20
    assert vector.calls[0]["top_k"] == 20
    assert vector.calls[0]["num_candidates"] == 50


def test_grounding_abstains_without_calling_llm():
    class EmptyRetriever:
        def search(self, query, **kwargs):
            return []
    class ExplodingLLM:
        def generate_completion(self, **kwargs):
            raise AssertionError("LLM must not be called")
    answer = answer_question(
        "Who is the Director?", AppConfig(), retriever=EmptyRetriever(),
        llm_client=ExplodingLLM(),
    )
    assert answer.abstained is True
    assert answer.abstain_reason == "no_retrieval_hits"
    assert answer.answer == ABSTAIN_ANSWER


def test_context_labels_html_pdf_and_kg_provenance():
    context = build_rag_context([
        _result("html", source_kind="html", document_name="page", source_url="https://cdac.gov.in/a",
                depth=2, page_numbers=[]),
        _result("pdf", figure_refs=["1.2"], figure_pages={"1.2": [2]}),
        _result("kg", source_kind="html", retrieval_method="knowledge_graph",
                source_url="https://cdac.gov.in/provenance"),
    ])
    assert "URL: https://cdac.gov.in/a" in context and "Depth: 2" in context
    assert "Document: manual.pdf" in context and "Figure pages: {'1.2': [2]}" in context
    assert "Type: Knowledge Graph" in context
    assert "Provenance: https://cdac.gov.in/provenance" in context
