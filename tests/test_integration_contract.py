from pathlib import Path

import pytest

from rag.answer.pipeline import answer_question, build_rag_context
from rag.config import AppConfig
from rag.index.indexer import ElasticsearchIndexer
from rag.ingestion import load_index_records
from rag.models import ChunkModel, PageContentArtifact
from rag.retrieve import SearchResult


def _result(content="grounded text", **changes):
    values = {
        "chunk_id": "c1", "document_id": "d1", "document_name": "source",
        "content": content, "score": 0.9, "retrieval_method": "bm25",
    }
    values.update(changes)
    return SearchResult(**values)


class Retriever:
    def __init__(self, results):
        self.results = results

    def search(self, query, **kwargs):
        return self.results


class LLM:
    def __init__(self):
        self.calls = []

    def generate_completion(self, **kwargs):
        self.calls.append(kwargs)
        return "grounded answer"


def test_index_mapping_and_exact_kb_source_loading(tmp_path):
    class Client:
        class Indices:
            pass
        indices = Indices()

    props = ElasticsearchIndexer(
        AppConfig(), client=Client()
    )._expected_mapping()["mappings"]["properties"]
    assert {"source_kind", "lineage", "page_numbers", "embedding"} <= set(props)

    for kind in ("html", "crawled_pdf", "uploaded_pdf"):
        artifact = PageContentArtifact(
            page_id=kind, url=f"https://example/{kind}", title=kind,
            source_type=kind,
            chunks=[ChunkModel(
                chunk_id=kind, document_id=kind, document_name=kind,
                content=f"{kind} content", source_kind=kind,
            )],
        )
        (tmp_path / f"{kind}.json").write_text(
            artifact.model_dump_json(), encoding="utf-8"
        )
    expected = {
        "html_only": {"html"},
        "pdf_only": {"crawled_pdf", "uploaded_pdf"},
        "html_and_pdf": {"html", "crawled_pdf", "uploaded_pdf"},
        "uploaded_pdf_only": {"uploaded_pdf"},
        "all": {"html", "crawled_pdf", "uploaded_pdf"},
    }
    for source, kinds in expected.items():
        config = AppConfig(extracted_data_dir=str(tmp_path), kb_source=source)
        assert {record.source_kind for record in load_index_records(config)} == kinds


def test_no_hits_abstains_without_llm():
    llm = LLM()
    answer = answer_question(
        "question", AppConfig(), retriever=Retriever([]), llm_client=llm
    )
    assert answer.abstained and llm.calls == []


def test_above_threshold_calls_llm_with_retrieved_text():
    llm = LLM()
    answer = answer_question(
        "question", AppConfig(), retriever=Retriever([_result("unique evidence")]),
        llm_client=llm,
    )
    assert not answer.abstained
    assert "unique evidence" in llm.calls[0]["user_prompt"]


def test_context_contains_pdf_references_and_html_url():
    context = build_rag_context([
        _result(
            source_kind="html", source_url="https://cdac.gov.in/page",
            page_numbers=[],
        ),
        _result(
            chunk_id="pdf", source_kind="uploaded_pdf", document_name="manual.pdf",
            page_numbers=[7], figure_refs=["1.5"], figure_pages={"1.5": [7]},
        ),
    ])
    assert "URL: https://cdac.gov.in/page" in context
    assert "Document: manual.pdf" in context
    assert "Figure pages: {'1.5': [7]}" in context


def test_include_kg_false_does_not_search_kg():
    class KG:
        def search(self, *args, **kwargs):
            raise AssertionError("KG must not be searched")

    llm = LLM()
    answer = answer_question(
        "question", AppConfig(), retriever=Retriever([_result()]),
        llm_client=llm, kg=KG(), include_kg=False,
    )
    assert answer.include_kg is False
    assert "Knowledge-graph evidence" not in llm.calls[0]["user_prompt"]


def test_include_kg_true_searches_kg():
    class KG:
        called = False

        def search(self, *args, **kwargs):
            self.called = True
            return [_result(
                chunk_id="kg", retrieval_method="knowledge_graph",
                source_url="kg-provenance",
            )]

    kg = KG()
    llm = LLM()
    answer = answer_question(
        "question", AppConfig(), retriever=Retriever([_result()]),
        llm_client=llm, kg=kg, include_kg=True,
    )
    assert kg.called and answer.include_kg is True
    assert "kg-provenance" in llm.calls[0]["user_prompt"]
    assert "Knowledge-graph evidence" in llm.calls[0]["user_prompt"]


def test_role_mismatch_abstains_before_llm():
    llm = LLM()
    answer = answer_question(
        "Who is the Director?", AppConfig(),
        retriever=Retriever([_result("The Executive Director leads the centre.")]),
        llm_client=llm,
    )
    assert answer.abstained
    assert answer.abstain_reason == "role_mismatch"
    assert llm.calls == []
