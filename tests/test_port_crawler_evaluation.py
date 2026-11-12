"""Tests for multi-mode evaluation runner."""

import json
from pathlib import Path

from rag.config import AppConfig
from rag.eval.runner import load_benchmark, run_evaluation


def test_load_benchmark(tmp_path):
    bench = tmp_path / "bench.jsonl"
    bench.write_text(
        json.dumps({"query": "test", "expected_document_ids": ["a"]}) + "\n",
        encoding="utf-8",
    )
    cases = load_benchmark(bench)
    assert len(cases) == 1
    assert cases[0].query == "test"


def test_run_evaluation_structure(tmp_path):
    bench = tmp_path / "bench.jsonl"
    bench.write_text(
        json.dumps({"query": "C-DAC", "expected_document_ids": ["x"], "query_type": "factual"}) + "\n",
        encoding="utf-8",
    )
    config = AppConfig(
        hybrid_eval_keyword_weight=0.3,
        hybrid_eval_semantic_weight=0.7,
        kg_store_path=str(tmp_path / "missing_graph.json"),
    )
    report = run_evaluation(config, bench, top_k=5, domain="cdac.gov.in")
    assert report["top_k"] == 5
    assert report["domain_filter"] == "cdac.gov.in"
    assert "bm25_only" in report
    assert "vector_only" in report
    assert "hybrid_0.3_0.7" in report
    assert "knowledge_graph" in report
    assert "summary" in report["bm25_only"]


def test_hit_at_k_metrics():
    from rag.eval.metrics import hit_at_k, reciprocal_rank

    retrieved = ["a", "b", "c"]
    assert hit_at_k(retrieved, {"b"}, 5) == 1
    assert reciprocal_rank(["x", "target", "y"], {"target"}) == 0.5

