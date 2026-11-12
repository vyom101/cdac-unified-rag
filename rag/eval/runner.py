"""Independent BM25, vector, hybrid, and KG retrieval evaluation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from rag.config import AppConfig
from rag.eval.metrics import hit_at_k, recall_at_k, reciprocal_rank, summarize_metrics
from rag.graph.store import KnowledgeGraphStore
from rag.retrieve import SearchResult
from rag.retrieve.bm25 import BM25Retriever
from rag.retrieve.hybrid import HybridRetriever
from rag.retrieve.vector import VectorRetriever


@dataclass
class BenchmarkCase:
    query: str
    expected_document_ids: List[str]
    query_type: str = "factual"
    unanswerable: bool = False


def load_benchmark(path: str | Path) -> List[BenchmarkCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cases.append(BenchmarkCase(
                query=row["query"],
                expected_document_ids=row.get("expected_document_ids", []),
                query_type=row.get("query_type", "factual"),
                unanswerable=bool(row.get("unanswerable", False)),
            ))
    return cases


def _evaluate_retriever(
    name: str, cases: List[BenchmarkCase],
    search_fn: Callable[[str, int], List[SearchResult]], top_k: int,
) -> Dict[str, Any]:
    rows = []
    details = []
    for case in cases:
        expected = set(case.expected_document_ids)
        start = time.monotonic()
        results = search_fn(case.query, top_k)
        latency = round(time.monotonic() - start, 4)
        identifiers = [result.document_id for result in results]
        hit5 = (1 if not identifiers else 0) if case.unanswerable else hit_at_k(
            identifiers, expected, top_k
        )
        row = {
            "hit@1": hit_at_k(identifiers, expected, 1),
            "hit@5": hit5,
            "mrr": reciprocal_rank(identifiers, expected),
            "recall@5": recall_at_k(identifiers, expected, top_k),
            "latency": latency,
        }
        rows.append(row)
        details.append({
            "query": case.query, "query_type": case.query_type,
            "expected_document_ids": case.expected_document_ids,
            "retrieved_ids": identifiers, "method": name, **row,
        })
    summary = summarize_metrics(rows)
    summary["success_count_pct"] = round(100.0 * summary.get("hit@5", 0.0), 2)
    return {"summary": summary, "details": details}


def run_evaluation(
    config: AppConfig, benchmark_path: str | Path, top_k: int | None = None,
    *, domain: str | None = None,
) -> Dict[str, Any]:
    k = top_k or config.top_k
    cases = load_benchmark(benchmark_path)
    bm25 = BM25Retriever(config)
    vector = VectorRetriever(config)
    hybrid = HybridRetriever(config.model_copy(update={
        "keyword_weight": config.hybrid_eval_keyword_weight,
        "semantic_weight": config.hybrid_eval_semantic_weight,
        "fusion_strategy": "min_max",
    }))
    kg = KnowledgeGraphStore(config.kg_store_path)
    active_domain = domain or (
        config.crawl_allowed_domain if config.retrieval_domain_filter_enabled else None
    )
    return {
        "benchmark_cases": len(cases), "top_k": k, "domain_filter": active_domain,
        "embedding_provider": config.embedding_provider,
        "note": "Use EMBEDDING_PROVIDER=local for retrieval-quality evaluation with real embeddings.",
        "bm25_only": _evaluate_retriever(
            "bm25", cases, lambda query, limit: bm25.search(query, top_k=limit, domain=active_domain), k
        ),
        "vector_only": _evaluate_retriever(
            "vector", cases, lambda query, limit: vector.search(query, top_k=limit, domain=active_domain), k
        ),
        "hybrid_0.3_0.7": _evaluate_retriever(
            "hybrid_0.3_0.7", cases,
            lambda query, limit: hybrid.search(query, top_k=limit, domain=active_domain), k,
        ),
        "knowledge_graph": _evaluate_retriever(
            "knowledge_graph", cases, lambda query, limit: kg.search(query, top_k=limit), k
        ),
    }
