"""Evaluation metrics for retrieval comparison."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set


@dataclass
class QueryJudgment:
    query: str
    expected_document_ids: Set[str]
    query_type: str = "factual"
    unanswerable: bool = False


def hit_at_k(retrieved_ids: List[str], expected: Set[str], k: int) -> int:
    return int(any(document_id in expected for document_id in retrieved_ids[:k]))


def reciprocal_rank(retrieved_ids: List[str], expected: Set[str]) -> float:
    for index, document_id in enumerate(retrieved_ids, start=1):
        if document_id in expected:
            return 1.0 / index
    return 0.0


def recall_at_k(retrieved_ids: List[str], expected: Set[str], k: int) -> float:
    return len(set(retrieved_ids[:k]) & expected) / len(expected) if expected else 0.0


def summarize_metrics(results: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(results)
    if not rows:
        return {}
    return {
        key: round(sum(row[key] for row in rows) / len(rows), 4)
        for key in rows[0]
    }
