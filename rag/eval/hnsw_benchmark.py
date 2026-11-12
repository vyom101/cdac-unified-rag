"""Brute-force versus Elasticsearch HNSW performance benchmark."""

from __future__ import annotations

import time
from typing import Any, Dict, Sequence

import numpy as np
from elasticsearch.helpers import bulk

from rag.config import AppConfig


def generate_unit_vectors(count: int, dimension: int, seed: int = 42) -> np.ndarray:
    vectors = np.random.default_rng(seed).standard_normal((count, dimension)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def brute_force_top_k(
    query: np.ndarray, matrix: np.ndarray, top_k: int
) -> tuple[list[int], float]:
    start = time.monotonic()
    scores = matrix @ query
    k = min(top_k, len(scores))
    if k == 0:
        return [], 0.0
    partition = np.argpartition(-scores, k - 1)[:k]
    ranked = sorted(partition.tolist(), key=lambda index: -scores[index])
    return ranked, round((time.monotonic() - start) * 1000, 3)


def _client(config: AppConfig):
    from elasticsearch import Elasticsearch

    auth = None
    if config.elasticsearch_username and config.elasticsearch_password:
        auth = (
            config.elasticsearch_username.get_secret_value(),
            config.elasticsearch_password.get_secret_value(),
        )
    return Elasticsearch(config.elasticsearch_url, basic_auth=auth)


def _index_synthetic_vectors(config: AppConfig, vectors: np.ndarray) -> tuple[str, float]:
    client = _client(config)
    index_name = f"{config.elasticsearch_index}_hnsw_bench"
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    client.indices.create(index=index_name, body={"mappings": {"properties": {
        "doc_id": {"type": "keyword"},
        "embedding": {
            "type": "dense_vector", "dims": config.embedding_dimension,
            "index": True, "similarity": "cosine",
        },
    }}})
    start = time.monotonic()
    bulk(client, [{
        "_index": index_name, "_id": str(index),
        "_source": {"doc_id": f"synthetic_{index}", "embedding": vector.tolist()},
    } for index, vector in enumerate(vectors)], refresh=True)
    return index_name, round((time.monotonic() - start) * 1000, 3)


def _hnsw_top_k(
    config: AppConfig, index_name: str, query_vector: np.ndarray,
    top_k: int, num_candidates: int,
) -> tuple[list[str], float]:
    client = _client(config)
    start = time.monotonic()
    response = client.search(index=index_name, body={
        "size": top_k,
        "knn": {
            "field": "embedding", "query_vector": query_vector.tolist(),
            "k": top_k, "num_candidates": max(num_candidates, top_k),
        },
    })
    return (
        [hit["_source"]["doc_id"] for hit in response["hits"]["hits"]],
        round((time.monotonic() - start) * 1000, 3),
    )


def run_hnsw_benchmark(
    config: AppConfig, *, vector_counts: Sequence[int] | None = None,
    top_k: int = 5, num_candidates: int | None = None,
    include_elasticsearch: bool = True,
) -> Dict[str, Any]:
    counts = list(vector_counts or config.hnsw_benchmark_sizes)
    dimension = config.embedding_dimension
    candidates = num_candidates or config.num_candidates
    query = generate_unit_vectors(1, dimension, seed=7)[0]
    rows = []
    elasticsearch_available = include_elasticsearch
    for count in counts:
        matrix = generate_unit_vectors(count, dimension, seed=count)
        identifiers, brute_force_ms = brute_force_top_k(query, matrix, top_k)
        row: dict[str, Any] = {
            "vector_count": count, "dimension": dimension, "top_k": top_k,
            "brute_force_ms": brute_force_ms,
            "brute_force_ids": [f"synthetic_{index}" for index in identifiers],
        }
        if elasticsearch_available:
            try:
                index_name, index_ms = _index_synthetic_vectors(config, matrix)
                hnsw_ids, hnsw_ms = _hnsw_top_k(config, index_name, query, top_k, candidates)
                row.update({
                    "index_build_ms": index_ms, "hnsw_ms": hnsw_ms,
                    "hnsw_ids": hnsw_ids,
                    "top_k_overlap_with_brute_force": len(set(row["brute_force_ids"]) & set(hnsw_ids)),
                    "speedup_vs_brute_force": round(brute_force_ms / hnsw_ms, 3) if hnsw_ms else None,
                })
            except Exception as error:
                row["elasticsearch_error"] = str(error)
                elasticsearch_available = False
        rows.append(row)
    return {
        "benchmark_type": "brute_force_vs_hnsw_performance",
        "note": "Engineering latency benchmark only; separate from C-DAC retrieval quality evaluation.",
        "embedding_dimension": dimension, "top_k": top_k,
        "vector_counts": counts, "rows": rows,
    }
