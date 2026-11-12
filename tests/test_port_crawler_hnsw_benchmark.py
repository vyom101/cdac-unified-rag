"""Tests for HNSW performance benchmark (offline brute-force path)."""

from rag.config import AppConfig
from rag.eval.hnsw_benchmark import brute_force_top_k, generate_unit_vectors, run_hnsw_benchmark


def test_brute_force_top_k_returns_k_results():
    matrix = generate_unit_vectors(100, 16, seed=1)
    query = generate_unit_vectors(1, 16, seed=2)[0]
    ids, elapsed = brute_force_top_k(query, matrix, top_k=5)
    assert len(ids) == 5
    assert elapsed >= 0


def test_hnsw_benchmark_offline_only():
    config = AppConfig(embedding_dimension=32, hnsw_benchmark_sizes=[200, 500])
    report = run_hnsw_benchmark(config, vector_counts=[200, 500], top_k=5, include_elasticsearch=False)
    assert report["benchmark_type"] == "brute_force_vs_hnsw_performance"
    assert len(report["rows"]) == 2
    assert "brute_force_ms" in report["rows"][0]
    assert "hnsw_ms" not in report["rows"][0]

