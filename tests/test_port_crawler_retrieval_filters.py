"""Tests for retrieval domain filters."""

from rag.config import AppConfig
from rag.retrieve.filters import bm25_query, knn_query, resolve_domain


def test_resolve_domain_defaults_to_crawl_domain():
    config = AppConfig(crawl_allowed_domain="cdac.gov.in", retrieval_domain_filter_enabled=True)
    assert resolve_domain(config) == "cdac.gov.in"


def test_resolve_domain_override():
    config = AppConfig(crawl_allowed_domain="cdac.gov.in")
    assert resolve_domain(config, "example.org") == "example.org"


def test_bm25_query_with_domain_filter():
    q = bm25_query("C-DAC supercomputer", "cdac.gov.in")
    assert "bool" in q
    assert q["bool"]["filter"] == [{"term": {"allowed_domain": "cdac.gov.in"}}]


def test_knn_query_with_domain_filter():
    q = knn_query([0.1, 0.2], k=5, num_candidates=20, domain="cdac.gov.in")
    assert q["filter"] == {"term": {"allowed_domain": "cdac.gov.in"}}

