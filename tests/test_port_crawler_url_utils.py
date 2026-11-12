"""Tests for URL canonicalization and domain checks."""

import pytest

from rag.crawl.url_utils import (
    canonicalize_url,
    is_private_or_local_host,
    is_same_domain,
    is_soft_error_page,
    url_length_exceeded,
)


def test_canonicalize_removes_fragment_and_tracking():
    url = "https://www.cdac.gov.in/about?id=hpc&utm_source=test#section"
    canonical = canonicalize_url(url)
    assert canonical == "https://cdac.gov.in/about?id=hpc"
    assert "#" not in canonical
    assert "utm_source" not in canonical


def test_canonicalize_preserves_meaningful_query_keys():
    url = "https://cdac.gov.in/index.aspx?id=ai&print=1"
    canonical = canonicalize_url(url)
    assert "id=ai" in canonical
    assert "print=1" in canonical


def test_canonicalize_normalizes_trailing_slash():
    assert canonicalize_url("https://cdac.gov.in/about/") == "https://cdac.gov.in/about"


def test_same_domain_boundary():
    assert is_same_domain("https://www.cdac.gov.in/page", "cdac.gov.in")
    assert not is_same_domain("https://evilcdac.gov.in.attacker.com/", "cdac.gov.in")


def test_private_host_detection():
    assert is_private_or_local_host("127.0.0.1")
    assert is_private_or_local_host("localhost")
    assert not is_private_or_local_host("cdac.gov.in")


def test_soft_error_detection():
    assert is_soft_error_page("Error", 200)
    assert not is_soft_error_page("About C-DAC", 200)


def test_url_length_exceeded():
    assert url_length_exceeded("https://cdac.gov.in/" + ("a" * 3000), 2048)


def test_unsupported_scheme():
    with pytest.raises(ValueError):
        canonicalize_url("ftp://cdac.gov.in/file")

