"""Tests for crawl ledger extension."""

from rag.crawl.extensions import CrawlLedgerExtension
from rag.models import DiscoveryRecord, PageState


def test_register_discovery_tracks_shortest_depth(tmp_path):
    ledger = CrawlLedgerExtension(
        output_path=str(tmp_path / "report.json"),
        seed_url="https://cdac.gov.in/",
        allowed_domain="cdac.gov.in",
    )
    c1, new1 = ledger.register_discovery("https://cdac.gov.in/about", None, 1)
    c2, new2 = ledger.register_discovery("https://cdac.gov.in/about/", "https://cdac.gov.in/", 2)
    assert c1 == c2
    assert new1 is True
    assert new2 is False
    assert ledger.summary.pages_duplicate >= 1


def test_record_skip_increments_counter(tmp_path):
    ledger = CrawlLedgerExtension(
        output_path=str(tmp_path / "report.json"),
        seed_url="https://cdac.gov.in/",
        allowed_domain="cdac.gov.in",
    )
    ledger.record_skip(
        DiscoveryRecord(
            raw_url="https://external.com",
            canonical_url="https://external.com",
            depth=1,
            state=PageState.SKIPPED,
            reason="external_domain",
        )
    )
    assert ledger.summary.pages_skipped == 1

