from rag.crawl.ledger import CrawlLedgerExtension
from rag.models import DiscoveryRecord, PageState


def test_register_discovery_tracks_shortest_depth(tmp_path):
    ledger = CrawlLedgerExtension(
        str(tmp_path / "report.json"), "https://cdac.gov.in/", "cdac.gov.in"
    )
    first, new_first = ledger.register_discovery("https://cdac.gov.in/about", None, 1)
    second, new_second = ledger.register_discovery(
        "https://cdac.gov.in/about/", "https://cdac.gov.in/", 2
    )
    assert first == second
    assert new_first is True
    assert new_second is False
    assert ledger.summary.pages_duplicate >= 1


def test_record_skip_increments_counter(tmp_path):
    ledger = CrawlLedgerExtension(
        str(tmp_path / "report.json"), "https://cdac.gov.in/", "cdac.gov.in"
    )
    ledger.record_skip(DiscoveryRecord(
        raw_url="https://external.com", canonical_url="https://external.com",
        depth=1, state=PageState.SKIPPED, reason="external_domain",
    ))
    assert ledger.summary.pages_skipped == 1
