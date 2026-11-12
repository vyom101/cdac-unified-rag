import json

from rag.crawl.ledger import CrawlLedgerExtension
from rag.crawl.logging import CrawlFileLogExtension
from rag.models import FailureRecord


def test_ledger_counts_human_review(tmp_path):
    ledger = CrawlLedgerExtension(
        str(tmp_path / "report.json"), "https://cdac.gov.in/", "cdac.gov.in"
    )
    ledger.item_scraped({
        "record_type": "failure",
        "failure": FailureRecord(
            url="https://cdac.gov.in/x", error_type="HTTPError",
            error_message="500", retry_count=3, needs_human_review=True,
        ).model_dump(),
    }, None, None)
    assert ledger.summary.pages_failed == 1
    assert ledger.summary.human_review_required == 1


def test_file_log_extension_writes_success_and_error(tmp_path):
    success, error = tmp_path / "success.log", tmp_path / "error.log"
    extension = CrawlFileLogExtension(str(success), str(error))

    class DummySpider:
        seed_url = "https://cdac.gov.in/"
        allowed_domain = "cdac.gov.in"

        class crawler:
            settings = {"CRAWL_RUN_ATTEMPT": 1}

    extension.spider_opened(DummySpider())
    extension.item_scraped({
        "raw_url": "https://cdac.gov.in/about",
        "canonical_url": "https://cdac.gov.in/about",
        "parent_url": "https://cdac.gov.in/", "sub_link": "/about",
        "depth": 1, "status": 200, "title": "About", "is_soft_error": False,
    }, None, DummySpider())
    extension.item_scraped({
        "record_type": "failure",
        "failure": FailureRecord(
            url="https://cdac.gov.in/broken", error_type="HTTPError",
            error_message="404", retry_count=3, needs_human_review=True,
        ).model_dump(),
    }, None, DummySpider())
    extension.spider_closed(DummySpider(), "finished")
    success_lines = success.read_text(encoding="utf-8").strip().splitlines()
    error_lines = error.read_text(encoding="utf-8").strip().splitlines()
    assert len(success_lines) >= 2
    assert any(json.loads(line)["event"] == "page_fetched" for line in success_lines)
    assert any(json.loads(line)["event"] == "http_request_failed" for line in error_lines)
