"""Tests for crawl file logging and retry configuration."""

import json
from pathlib import Path

from rag.config import AppConfig
from rag.crawl.crawl_logging import CrawlFileLogExtension
from rag.crawl.extensions import CrawlLedgerExtension
from rag.models import FailureRecord


def test_config_retry_defaults():
    config = AppConfig()
    assert config.crawl_retry_max_attempts == 3
    assert config.crawl_retry_interval_seconds == 2.0


def test_failure_record_human_review_flag():
    failure = FailureRecord(
        url="https://cdac.gov.in/missing",
        error_type="TimeoutError",
        error_message="timed out",
        retry_count=3,
        failure_kind="http_request",
        needs_human_review=True,
    )
    assert failure.needs_human_review is True
    assert failure.failure_kind == "http_request"


def test_ledger_counts_human_review(tmp_path):
    ledger = CrawlLedgerExtension(
        output_path=str(tmp_path / "report.json"),
        seed_url="https://cdac.gov.in/",
        allowed_domain="cdac.gov.in",
    )
    ledger.item_scraped(
        {
            "record_type": "failure",
            "failure": FailureRecord(
                url="https://cdac.gov.in/x",
                error_type="HTTPError",
                error_message="500",
                retry_count=3,
                needs_human_review=True,
            ).model_dump(),
        },
        response=None,
        spider=None,
    )
    assert ledger.summary.pages_failed == 1
    assert ledger.summary.human_review_required == 1


def test_file_log_extension_writes_success_and_error(tmp_path):
    success = tmp_path / "success.log"
    error = tmp_path / "error.log"
    ext = CrawlFileLogExtension(str(success), str(error))

    class DummySpider:
        seed_url = "https://cdac.gov.in/"
        allowed_domain = "cdac.gov.in"

        class crawler:
            settings = {"CRAWL_RUN_ATTEMPT": 1}

    ext.spider_opened(DummySpider())
    ext.item_scraped(
        {
            "raw_url": "https://cdac.gov.in/about",
            "canonical_url": "https://cdac.gov.in/about",
            "parent_url": "https://cdac.gov.in/",
            "sub_link": "/about",
            "depth": 1,
            "status": 200,
            "title": "About",
            "is_soft_error": False,
        },
        response=None,
        spider=DummySpider(),
    )
    ext.item_scraped(
        {
            "record_type": "failure",
            "failure": FailureRecord(
                url="https://cdac.gov.in/broken",
                error_type="HTTPError",
                error_message="404",
                retry_count=3,
                needs_human_review=True,
            ).model_dump(),
        },
        response=None,
        spider=DummySpider(),
    )
    ext.spider_closed(DummySpider(), "finished")

    success_lines = success.read_text(encoding="utf-8").strip().splitlines()
    error_lines = error.read_text(encoding="utf-8").strip().splitlines()
    assert len(success_lines) >= 2
    assert any(json.loads(line)["event"] == "page_fetched" for line in success_lines)
    assert any(json.loads(line)["event"] == "http_request_failed" for line in error_lines)

