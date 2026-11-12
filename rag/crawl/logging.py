"""Separate structured success and error crawl logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

from scrapy import signals
from scrapy.exceptions import NotConfigured

from rag.models import utc_now_iso


class CrawlFileLogExtension:
    def __init__(self, success_path: str, error_path: str):
        self.success_path = Path(success_path)
        self.error_path = Path(error_path)
        self._success_fp: TextIO | None = None
        self._error_fp: TextIO | None = None

    @classmethod
    def from_crawler(cls, crawler):
        success_path = crawler.settings.get("CRAWL_SUCCESS_LOG_PATH")
        error_path = crawler.settings.get("CRAWL_ERROR_LOG_PATH")
        if not success_path or not error_path:
            raise NotConfigured("Crawl log paths not configured")
        extension = cls(success_path, error_path)
        crawler.signals.connect(extension.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(extension.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(extension.item_scraped, signal=signals.item_scraped)
        return extension

    def spider_opened(self, spider):
        self.success_path.parent.mkdir(parents=True, exist_ok=True)
        self.error_path.parent.mkdir(parents=True, exist_ok=True)
        self._success_fp = self.success_path.open("a", encoding="utf-8")
        self._error_fp = self.error_path.open("a", encoding="utf-8")
        settings = spider.crawler.settings
        run_attempt = (
            settings.getint("CRAWL_RUN_ATTEMPT", 1)
            if hasattr(settings, "getint")
            else int(settings.get("CRAWL_RUN_ATTEMPT", 1))
        )
        self._write(self._success_fp, {
            "event": "crawl_started", "timestamp": utc_now_iso(),
            "seed_url": spider.seed_url, "allowed_domain": spider.allowed_domain,
            "crawl_run_attempt": run_attempt,
        })

    def spider_closed(self, spider, reason):
        ledger = getattr(spider, "crawl_ledger", None)
        self._write(self._success_fp, {
            "event": "crawl_finished", "timestamp": utc_now_iso(),
            "termination_reason": reason,
            "summary": ledger.summary.model_dump() if ledger else {},
        })
        if self._success_fp:
            self._success_fp.close()
        if self._error_fp:
            self._error_fp.close()

    def item_scraped(self, item, response, spider):
        if not isinstance(item, dict):
            return
        record_type = item.get("record_type")
        if record_type == "failure":
            self._write(self._error_fp, {
                "event": "http_request_failed", "timestamp": utc_now_iso(),
                "failure_kind": "http_request", **item["failure"],
            })
        elif record_type == "skip":
            self._write(self._error_fp, {
                "event": "url_skipped", "timestamp": utc_now_iso(),
                "failure_kind": "skip", **item["skip"],
            })
        elif record_type == "pdf_download":
            self._write(self._success_fp, {
                "event": "pdf_downloaded", "timestamp": utc_now_iso(),
                **(item.get("pdf_record") or item.get("pdf_meta", {})),
            })
        elif record_type == "http_retry":
            self._write(self._error_fp, {"timestamp": utc_now_iso(), **item["http_retry"]})
        else:
            self._write(self._success_fp, {
                "event": "page_fetched", "timestamp": utc_now_iso(),
                "url": item.get("raw_url"), "canonical_url": item.get("canonical_url"),
                "parent_url": item.get("parent_url"), "sub_link": item.get("sub_link"),
                "depth": item.get("depth"), "status": item.get("status"),
                "title": item.get("title"), "is_soft_error": item.get("is_soft_error", False),
            })

    def _write(self, stream: TextIO | None, payload: dict[str, Any]) -> None:
        if stream is not None:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stream.flush()
