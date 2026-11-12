"""Downloader middleware providing interval-based HTTP retries."""

from __future__ import annotations

import time
from typing import Optional

from scrapy.downloadermiddlewares.retry import RetryMiddleware
from scrapy.http import Response
from scrapy.spiders import Spider


class IntervalRetryMiddleware(RetryMiddleware):
    def __init__(self, settings):
        super().__init__(settings)
        self.retry_interval_seconds = settings.getfloat("CRAWL_RETRY_INTERVAL_SECONDS", 2.0)

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)

    def process_response(self, request, response, spider):
        if request.meta.get("dont_retry", False):
            return response
        if response.status in self.retry_http_codes:
            retries = request.meta.get("retry_times", 0)
            if retries < self.max_retry_times:
                self._log_http_retry(spider, request, response, retries + 1)
                time.sleep(self.retry_interval_seconds)
        return super().process_response(request, response, spider)

    def process_exception(self, request, exception, spider):
        retries = request.meta.get("retry_times", 0)
        if retries < self.max_retry_times:
            self._log_http_retry(spider, request, None, retries + 1, error=str(exception))
            time.sleep(self.retry_interval_seconds)
        return super().process_exception(request, exception, spider)

    def _log_http_retry(
        self, spider: Spider, request, response: Optional[Response],
        attempt: int, error: str | None = None,
    ) -> None:
        status = response.status if response is not None else None
        message = (
            f"HTTP retry {attempt}/{self.max_retry_times} for {request.url}"
            + (f" status={status}" if status is not None else "")
            + (f" error={error}" if error else "")
            + f" (waiting {self.retry_interval_seconds}s)"
        )
        spider.logger.warning(message)
        ledger = getattr(spider, "crawl_ledger", None)
        if ledger:
            ledger.record_http_retry()
