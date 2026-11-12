"""Authoritative crawl run reporting and accounting."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from scrapy import signals
from scrapy.exceptions import NotConfigured

from rag.crawl.url_utils import canonicalize_url, is_soft_error_page
from rag.models import (
    CrawlRunReport, DiscoveryRecord, FailureRecord, PageRecord,
    RobotsObservation, RunSummary, utc_now_iso,
)


class CrawlLedgerExtension:
    def __init__(self, output_path: str, seed_url: str, allowed_domain: str):
        self.output_path = Path(output_path)
        self.seed_url = seed_url
        self.allowed_domain = allowed_domain
        self.run_id = str(uuid.uuid4())
        self.started_at = utc_now_iso()
        self._start_monotonic = time.monotonic()
        self.summary = RunSummary(
            run_id=self.run_id, seed_url=seed_url,
            allowed_domain=allowed_domain, started_at=self.started_at,
        )
        self.pages: list[PageRecord] = []
        self.failures: list[FailureRecord] = []
        self.skipped: list[DiscoveryRecord] = []
        self.robots_observations: list[RobotsObservation] = []
        self._shortest_depth: dict[str, int] = {}
        self._seen_canonical: set[str] = set()
        self._depth_histogram: dict[int, int] = defaultdict(int)

    @classmethod
    def from_crawler(cls, crawler):
        output_path = crawler.settings.get("CRAWL_RUN_REPORT_PATH")
        seed_url = crawler.settings.get("CRAWL_SEED_URL")
        allowed_domain = crawler.settings.get("CRAWL_ALLOWED_DOMAIN")
        if not output_path or not seed_url or not allowed_domain:
            raise NotConfigured("Crawl ledger settings missing")
        extension = cls(output_path, seed_url, allowed_domain)
        crawler.signals.connect(extension.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(extension.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(extension.item_scraped, signal=signals.item_scraped)
        return extension

    def spider_opened(self, spider):
        spider.crawl_ledger = self

    def record_skip(self, record: DiscoveryRecord):
        self.skipped.append(record)
        self.summary.pages_skipped += 1

    def record_robots(self, observation: RobotsObservation):
        self.robots_observations.append(observation)
        if observation.allowed:
            self.summary.robots_allowed += 1
        else:
            self.summary.robots_disallowed += 1

    def record_http_retry(self):
        self.summary.http_retry_events += 1

    def record_pdf_link_detected(self):
        self.summary.pdf_links_detected += 1

    def record_pdf_downloaded(self):
        self.summary.pdf_downloaded += 1

    def register_discovery(
        self, raw_url: str, parent_url: str | None, depth: int
    ) -> tuple[str, bool]:
        canonical = canonicalize_url(raw_url)
        is_new = canonical not in self._seen_canonical
        if is_new:
            self._seen_canonical.add(canonical)
            self._shortest_depth[canonical] = depth
            self.summary.pages_discovered += 1
        else:
            self._shortest_depth[canonical] = min(self._shortest_depth.get(canonical, depth), depth)
            self.summary.pages_duplicate += 1
        return canonical, is_new

    def item_scraped(self, item, response, spider):
        if not isinstance(item, dict):
            return
        if item.get("record_type") == "failure":
            failure = FailureRecord(**item["failure"])
            self.failures.append(failure)
            self.summary.pages_failed += 1
            if failure.needs_human_review:
                self.summary.human_review_required += 1
            return
        if item.get("record_type") == "skip":
            self.skipped.append(DiscoveryRecord(**item["skip"]))
            self.summary.pages_skipped += 1
            return
        if item.get("record_type"):
            return
        excluded = {
            "sub_link", "page_id", "section_count", "chunk_count",
            "chunk_summary_count", "composite_summary_count", "chunking_stats", "_response",
        }
        page = PageRecord(**{key: value for key, value in item.items() if key not in excluded})
        self.pages.append(page)
        self.summary.pages_fetched += 1
        if page.is_soft_error:
            self.summary.soft_error_pages += 1
        self._depth_histogram[page.depth] += 1
        self.summary.max_effective_depth = max(self.summary.max_effective_depth, page.depth)

    def spider_closed(self, spider, reason):
        self.summary.finished_at = utc_now_iso()
        self.summary.duration_seconds = round(time.monotonic() - self._start_monotonic, 3)
        self.summary.termination_reason = reason
        self.summary.crawl_run_attempt = int(spider.crawler.settings.get("CRAWL_RUN_ATTEMPT", 1))
        self.summary.depth_histogram = {
            str(key): value for key, value in sorted(self._depth_histogram.items())
        }
        report = CrawlRunReport(
            summary=self.summary, pages=self.pages, failures=self.failures,
            skipped=self.skipped, robots_observations=self.robots_observations,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        spider.logger.info("Crawl run report written to %s", self.output_path)


def build_page_item(
    response, *, parent_url: str | None, depth: int,
    fetch_duration_ms: float | None = None, sub_link: str | None = None,
) -> dict[str, Any]:
    title = response.css("title::text").get(default="").strip()
    headings = {
        level: [heading.strip() for heading in response.css(f"{level}::text").getall() if heading.strip()]
        for level in ("h1", "h2", "h3")
    }
    content_type = response.headers.get("Content-Type", b"").decode("latin-1", errors="ignore")
    return PageRecord(
        raw_url=response.url,
        canonical_url=canonicalize_url(response.url),
        parent_url=parent_url,
        depth=depth,
        status=response.status,
        content_type=content_type,
        title=title,
        meta_description=response.css('meta[name="description"]::attr(content)').get(default=""),
        headings=headings,
        is_soft_error=is_soft_error_page(title, response.status),
        fetch_duration_ms=fetch_duration_ms,
        redirect_chain=list(getattr(response, "redirect_urls", []) or []),
    ).model_dump() | {"sub_link": sub_link}
