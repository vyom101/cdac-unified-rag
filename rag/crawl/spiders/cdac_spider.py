"""C-DAC website spider with crawl ledger, discovery, and safety controls."""

from __future__ import annotations

import time
from urllib.parse import urlparse

import scrapy

from rag.crawl.discovery import (
    DiscoveryConfig, discover_sitemap_urls, fetch_robots_txt, is_robots_allowed,
    is_trap_url, observe_robots, parse_robots_disallowed_paths,
)
from rag.crawl.ledger import build_page_item
from rag.crawl.url_utils import (
    SUPPORTED_SCHEMES, canonicalize_url, is_html_content_type,
    is_irrelevant_asset_url, is_pdf_content_type, is_pdf_url,
    is_private_or_local_host, is_same_domain, url_length_exceeded,
)
from rag.models import DiscoveryRecord, FailureRecord, PageState, SkipReason


class CdacSpider(scrapy.Spider):
    name = "cdac"

    def __init__(
        self, start_url: str | None = None, allowed_domain: str | None = None,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seed_url = start_url or "https://cdac.gov.in/"
        parsed = urlparse(self.seed_url)
        self.allowed_domain = allowed_domain or parsed.netloc.replace("www.", "")
        self.allowed_domains = [self.allowed_domain, f"www.{self.allowed_domain}"]
        self.start_urls = [self.seed_url]
        self.discovery = DiscoveryConfig(allowed_domain=self.allowed_domain)
        self._robots_disallowed: set[str] = set()
        self._queued_canonical: set[str] = set()
        self._fetch_start: dict[str, float] = {}

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = cls(
            start_url=crawler.settings.get("CRAWL_SEED_URL"),
            allowed_domain=crawler.settings.get("CRAWL_ALLOWED_DOMAIN"),
            *args, **kwargs,
        )
        spider._set_crawler(crawler)
        spider.discovery = DiscoveryConfig(
            allowed_domain=spider.allowed_domain,
            max_urls=int(crawler.settings.get("CRAWL_MAX_URLS", 500)),
            max_depth=crawler.settings.get("CRAWL_MAX_DEPTH"),
            max_duration_seconds=int(crawler.settings.get("CRAWL_MAX_DURATION_SECONDS", 3600)),
            max_response_bytes=int(crawler.settings.get("CRAWL_MAX_RESPONSE_BYTES", 5_000_000)),
            max_redirects=int(crawler.settings.get("CRAWL_MAX_REDIRECTS", 5)),
            max_url_length=int(crawler.settings.get("CRAWL_MAX_URL_LENGTH", 2048)),
            seed_sitemaps=bool(crawler.settings.get("CRAWL_SEED_SITEMAPS", True)),
            user_agent=crawler.settings.get("CRAWL_USER_AGENT", crawler.settings.get("USER_AGENT")),
        )
        return spider

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "SCHEDULER_DISK_QUEUE": "scrapy.squeues.PickleFifoDiskQueue",
        "SCHEDULER_MEMORY_QUEUE": "scrapy.squeues.FifoMemoryQueue",
        "DUPEFILTER_CLASS": "scrapy.dupefilters.RFPDupeFilter",
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "DEPTH_LIMIT": 0,
        "DEPTH_PRIORITY": 1,
        "FEEDS": {
            "cdac_output.json": {"format": "json", "encoding": "utf8", "indent": 2, "overwrite": True},
            "cdac_output.csv": {"format": "csv", "overwrite": True},
        },
        "LOG_LEVEL": "INFO",
        "ITEM_PIPELINES": {
            "rag.crawl.pipelines.PdfDownloadPipeline": 200,
            "rag.crawl.pipelines.ContentExtractionPipeline": 300,
        },
        "EXTENSIONS": {
            "rag.crawl.ledger.CrawlLedgerExtension": 500,
            "rag.crawl.logging.CrawlFileLogExtension": 510,
        },
    }

    def start_requests(self):
        user_agent = self.discovery.user_agent
        robots_text = fetch_robots_txt(self.seed_url, user_agent)
        self._robots_disallowed = parse_robots_disallowed_paths(robots_text, user_agent)
        yield from self._schedule_url(self.seed_url, None, 0, "seed")
        if self.discovery.seed_sitemaps:
            sitemap_urls = discover_sitemap_urls(self.seed_url, user_agent)
            if hasattr(self, "crawl_ledger"):
                self.crawl_ledger.summary.sitemap_urls_seeded = len(sitemap_urls)
            for sitemap_url in sitemap_urls:
                yield from self._schedule_url(sitemap_url, self.seed_url, 1, "sitemap")

    def _schedule_url(
        self, raw_url: str, parent_url: str | None, depth: int,
        source: str, sub_link: str | None = None,
    ):
        ledger = getattr(self, "crawl_ledger", None)
        parsed = urlparse(raw_url)
        scheme = (parsed.scheme or "https").lower()
        checks = (
            (scheme not in SUPPORTED_SCHEMES, SkipReason.UNSUPPORTED_SCHEME.value),
            (is_private_or_local_host(parsed.hostname or ""), SkipReason.PRIVATE_NETWORK.value),
            (url_length_exceeded(raw_url, self.discovery.max_url_length), SkipReason.URL_TOO_LONG.value),
            (not is_same_domain(raw_url, self.allowed_domain), SkipReason.EXTERNAL_DOMAIN.value),
            (is_trap_url(raw_url), SkipReason.TRAP_PATTERN.value),
            (is_irrelevant_asset_url(raw_url), SkipReason.IRRELEVANT_BINARY.value),
        )
        for rejected, reason in checks:
            if rejected:
                yield from self._yield_skip(raw_url, parent_url, depth, reason)
                return
        try:
            canonical = canonicalize_url(raw_url)
        except ValueError as error:
            yield from self._yield_skip(raw_url, parent_url, depth, str(error))
            return
        allowed = is_robots_allowed(canonical, self._robots_disallowed)
        if ledger:
            ledger.record_robots(observe_robots(canonical, allowed, self.discovery.user_agent))
        if not allowed:
            yield from self._yield_skip(raw_url, parent_url, depth, SkipReason.ROBOTS_DISALLOWED.value)
            return
        if ledger:
            _, is_new = ledger.register_discovery(raw_url, parent_url, depth)
            if not is_new and canonical in self._queued_canonical:
                yield from self._yield_skip(raw_url, parent_url, depth, SkipReason.DUPLICATE.value)
                return
        if self.discovery.max_depth is not None and depth > self.discovery.max_depth:
            yield from self._yield_skip(raw_url, parent_url, depth, SkipReason.SAFETY_CAP.value)
            return
        self._queued_canonical.add(canonical)
        self._fetch_start[raw_url] = time.monotonic()
        metadata = {
            "depth": depth, "parent_url": parent_url, "canonical_url": canonical,
            "source": source, "sub_link": sub_link or raw_url,
        }
        if is_pdf_url(canonical) or is_pdf_url(raw_url):
            if ledger:
                ledger.record_pdf_link_detected()
            metadata["is_pdf"] = True
            yield scrapy.Request(raw_url, callback=self.parse_pdf, errback=self.handle_error, meta=metadata)
        else:
            yield scrapy.Request(raw_url, callback=self.parse, errback=self.handle_error, meta=metadata)

    def parse(self, response):
        depth = int(response.meta.get("depth", 0))
        parent_url = response.meta.get("parent_url")
        canonical = response.meta.get("canonical_url") or canonicalize_url(response.url)
        started = self._fetch_start.pop(response.url, None)
        fetch_ms = round((time.monotonic() - started) * 1000, 2) if started else None
        content_type = response.headers.get("Content-Type", b"").decode("latin-1", errors="ignore")
        if is_pdf_content_type(content_type) or response.url.lower().endswith(".pdf"):
            yield from self._handle_pdf_response(response)
            return
        if not is_html_content_type(content_type):
            yield from self._yield_skip(response.url, parent_url, depth, SkipReason.NON_HTML.value)
            return
        if (
            len(response.body) > self.discovery.max_response_bytes
            or len(getattr(response, "redirect_urls", []) or []) > self.discovery.max_redirects
        ):
            yield from self._yield_skip(response.url, parent_url, depth, SkipReason.SAFETY_CAP.value)
            return
        item = build_page_item(
            response, parent_url=parent_url, depth=depth,
            fetch_duration_ms=fetch_ms, sub_link=response.meta.get("sub_link"),
        )
        item["_response"] = response
        if item.get("is_soft_error"):
            yield from self._yield_skip(response.url, parent_url, depth, SkipReason.SOFT_ERROR.value)
            yield item
            return
        yield item
        for href in response.css("a::attr(href)").getall():
            yield from self._schedule_url(
                response.urljoin(href), canonical, depth + 1, "anchor", href
            )

    def parse_pdf(self, response):
        yield from self._handle_pdf_response(response)

    def _handle_pdf_response(self, response):
        depth = int(response.meta.get("depth", 0))
        parent_url = response.meta.get("parent_url")
        canonical = response.meta.get("canonical_url") or canonicalize_url(response.url)
        started = self._fetch_start.pop(response.url, None)
        fetch_ms = round((time.monotonic() - started) * 1000, 2) if started else None
        content_type = response.headers.get("Content-Type", b"").decode("latin-1", errors="ignore")
        if not (is_pdf_content_type(content_type) or response.body[:4] == b"%PDF"):
            yield from self._yield_skip(response.url, parent_url, depth, SkipReason.NON_HTML.value)
            return
        if len(response.body) > self.discovery.max_response_bytes:
            yield from self._yield_skip(response.url, parent_url, depth, SkipReason.SAFETY_CAP.value)
            return
        yield {
            "record_type": "pdf_download",
            "pdf_meta": {
                "source_url": response.url, "canonical_url": canonical,
                "parent_url": parent_url, "sub_link": response.meta.get("sub_link"),
                "depth": depth, "status": response.status,
                "content_type": content_type, "fetch_duration_ms": fetch_ms,
            },
            "_body": response.body,
        }

    def handle_error(self, failure):
        request = failure.request
        retry_times = int(request.meta.get("retry_times", 0))
        max_retries = int(self.crawler.settings.get("RETRY_TIMES", 3))
        exhausted = retry_times >= max_retries
        yield {
            "record_type": "failure",
            "failure": FailureRecord(
                url=request.url,
                canonical_url=request.meta.get("canonical_url"),
                parent_url=request.meta.get("parent_url"),
                depth=int(request.meta.get("depth", 0)),
                error_type=failure.type.__name__ if failure.type else "Unknown",
                error_message=str(failure.value),
                retry_count=retry_times,
                failure_kind="http_request",
                needs_human_review=exhausted,
            ).model_dump(),
        }
        if exhausted:
            self.logger.error(
                "Request failed after %d HTTP retries — marked for human review: %s | %s",
                retry_times, request.url, failure.value,
            )
        else:
            self.logger.warning("Request failed: %s | %s", request.url, failure.value)

    def _yield_skip(self, raw_url: str, parent_url: str | None, depth: int, reason: str):
        try:
            canonical = canonicalize_url(raw_url)
        except ValueError:
            canonical = raw_url
        record = DiscoveryRecord(
            raw_url=raw_url, canonical_url=canonical, parent_url=parent_url,
            depth=depth, state=PageState.SKIPPED, reason=reason,
        )
        if hasattr(self, "crawl_ledger"):
            self.crawl_ledger.record_skip(record)
        yield {"record_type": "skip", "skip": record.model_dump()}
