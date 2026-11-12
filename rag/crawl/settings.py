"""Scrapy settings backed by the unified application configuration."""

from rag.config import load_config

_cfg = load_config()

BOT_NAME = "rag"
SPIDER_MODULES = ["rag.crawl.spiders"]
NEWSPIDER_MODULE = "rag.crawl.spiders"
USER_AGENT = _cfg.crawl_user_agent
ROBOTSTXT_OBEY = True
DOWNLOAD_DELAY = _cfg.crawl_download_delay
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
CONCURRENT_REQUESTS = 4
CONCURRENT_REQUESTS_PER_DOMAIN = 2
DEPTH_LIMIT = 0
COOKIES_ENABLED = True
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en",
}
RETRY_ENABLED = True
RETRY_TIMES = _cfg.crawl_retry_max_attempts
DOWNLOADER_MIDDLEWARES = {"rag.crawl.middlewares.IntervalRetryMiddleware": 550}
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"
CRAWL_SEED_URL = _cfg.crawl_start_url
CRAWL_ALLOWED_DOMAIN = _cfg.crawl_allowed_domain
CRAWL_MAX_URLS = _cfg.crawl_max_urls
CRAWL_MAX_DEPTH = _cfg.crawl_max_depth
CRAWL_MAX_DURATION_SECONDS = _cfg.crawl_max_duration_seconds
CRAWL_MAX_RESPONSE_BYTES = _cfg.crawl_max_response_bytes
CRAWL_MAX_REDIRECTS = _cfg.crawl_max_redirects
CRAWL_MAX_URL_LENGTH = _cfg.crawl_max_url_length
CRAWL_SEED_SITEMAPS = _cfg.crawl_seed_sitemaps
CRAWL_USER_AGENT = _cfg.crawl_user_agent
CRAWL_RUN_REPORT_PATH = _cfg.crawl_run_report_path
CRAWL_SUCCESS_LOG_PATH = _cfg.crawl_success_log_path
CRAWL_ERROR_LOG_PATH = _cfg.crawl_error_log_path
CRAWL_RETRY_INTERVAL_SECONDS = _cfg.crawl_retry_interval_seconds
CRAWL_RUN_ATTEMPT = 1
CONTENT_ARTIFACT_DIR = _cfg.extracted_data_dir
PDF_DATA_DIR = _cfg.pdf_data_dir
PDF_MANIFEST_PATH = _cfg.pdf_manifest_path
COMPOSITE_GROUP_SIZE = _cfg.composite_group_size
HTML_MAX_CHUNK_SIZE = _cfg.html_max_chunk_size
HTML_CHUNK_OVERLAP = _cfg.html_chunk_overlap
HTML_MIN_CHUNK_SIZE = _cfg.html_min_chunk_size
DUPLICATE_SIMILARITY_THRESHOLD = _cfg.duplicate_similarity_threshold
ITEM_PIPELINES = {
    "rag.crawl.pipelines.PdfDownloadPipeline": 200,
    "rag.crawl.pipelines.ContentExtractionPipeline": 300,
}
EXTENSIONS = {
    "rag.crawl.extensions.CrawlLedgerExtension": 500,
    "rag.crawl.crawl_logging.CrawlFileLogExtension": 510,
}
CLOSESPIDER_PAGECOUNT = _cfg.crawl_max_urls
CLOSESPIDER_TIMEOUT = _cfg.crawl_max_duration_seconds
