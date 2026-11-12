"""Compatibility export for interval retry middleware."""

import time

from rag.crawl.middlewares import IntervalRetryMiddleware

__all__ = ["IntervalRetryMiddleware", "time"]
