from unittest.mock import MagicMock, patch

from rag.crawl.retries import IntervalRetryMiddleware


def _settings(retry_times=3, interval=0.0):
    settings = MagicMock()
    settings.getint.side_effect = lambda key, default=0: {
        "RETRY_TIMES": retry_times,
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],
    }.get(key, default)
    settings.getlist.return_value = [500, 502, 503, 504, 408, 429]
    settings.getfloat.return_value = interval
    return settings


def test_middleware_reads_retry_interval():
    middleware = IntervalRetryMiddleware(_settings(retry_times=3, interval=2.5))
    assert middleware.max_retry_times == 3
    assert middleware.retry_interval_seconds == 2.5


@patch("rag.crawl.retries.time.sleep")
def test_middleware_logging_does_not_sleep(mock_sleep):
    middleware = IntervalRetryMiddleware(_settings(retry_times=3, interval=1.5))
    request = MagicMock()
    request.meta = {"retry_times": 0}
    request.url = "https://cdac.gov.in/test"
    response = MagicMock()
    response.status = 503
    spider = MagicMock()
    spider.crawl_ledger = MagicMock()
    middleware._log_http_retry(spider, request, response, 1)
    mock_sleep.assert_not_called()
    spider.crawl_ledger.record_http_retry.assert_called_once()
