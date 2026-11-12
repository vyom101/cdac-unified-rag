from rag.crawl.discovery import is_trap_url, should_stop_discovery


def test_trap_pattern_detects_deep_pagination():
    assert is_trap_url("https://cdac.gov.in/news?page=1000")


def test_should_stop_on_frontier_exhaustion():
    stop, reason = should_stop_discovery(
        fetched_count=10, max_urls=100, elapsed_seconds=5,
        max_duration_seconds=3600, frontier_empty=True,
    )
    assert stop is True
    assert reason == "frontier_exhausted"


def test_should_stop_on_page_cap():
    stop, reason = should_stop_discovery(
        fetched_count=100, max_urls=100, elapsed_seconds=5,
        max_duration_seconds=3600, frontier_empty=False,
    )
    assert stop is True
    assert reason == "page_cap_reached"
