"""Bounded discovery helpers: sitemap seeding, robots observation, trap detection."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Set
from urllib.parse import urljoin, urlparse

import requests

from rag.crawl.url_utils import canonicalize_url, is_same_domain
from rag.models import RobotsObservation

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
TRAP_PATTERNS = [
    re.compile(r"[?&]page=\d{3,}"),
    re.compile(r"[?&]year=\d{4}&month=\d{2}&day=\d{2}"),
    re.compile(r"[?&](session|sid|token)=[^&]+", re.I),
]


@dataclass
class DiscoveryConfig:
    allowed_domain: str
    max_urls: int = 500
    max_depth: int | None = None
    max_duration_seconds: int = 3600
    max_response_bytes: int = 5_000_000
    max_redirects: int = 5
    max_url_length: int = 2048
    seed_sitemaps: bool = True
    user_agent: str = "CdacResearchBot/1.0"


@dataclass
class DiscoveryStats:
    sitemap_urls_seeded: int = 0
    trap_skips: int = 0
    safety_caps_triggered: List[str] = field(default_factory=list)


def is_trap_url(url: str) -> bool:
    return any(pattern.search(url) for pattern in TRAP_PATTERNS)


def fetch_robots_txt(base_url: str, user_agent: str, timeout: int = 10) -> str:
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        response = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=timeout)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        pass
    return ""


def parse_robots_disallowed_paths(robots_text: str, user_agent: str) -> Set[str]:
    disallowed: set[str] = set()
    current_agents: list[str] = []
    for line in robots_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("user-agent:"):
            current_agents = [line.split(":", 1)[1].strip().lower()]
        elif line.lower().startswith("disallow:"):
            path = line.split(":", 1)[1].strip()
            if path and any(a in {"*", user_agent.lower()} for a in current_agents):
                disallowed.add(path)
    return disallowed


def is_robots_allowed(url: str, disallowed_paths: Set[str]) -> bool:
    path = urlparse(url).path or "/"
    return not any(disallowed == "/" or path.startswith(disallowed) for disallowed in disallowed_paths)


def observe_robots(url: str, allowed: bool, user_agent: str) -> RobotsObservation:
    return RobotsObservation(url=url, allowed=allowed, user_agent=user_agent)


def discover_sitemap_urls(base_url: str, user_agent: str, timeout: int = 15) -> List[str]:
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    discovered: list[str] = []
    for line in fetch_robots_txt(base_url, user_agent, timeout=timeout).splitlines():
        if line.lower().startswith("sitemap:"):
            sitemap_url = line.split(":", 1)[1].strip()
            if sitemap_url.startswith("http"):
                discovered.append(sitemap_url)
    for candidate in (urljoin(origin, "/sitemap.xml"), urljoin(origin, "/sitemap_index.xml")):
        if candidate not in discovered:
            discovered.append(candidate)
    all_urls: list[str] = []
    seen_sitemaps: set[str] = set()
    for sitemap_url in discovered:
        all_urls.extend(_parse_sitemap_recursive(sitemap_url, user_agent, seen_sitemaps, timeout))
    result: list[str] = []
    seen: set[str] = set()
    for url in all_urls:
        try:
            canonical = canonicalize_url(url)
        except ValueError:
            continue
        if is_same_domain(canonical, parsed.netloc) and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def _parse_sitemap_recursive(
    sitemap_url: str, user_agent: str, seen_sitemaps: Set[str], timeout: int
) -> List[str]:
    if sitemap_url in seen_sitemaps:
        return []
    seen_sitemaps.add(sitemap_url)
    try:
        response = requests.get(sitemap_url, headers={"User-Agent": user_agent}, timeout=timeout)
        if response.status_code != 200:
            return []
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError):
        return []
    tag = _strip_ns(root.tag)
    urls: list[str] = []
    if tag == "sitemapindex":
        for loc in root.findall(".//sm:loc", SITEMAP_NS):
            if loc.text:
                urls.extend(_parse_sitemap_recursive(loc.text.strip(), user_agent, seen_sitemaps, timeout))
    elif tag == "urlset":
        urls.extend(loc.text.strip() for loc in root.findall(".//sm:loc", SITEMAP_NS) if loc.text)
    else:
        urls.extend(loc.text.strip() for loc in root.iter() if _strip_ns(loc.tag) == "loc" and loc.text)
    return urls


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def should_stop_discovery(
    *, fetched_count: int, max_urls: int, elapsed_seconds: float,
    max_duration_seconds: int, frontier_empty: bool,
) -> tuple[bool, str]:
    if frontier_empty:
        return True, "frontier_exhausted"
    if fetched_count >= max_urls:
        return True, "page_cap_reached"
    if elapsed_seconds >= max_duration_seconds:
        return True, "duration_cap_reached"
    return False, "in_progress"
