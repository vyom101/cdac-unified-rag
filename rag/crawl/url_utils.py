"""Conservative URL canonicalization and domain checks."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "gclid", "fbclid", "sessionid", "session_id", "sid", "phpsessid",
        "jsessionid", "_ga", "_gid",
    }
)
MEANINGFUL_QUERY_KEYS = frozenset(
    {"id", "print", "cat", "category", "page", "type", "lang", "section"}
)
DEFAULT_SCHEME = "https"
SUPPORTED_SCHEMES = frozenset({"http", "https"})


def _normalize_host(host: str) -> str:
    host = host.lower().strip(".")
    return host[4:] if host.startswith("www.") else host


def _normalize_path(path: str) -> str:
    if not path:
        return "/"
    path = re.sub(r"/{2,}", "/", path)
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return path or "/"


def canonicalize_url(url: str, *, base_url: str | None = None) -> str:
    """Return a conservative canonical URL identity."""
    if base_url:
        from urllib.parse import urljoin

        url = urljoin(base_url, url)
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or DEFAULT_SCHEME).lower()
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f"Unsupported scheme: {scheme}")
    host = _normalize_host(parsed.hostname or "")
    if not host:
        raise ValueError("Missing hostname")
    port = parsed.port
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    query_pairs = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        key_lower = key.lower()
        if key_lower in TRACKING_PARAMS:
            continue
        if key_lower in MEANINGFUL_QUERY_KEYS or key_lower not in TRACKING_PARAMS:
            query_pairs.append((key, value))
    query_pairs.sort(key=lambda kv: (kv[0].lower(), kv[1]))
    return urlunparse(
        (scheme, netloc, _normalize_path(parsed.path), "", urlencode(query_pairs, doseq=True), "")
    )


def is_same_domain(url: str, allowed_domain: str) -> bool:
    try:
        host = _normalize_host(urlparse(url).hostname or "")
        allowed = _normalize_host(allowed_domain)
    except Exception:
        return False
    return host == allowed or host.endswith(f".{allowed}")


def is_private_or_local_host(hostname: str) -> bool:
    if not hostname:
        return True
    host = hostname.lower().strip(".")
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_link_local or ip.is_loopback or ip.is_reserved
    except ValueError:
        return host.endswith(".local") or host.endswith(".internal")


def is_html_content_type(content_type: str | None) -> bool:
    if not content_type:
        return False
    return content_type.split(";")[0].strip().lower() in {
        "text/html", "application/xhtml+xml",
    }


def is_pdf_content_type(content_type: str | None) -> bool:
    return bool(content_type) and content_type.split(";")[0].strip().lower() == "application/pdf"


def is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


IRRELEVANT_ASSET_EXTENSIONS = frozenset(
    {
        ".css", ".js", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico",
        ".webp", ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3",
        ".avi", ".zip", ".rar", ".7z", ".exe", ".dmg", ".xml",
    }
)


def is_irrelevant_asset_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return False if is_pdf_url(url) else any(path.endswith(ext) for ext in IRRELEVANT_ASSET_EXTENSIONS)


def is_processable_url(url: str) -> bool:
    if is_pdf_url(url):
        return True
    path = urlparse(url).path.lower()
    leaf = path.split("/")[-1]
    if path.endswith((".html", ".htm", ".aspx", "/")) or "." not in leaf:
        return True
    return not is_irrelevant_asset_url(url)


def is_soft_error_page(title: str, status: int) -> bool:
    return status == 200 and (title or "").strip().lower() in {
        "error", "404", "page not found", "not found",
    }


def url_length_exceeded(url: str, max_length: int) -> bool:
    return len(url) > max_length
