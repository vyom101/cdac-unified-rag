"""Logical HTML content extraction with boilerplate removal."""

from __future__ import annotations

import hashlib
import re
from typing import List, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

from rag.models import SectionRecord

BOILERPLATE_TAGS = {"nav", "header", "footer", "aside", "script", "style", "noscript", "form"}
BOILERPLATE_CLASS_RE = re.compile(
    r"(nav|menu|footer|header|sidebar|breadcrumb|cookie|banner|social|share|widget)", re.I
)


def _text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _tag_id_class(tag: Tag) -> str:
    attrs = tag.attrs or {}
    tag_id = attrs.get("id", "") if isinstance(attrs, dict) else ""
    classes = attrs.get("class", []) if isinstance(attrs, dict) else []
    if isinstance(classes, str):
        classes = [classes]
    return f"{tag_id} {' '.join(classes)}"


def _score_main_container(tag: Tag) -> int:
    score = 50 if tag.name in {"main", "article"} else 0
    attrs = tag.attrs or {}
    role = ((attrs.get("role") if isinstance(attrs, dict) else None) or "").lower()
    if role in {"main", "article"}:
        score += 40
    if any(key in _tag_id_class(tag).lower() for key in ("content", "main", "article", "body")):
        score += 20
    return score + min(len(tag.find_all(["p", "li", "td"])), 100)


def _find_main_container(soup: BeautifulSoup) -> Tag:
    candidates = soup.find_all(["main", "article", "div", "section"])
    return max(candidates, key=_score_main_container) if candidates else (soup.body or soup)


def _remove_boilerplate(root: Tag) -> None:
    for tag in list(root.find_all(True)):
        if not isinstance(tag, Tag):
            continue
        if tag.name in BOILERPLATE_TAGS or BOILERPLATE_CLASS_RE.search(_tag_id_class(tag)):
            tag.decompose()


def _extract_blocks(container: Tag) -> List[Tuple[List[str], str]]:
    sections: list[tuple[list[str], str]] = []
    heading_path: list[str] = []
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        text = "\n".join(line for line in current_lines if line.strip())
        if text.strip():
            sections.append((list(heading_path), text))
        current_lines = []

    for element in container.descendants:
        if isinstance(element, NavigableString):
            if element.parent and element.parent.name in BOILERPLATE_TAGS:
                continue
            text = str(element).strip()
            if text:
                current_lines.append(text)
        elif isinstance(element, Tag):
            if element.name in {"h1", "h2", "h3", "h4"}:
                flush()
                heading_text = element.get_text(" ", strip=True)
                if heading_text:
                    level = int(element.name[1])
                    heading_path = heading_path[: level - 1]
                    heading_path.append(heading_text)
            elif element.name in {"p", "li", "td", "th", "blockquote"}:
                line = element.get_text(" ", strip=True)
                if line:
                    current_lines.append(line)
            elif element.name == "table":
                rows = []
                for row in element.find_all("tr"):
                    cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    current_lines.append("\n".join(rows))
    flush()
    return sections


def extract_sections(
    html: str, *, page_id: str, source_url: str, title: str = ""
) -> List[SectionRecord]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    container = _find_main_container(soup)
    _remove_boilerplate(container)
    blocks = _extract_blocks(container)
    if not blocks:
        fallback_text = container.get_text("\n", strip=True)
        if fallback_text:
            blocks = [([], fallback_text)]
    sections: list[SectionRecord] = []
    for index, (heading_path, text) in enumerate(blocks):
        if _word_count(text) < 5:
            continue
        sections.append(
            SectionRecord(
                section_id=f"{page_id}_sec_{index:03d}",
                page_id=page_id,
                heading_path=heading_path,
                text=text,
                word_count=_word_count(text),
                content_hash=_text_hash(text),
                source_url=source_url,
                metadata={"title": title, "heading_path": " > ".join(heading_path) or title},
            )
        )
    return sections
