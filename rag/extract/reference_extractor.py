"""Extract literal figure, table, and equation references from PDF text."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

_NUMBER = r"\d+(?:\.\d+)*"
_FIGURE_RE = re.compile(rf"\b(?:fig(?:ure)?s?)\s*\.?\s*({_NUMBER})", re.IGNORECASE)
_TABLE_RE = re.compile(rf"\b(?:tab(?:le)?s?)\s*\.?\s*({_NUMBER})", re.IGNORECASE)
_EQUATION_RE = re.compile(
    rf"\b(?:eq(?:uation)?s?)\s*\.?\s*\(?\s*({_NUMBER})\s*\)?", re.IGNORECASE
)
_FIGURE_CAPTION_RE = re.compile(
    rf"^(fig(?:ure)?s?\s*\.?\s*{_NUMBER}\b.*)$", re.IGNORECASE | re.MULTILINE
)
_TABLE_CAPTION_RE = re.compile(
    rf"^(tab(?:le)?s?\s*\.?\s*{_NUMBER}\b.*)$", re.IGNORECASE | re.MULTILINE
)


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _find_numbers(pattern: re.Pattern, text: str) -> List[str]:
    if not text:
        return []
    return _dedupe_preserve_order([match.group(1) for match in pattern.finditer(text)])


def _find_captions(pattern: re.Pattern, text: str) -> List[str]:
    if not text:
        return []
    captions = [match.group(1).strip() for match in pattern.finditer(text)]
    return _dedupe_preserve_order([caption[:200] for caption in captions if caption])


def extract_references(text: str) -> Dict[str, List[str]]:
    return {
        "figure_refs": _find_numbers(_FIGURE_RE, text),
        "table_refs": _find_numbers(_TABLE_RE, text),
        "equation_refs": _find_numbers(_EQUATION_RE, text),
        "figure_captions": _find_captions(_FIGURE_CAPTION_RE, text),
        "table_captions": _find_captions(_TABLE_CAPTION_RE, text),
    }


def extract_query_references(query: str) -> Dict[str, List[str]]:
    return {
        "figure_refs": _find_numbers(_FIGURE_RE, query),
        "table_refs": _find_numbers(_TABLE_RE, query),
        "equation_refs": _find_numbers(_EQUATION_RE, query),
    }


def query_has_references(query: str) -> bool:
    return any(extract_query_references(query).values())


_VISUAL_KEYWORDS = (
    "figure", "fig", "diagram", "plot", "chart", "graph", "image", "picture",
    "screenshot", "visualization", "visualisation", "visualize", "visualise",
    "confusion matrix", "output",
)
_VISUAL_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in _VISUAL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def is_visual_query(query: str) -> bool:
    if not query:
        return False
    return query_has_references(query) or bool(_VISUAL_KEYWORD_RE.search(query))


def build_page_reference_index(
    pages: Sequence[Tuple[int, str]],
) -> Dict[str, Dict[str, List[int]]]:
    figures: Dict[str, List[int]] = {}
    tables: Dict[str, List[int]] = {}
    equations: Dict[str, List[int]] = {}

    def register(target: Dict[str, List[int]], ref: str, page_number: int) -> None:
        page_numbers = target.setdefault(ref, [])
        if page_number not in page_numbers:
            page_numbers.append(page_number)

    for page_number, page_text in pages:
        refs = extract_references(page_text)
        for ref in refs["figure_refs"]:
            register(figures, ref, page_number)
        for ref in refs["table_refs"]:
            register(tables, ref, page_number)
        for ref in refs["equation_refs"]:
            register(equations, ref, page_number)
    return {"figures": figures, "tables": tables, "equations": equations}


def resolve_reference_pages(
    refs: Sequence[str], index: Dict[str, List[int]]
) -> Dict[str, List[int]]:
    return {ref: list(index[ref]) for ref in refs if ref in index and index[ref]}
