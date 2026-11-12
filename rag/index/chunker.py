"""Section-aware HTML chunking with conservative deduplication."""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import List, Optional, Set, Tuple

from pydantic import BaseModel

from rag.models import ChunkModel, SectionRecord


class ChunkingStatistics(BaseModel):
    total_sections: int = 0
    accepted_chunks: int = 0
    rejected_empty: int = 0
    rejected_exact_duplicate: int = 0
    rejected_near_duplicate: int = 0
    merged_small: int = 0


def _document_id(source_url: str) -> str:
    return hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]


def _content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_text_with_overlap(text: str, max_size: int, overlap: int) -> List[str]:
    if len(text) <= max_size:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(para) > max_size:
                start = 0
                while start < len(para):
                    chunks.append(para[start : start + max_size])
                    start += max_size - overlap
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{prev_tail}\n{chunks[i]}".strip())
        return overlapped
    return chunks


class SectionChunker:
    def __init__(
        self,
        *,
        max_chunk_size: int = 1200,
        chunk_overlap: int = 150,
        min_chunk_size: int = 80,
        duplicate_similarity_threshold: float = 0.92,
    ):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.duplicate_similarity_threshold = duplicate_similarity_threshold

    def process_sections(
        self,
        sections: List[SectionRecord],
        *,
        document_name: str,
        source_url: str,
        parent_url: str | None = None,
        depth: int = 0,
        sub_link: str | None = None,
    ) -> Tuple[List[ChunkModel], ChunkingStatistics]:
        stats = ChunkingStatistics(total_sections=len(sections))
        doc_id = _document_id(source_url)
        accepted: list[ChunkModel] = []
        exact_hashes: set[str] = set()
        recent_texts: list[str] = []

        chunk_index = 0
        for section in sections:
            slices = split_text_with_overlap(
                section.text, self.max_chunk_size, self.chunk_overlap
            )
            for slice_text in slices:
                valid, reason = self._validate_chunk(
                    slice_text, exact_hashes, recent_texts
                )
                if not valid:
                    if reason == "empty":
                        stats.rejected_empty += 1
                    elif reason == "exact":
                        stats.rejected_exact_duplicate += 1
                    elif reason == "near":
                        stats.rejected_near_duplicate += 1
                    continue

                chunk_id = f"doc_{doc_id}_chunk_{chunk_index:03d}"
                heading = (
                    " > ".join(section.heading_path)
                    if section.heading_path
                    else section.metadata.get("title")
                )
                accepted.append(ChunkModel(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    document_name=document_name,
                    section=heading,
                    content=slice_text,
                    source_url=source_url,
                    parent_url=parent_url,
                    sub_link=sub_link,
                    depth=depth,
                    chunk_index=chunk_index,
                    content_kind="chunk",
                    source_kind="html",
                    metadata={
                        "section_id": section.section_id,
                        "content_hash": _content_hash(slice_text),
                        "word_count": len(re.findall(r"\w+", slice_text)),
                    },
                ))
                exact_hashes.add(_content_hash(slice_text))
                recent_texts.append(slice_text)
                if len(recent_texts) > 5:
                    recent_texts.pop(0)
                chunk_index += 1
                stats.accepted_chunks += 1
        return accepted, stats

    def _validate_chunk(
        self,
        text: str,
        exact_hashes: Set[str],
        recent_texts: List[str],
    ) -> Tuple[bool, Optional[str]]:
        cleaned = text.strip()
        if len(cleaned) < self.min_chunk_size:
            return False, "empty"
        content_hash = _content_hash(cleaned)
        if content_hash in exact_hashes:
            return False, "exact"
        for recent in recent_texts:
            ratio = SequenceMatcher(None, cleaned.lower(), recent.lower()).ratio()
            if ratio >= self.duplicate_similarity_threshold:
                return False, "near"
        return True, None
