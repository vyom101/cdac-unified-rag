"""Deterministic extractive summaries for HTML chunks."""

import hashlib
import re
from typing import Sequence

from rag.models import ChunkModel, CompositeSummaryModel, SummaryModel


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


def extractive_summary(text: str, max_sentences: int = 3) -> str:
    sentences = _sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences) or text.strip()[:240]
    frequencies: dict[str, int] = {}
    for word in re.findall(r"\w+", text.casefold()):
        if len(word) > 3:
            frequencies[word] = frequencies.get(word, 0) + 1
    scored = [
        (sum(frequencies.get(word.casefold(), 0)
             for word in re.findall(r"\w+", sentence)), index, sentence)
        for index, sentence in enumerate(sentences)
    ]
    selected = sorted(sorted(scored, key=lambda row: (-row[0], row[1]))[:max_sentences],
                      key=lambda row: row[1])
    return " ".join(sentence for _, _, sentence in selected)


def build_chunk_summaries(
    chunks: Sequence[ChunkModel], max_sentences: int = 3
) -> list[SummaryModel]:
    output = []
    for chunk in chunks:
        text = extractive_summary(chunk.content, max_sentences)
        output.append(SummaryModel(
            summary_id=f"sum_{chunk.chunk_id}", chunk_id=chunk.chunk_id,
            document_id=chunk.document_id, source_url=chunk.source_url,
            parent_url=chunk.parent_url, sub_link=chunk.sub_link, depth=chunk.depth,
            summary_text=text, content_hash=hashlib.sha256(text.encode()).hexdigest(),
            source_kind=chunk.source_kind,
            metadata={"derived_from": chunk.chunk_id, "kind": "chunk_summary"},
        ))
    return output


def build_composite_summaries(
    summaries: Sequence[SummaryModel], *, group_size: int = 10,
    max_sentences: int = 5,
) -> list[CompositeSummaryModel]:
    output = []
    for start in range(0, len(summaries), group_size):
        group = list(summaries[start:start + group_size])
        if not group:
            continue
        anchor = group[0]
        text = extractive_summary("\n".join(item.summary_text for item in group), max_sentences)
        output.append(CompositeSummaryModel(
            composite_id=f"comp_{anchor.document_id}_{start // group_size:03d}",
            document_id=anchor.document_id, source_url=anchor.source_url,
            parent_url=anchor.parent_url, depth=anchor.depth,
            chunk_ids=[item.chunk_id for item in group],
            summary_ids=[item.summary_id for item in group],
            composite_text=text, group_index=start // group_size, group_size=len(group),
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            source_kind=anchor.source_kind,
            metadata={"kind": "composite_summary", "covers_chunks": len(group)},
        ))
    return output
