"""Tests for chunk and composite summary generation."""

from rag.models import ChunkModel
from rag.index.summary_builder import build_chunk_summaries, build_composite_summaries, extractive_summary


def _chunk(idx: int, text: str) -> ChunkModel:
    return ChunkModel(
        chunk_id=f"doc_ab_chunk_{idx:03d}",
        document_id="ab",
        document_name="About",
        content=text,
        source_url="https://cdac.gov.in/about",
        parent_url="https://cdac.gov.in/",
        sub_link="/about",
        depth=1,
        chunk_index=idx,
    )


def test_extractive_summary_shortens_text():
    text = (
        "C-DAC is a premier R&D organization. "
        "It operates from Pune, India. "
        "The centre develops advanced computing systems."
    )
    summary = extractive_summary(text, max_sentences=2)
    assert len(summary) < len(text)
    assert "C-DAC" in summary or "Pune" in summary


def test_build_chunk_summaries_for_each_chunk():
    chunks = [_chunk(0, "C-DAC develops PARAM supercomputers in Pune."), _chunk(1, "Advanced computing research continues.")]
    summaries = build_chunk_summaries(chunks)
    assert len(summaries) == 2
    assert summaries[0].chunk_id == chunks[0].chunk_id
    assert summaries[0].parent_url == "https://cdac.gov.in/"
    assert summaries[0].depth == 1


def test_build_composite_summaries_groups_by_ten():
    chunks = [_chunk(i, f"Chunk content number {i} about C-DAC research topics.") for i in range(12)]
    summaries = build_chunk_summaries(chunks)
    composites = build_composite_summaries(summaries, group_size=10)
    assert len(composites) == 2
    assert composites[0].group_size == 10
    assert composites[1].group_size == 2
    assert len(composites[0].chunk_ids) == 10

