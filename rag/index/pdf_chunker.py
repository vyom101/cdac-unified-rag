"""Structure-aware PDF chunking with overlap, deduplication, and OCR image chunks."""

import difflib
import hashlib
import json
import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Set

from rag.answer.llm_client import BaseLLMClient, get_llm_client
from rag.config import AppConfig, load_config
from rag.extract.reference_extractor import (
    build_page_reference_index,
    extract_references,
    resolve_reference_pages,
)
from rag.models import (
    BlockModel,
    ChunkingStatistics,
    ChunkModel,
    DocumentModel,
)

logger = logging.getLogger("pdf_chunker")
DEFAULT_MAX_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MIN_CHUNK_SIZE = 300


def split_text_with_overlap(text: str, max_size: int, overlap: int) -> List[str]:
    if len(text) <= max_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        search_min = start + overlap
        boundary = text.rfind("\n\n", search_min, end)
        if boundary != -1:
            boundary += 2
        else:
            boundary = text.rfind("\n", search_min, end)
            if boundary != -1:
                boundary += 1
            else:
                sentence_ends = [
                    search_min + match.start()
                    for match in re.finditer(
                        r"\b\.\s|\b\?\s|\b\!\s", text[search_min:end]
                    )
                ]
                boundary = sentence_ends[-1] + 2 if sentence_ends else end
        chunks.append(text[start:boundary].strip())
        start = max(0, boundary - overlap)
    return chunks


class Chunker:
    def __init__(
        self,
        config: Optional[AppConfig] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self.config = config or load_config()
        self.llm_client = llm_client or get_llm_client(self.config)
        self.logger = logger

    @property
    def max_size(self) -> int:
        compatibility = getattr(
            self.config, "max_chunk_size", DEFAULT_MAX_CHUNK_SIZE
        )
        if compatibility != DEFAULT_MAX_CHUNK_SIZE:
            return compatibility
        return getattr(self.config, "pdf_max_chunk_size", compatibility)

    @property
    def overlap(self) -> int:
        compatibility = getattr(
            self.config, "chunk_overlap", DEFAULT_CHUNK_OVERLAP
        )
        if compatibility != DEFAULT_CHUNK_OVERLAP:
            return compatibility
        return getattr(self.config, "pdf_chunk_overlap", compatibility)

    @property
    def min_size(self) -> int:
        compatibility = getattr(
            self.config, "min_chunk_size", DEFAULT_MIN_CHUNK_SIZE
        )
        if compatibility != DEFAULT_MIN_CHUNK_SIZE:
            return compatibility
        return getattr(self.config, "pdf_min_chunk_size", compatibility)

    def _validate_chunk(
        self,
        chunk: ChunkModel,
        source_pages: Set[int],
        existing_hashes: Set[str],
        recent_chunks: Optional[Deque[ChunkModel]] = None,
    ) -> Optional[str]:
        if not chunk.content or not chunk.content.strip():
            return "Empty content segment."
        validation_minimum = min(20, max(0, self.min_size))
        if len(chunk.content.strip()) < validation_minimum:
            return (
                f"Segment is too short to be semantically useful "
                f"({len(chunk.content)} chars)."
            )
        if not chunk.document_id or not chunk.chunk_id:
            return "Missing document_id or chunk_id."
        if not chunk.page_numbers:
            return "No page numbers registered."
        for page in chunk.page_numbers:
            if page not in source_pages:
                return (
                    f"Hallucinated page reference {page} "
                    f"(must exist in {source_pages})."
                )
        content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
        if content_hash in existing_hashes:
            return "Duplicate content segment detected."
        threshold = getattr(self.config, "duplicate_similarity_threshold", 0.92)
        if recent_chunks and threshold < 1.0:
            for other in recent_chunks:
                similarity = difflib.SequenceMatcher(
                    None, chunk.content, other.content
                ).ratio()
                if similarity >= threshold:
                    return (
                        "Duplicate content segment detected "
                        f"(near-duplicate, similarity={similarity:.2f})."
                    )
        existing_hashes.add(content_hash)
        return None

    def chunk_structurally(self, document: DocumentModel) -> tuple:
        blocks: List[BlockModel] = []
        source_pages = set()
        for page in document.pages:
            source_pages.add(page.page_number)
            if page.extraction_status != "FAILED":
                blocks.extend(block for block in page.blocks if block.text.strip())

        chunks = []
        existing_hashes: Set[str] = set()
        recent: Deque[ChunkModel] = deque(maxlen=5)
        duplicate_count = rejected_count = 0
        chunk_index = 1
        active_section = "Introduction"
        accumulated: List[BlockModel] = []
        current_size = 0

        def emit(group: List[BlockModel], section: str) -> None:
            nonlocal chunk_index, duplicate_count, rejected_count
            if not group:
                return
            combined = "\n\n".join(block.text for block in group)
            pages = sorted({block.page_number for block in group})
            slices = (
                [combined]
                if len(combined) <= self.max_size
                else split_text_with_overlap(combined, self.max_size, self.overlap)
            )
            for split_index, content in enumerate(slices):
                chunk = ChunkModel(
                    chunk_id=f"doc_{document.document_id}_chunk_{chunk_index:03d}",
                    document_id=document.document_id,
                    document_name=document.document_name,
                    section=section,
                    page_numbers=pages,
                    content=content,
                    source_url=document.source_url,
                    chunk_index=chunk_index,
                    metadata={"split_index": split_index} if len(slices) > 1 else {},
                )
                rejection = self._validate_chunk(
                    chunk, source_pages, existing_hashes, recent
                )
                if rejection is None:
                    chunks.append(chunk)
                    recent.append(chunk)
                    chunk_index += 1
                elif "Duplicate" in rejection:
                    duplicate_count += 1
                else:
                    rejected_count += 1

        for block in blocks:
            if block.block_type == "heading":
                emit(accumulated, active_section)
                accumulated = []
                current_size = 0
                active_section = block.text
                continue
            if current_size + len(block.text) > self.max_size and accumulated:
                emit(accumulated, active_section)
                carryover = []
                current_size = 0
                if self.overlap > 0:
                    tail = "\n\n".join(item.text for item in accumulated)[-self.overlap:]
                    first_space = tail.find(" ")
                    if 0 <= first_space < len(tail) - 1:
                        tail = tail[first_space + 1:]
                    tail = tail.strip()
                    if tail:
                        previous = accumulated[-1]
                        carryover = [BlockModel(
                            block_id=f"{previous.block_id}_overlap",
                            page_number=previous.page_number,
                            text=tail,
                            block_type="paragraph",
                            position_order=0,
                        )]
                        current_size = len(tail)
                accumulated = carryover
            accumulated.append(block)
            current_size += len(block.text)
        emit(accumulated, active_section)
        return chunks, duplicate_count, rejected_count

    def chunk_with_llm(self, document: DocumentModel) -> tuple:
        pages = [page for page in document.pages if page.extraction_status != "FAILED"]
        batch_size = getattr(self.config, "llm_batch_size", 2)
        chunks = []
        existing_hashes: Set[str] = set()
        recent: Deque[ChunkModel] = deque(maxlen=5)
        chunk_index = 1
        duplicate_count = rejected_count = llm_calls = 0
        system_prompt = (
            "Organize the supplied PDF blocks into coherent chunks. Return a JSON "
            "object with a 'chunks' array; each item must contain section, content, "
            "and source_pages. Never invent facts or page numbers."
        )
        for offset in range(0, len(pages), batch_size):
            batch = pages[offset:offset + batch_size]
            batch_pages = {page.page_number for page in batch}
            payload = [
                {
                    "block_id": block.block_id,
                    "page_number": block.page_number,
                    "type": block.block_type,
                    "text": block.text,
                }
                for page in batch
                for block in page.blocks
                if block.text.strip()
            ]
            if not payload:
                continue
            llm_calls += 1
            try:
                response = self.llm_client.generate_structured_completion(
                    system_prompt,
                    f"Here is the batch of text blocks to organize:\n"
                    f"{json.dumps(payload, indent=2)}",
                    temperature=0.0,
                )
                if not isinstance(response.get("chunks"), list):
                    raise ValueError("LLM response is missing the required 'chunks' array.")
                for raw in response["chunks"]:
                    chunk = ChunkModel(
                        chunk_id=(
                            f"doc_{document.document_id}_chunk_{chunk_index:03d}"
                        ),
                        document_id=document.document_id,
                        document_name=document.document_name,
                        section=raw.get("section") or "Unknown",
                        page_numbers=raw.get("source_pages") or [],
                        content=raw.get("content") or "",
                        source_url=document.source_url,
                        chunk_index=chunk_index,
                    )
                    rejection = self._validate_chunk(
                        chunk, batch_pages, existing_hashes, recent
                    )
                    if rejection is None:
                        chunks.append(chunk)
                        recent.append(chunk)
                        chunk_index += 1
                    elif "Duplicate" in rejection:
                        duplicate_count += 1
                    else:
                        rejected_count += 1
            except Exception:
                temporary = document.model_copy(update={
                    "pages": batch,
                    "total_pages": len(batch),
                    "successfully_extracted_pages": len(batch),
                    "failed_pages": 0,
                    "pages_requiring_ocr": 0,
                })
                fallback, fallback_duplicates, fallback_rejected = (
                    self.chunk_structurally(temporary)
                )
                duplicate_count += fallback_duplicates
                rejected_count += fallback_rejected
                for chunk in fallback:
                    chunks.append(chunk.model_copy(update={
                        "chunk_id": (
                            f"doc_{document.document_id}_chunk_{chunk_index:03d}"
                        ),
                        "chunk_index": chunk_index,
                    }))
                    chunk_index += 1
        return chunks, duplicate_count, rejected_count, llm_calls

    def _merge_small_chunks(self, chunks: List[ChunkModel]) -> tuple:
        if self.min_size <= 0 or len(chunks) <= 1:
            return chunks, 0
        ceiling = self.max_size * 1.3

        def combine(first: ChunkModel, second: ChunkModel) -> ChunkModel:
            section = (
                first.section
                if first.section == second.section
                else f"{first.section} / {second.section}"
            )
            return first.model_copy(update={
                "content": f"{first.content.strip()}\n\n{second.content.strip()}",
                "page_numbers": sorted(set(first.page_numbers) | set(second.page_numbers)),
                "section": section,
            })

        merged = []
        buffer = None
        merge_count = 0
        for chunk in chunks:
            if buffer is None:
                buffer = chunk
            elif (
                len(buffer.content) < self.min_size
                and buffer.section == chunk.section
                and len(buffer.content) + len(chunk.content) <= ceiling
            ):
                buffer = combine(buffer, chunk)
                merge_count += 1
            else:
                merged.append(buffer)
                buffer = chunk
        if buffer is not None:
            merged.append(buffer)
        if (
            len(merged) > 1
            and len(merged[-1].content) < self.min_size
            and merged[-2].section == merged[-1].section
        ):
            merged[-2] = combine(merged[-2], merged.pop())
            merge_count += 1
        for index, chunk in enumerate(merged, start=1):
            merged[index - 1] = chunk.model_copy(update={
                "chunk_index": index,
                "chunk_id": f"doc_{chunk.document_id}_chunk_{index:03d}",
            })
        return merged, merge_count

    def chunk_images(
        self, document: DocumentModel, start_index: int = 1
    ) -> List[ChunkModel]:
        chunks = []
        index = start_index
        minimum = getattr(self.config, "min_image_chunk_chars", 15)
        for page in document.pages:
            for image in page.images or []:
                text = (image.ocr_text or "").strip()
                if len(text) < minimum:
                    continue
                chunks.append(ChunkModel(
                    chunk_id=f"doc_{document.document_id}_chunk_{index:03d}",
                    document_id=document.document_id,
                    document_name=document.document_name,
                    section="Figure/Image",
                    page_numbers=[page.page_number],
                    content=f"[Figure/Image on page {page.page_number}]\n{text}",
                    source_url=document.source_url,
                    chunk_index=index,
                    content_kind="image_ocr",
                    metadata={
                        "chunk_type": "image",
                        "image_id": image.image_id,
                        "image_path": image.file_path,
                    },
                ))
                index += 1
        return chunks

    def enrich_reference_metadata(
        self, chunks: List[ChunkModel], document: DocumentModel
    ) -> int:
        if not getattr(self.config, "enable_reference_metadata", True):
            return 0
        page_index = build_page_reference_index([
            (page.page_number, page.text or "")
            for page in document.pages
            if page.extraction_status != "FAILED"
        ])
        annotated = 0
        for chunk in chunks:
            refs = extract_references(chunk.content or "")
            if not any((
                refs["figure_refs"], refs["table_refs"], refs["equation_refs"]
            )):
                continue
            chunk.metadata["figure_refs"] = refs["figure_refs"]
            chunk.metadata["table_refs"] = refs["table_refs"]
            if refs["equation_refs"]:
                chunk.metadata["equation_refs"] = refs["equation_refs"]
            if refs["figure_captions"]:
                chunk.metadata["figure_captions"] = refs["figure_captions"]
            if refs["table_captions"]:
                chunk.metadata["table_captions"] = refs["table_captions"]
            figure_pages = resolve_reference_pages(
                refs["figure_refs"], page_index["figures"]
            )
            table_pages = resolve_reference_pages(
                refs["table_refs"], page_index["tables"]
            )
            if figure_pages:
                chunk.metadata["figure_pages"] = figure_pages
            if table_pages:
                chunk.metadata["table_pages"] = table_pages
            annotated += 1
        return annotated

    def process_document(
        self, document: DocumentModel
    ) -> tuple[List[ChunkModel], ChunkingStatistics]:
        started = time.time()
        if getattr(self.config, "use_llm_assisted_organization", False):
            chunks, duplicates, rejected, llm_calls = self.chunk_with_llm(document)
        else:
            chunks, duplicates, rejected = self.chunk_structurally(document)
            llm_calls = 0
        if llm_calls:
            merged_count = 0
        else:
            chunks, merged_count = self._merge_small_chunks(chunks)
        image_chunks = self.chunk_images(document, len(chunks) + 1)
        chunks.extend(image_chunks)
        self.enrich_reference_metadata(chunks, document)
        sizes = [len(chunk.content) for chunk in chunks]
        statistics = ChunkingStatistics(
            total_input_pages=document.total_pages,
            total_input_blocks=sum(
                len(page.blocks)
                for page in document.pages
                if page.extraction_status != "FAILED"
            ),
            total_generated_chunks=len(chunks),
            average_chunk_size=sum(sizes) / len(sizes) if sizes else 0.0,
            minimum_chunk_size=min(sizes) if sizes else 0,
            maximum_chunk_size=max(sizes) if sizes else 0,
            chunks_spanning_multiple_pages=sum(
                len(chunk.page_numbers) > 1 for chunk in chunks
            ),
            rejected_chunks=rejected,
            duplicate_chunks_removed=duplicates,
            processing_duration_seconds=time.time() - started,
            llm_calls_made=llm_calls,
            merged_low_information_chunks=merged_count,
            image_chunks_generated=len(image_chunks),
        )
        return chunks, statistics

    def get_debug_representation(self, chunks: List[ChunkModel]) -> str:
        lines = []
        for chunk in chunks:
            lines.extend([
                "=" * 80,
                f"CHUNK {chunk.chunk_index:03d}",
                f"Document : {chunk.document_name}",
                f"Section  : {chunk.section or 'None'}",
                f"Pages    : {', '.join(map(str, chunk.page_numbers))}",
                "-" * 80,
                chunk.content,
                "=" * 80 + "\n",
            ])
        return "\n".join(lines)

    def save_chunks(
        self, chunks: List[ChunkModel], output_path: Optional[Path] = None
    ) -> Path:
        if output_path is None:
            output_dir = Path(self.config.extracted_data_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{Path(chunks[0].document_name).stem}_chunks.json"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                [chunk.model_dump() for chunk in chunks],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return output_path
