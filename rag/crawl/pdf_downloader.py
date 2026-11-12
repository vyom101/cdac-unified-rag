"""PDF download storage during crawl (online discovery only)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rag.models import PdfRecord


def pdf_id_from_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def content_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PdfDownloadStore:
    """Persist downloaded PDF binaries and append-only manifest."""

    def __init__(self, pdf_dir: str, manifest_path: str):
        self.pdf_dir = Path(pdf_dir)
        self.manifest_path = Path(manifest_path)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def save(
        self, *, source_url: str, canonical_url: str, body: bytes,
        parent_url: str | None, sub_link: str | None, depth: int,
    ) -> PdfRecord:
        pdf_id = pdf_id_from_url(canonical_url)
        local_path = self.pdf_dir / f"{pdf_id}.pdf"
        local_path.write_bytes(body)
        record = PdfRecord(
            pdf_id=pdf_id,
            source_url=source_url,
            canonical_url=canonical_url,
            parent_url=parent_url,
            sub_link=sub_link,
            depth=depth,
            local_path=str(local_path),
            file_size=len(body),
            content_hash=content_hash_bytes(body),
        )
        with self.manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(record.model_dump_json() + "\n")
        return record

    def list_downloaded(self) -> list[PdfRecord]:
        if not self.manifest_path.exists():
            return []
        records: list[PdfRecord] = []
        seen: set[str] = set()
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = PdfRecord.model_validate_json(line)
            if record.pdf_id not in seen:
                seen.add(record.pdf_id)
                records.append(record)
        return records
