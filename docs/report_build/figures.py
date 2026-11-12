"""Workflow figures for the report.

Graphviz and Mermaid sources live in render_open_diagrams.mjs.
This module only invokes that renderer and returns the PNG paths.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

KEYS = (
    "architecture",
    "pipeline",
    "ingest",
    "extract",
    "ocr",
    "clean",
    "chunk",
    "lineage",
    "embed",
    "elastic",
    "bm25",
    "vector",
    "hnsw",
    "hybrid",
    "minmax",
    "context",
    "prompt",
    "citation",
    "query",
    "tests",
)


def build_all(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("render_open_diagrams.mjs")
    paths = {key: directory / f"{key}.png" for key in KEYS}
    script_mtime = script.stat().st_mtime
    stale = any(
        not path.is_file() or path.stat().st_mtime < script_mtime
        for path in paths.values()
    )
    if stale:
        subprocess.run(["node", str(script)], check=True, cwd=script.parent)
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"diagram renderer did not write {missing}")
    return paths
