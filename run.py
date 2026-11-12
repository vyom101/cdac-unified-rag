#!/usr/bin/env python3
"""Unified C-DAC crawl, ingestion, indexing, retrieval, and reporting CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag.config import load_config

SOURCE_CHOICES = ["all", "html", "pdf", "uploaded_pdf", "crawled_pdf"]
MODE_CHOICES = ["bm25", "vector", "hybrid"]


def _print(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    print(json.dumps(value, indent=2, default=str))


def cmd_status(_args: argparse.Namespace) -> int:
    config = load_config()
    available = False
    error = None
    try:
        from rag.index.indexer import ElasticsearchIndexer
        available = bool(ElasticsearchIndexer(config).client.ping())
    except Exception as exc:
        error = str(exc)
    _print({
        "project": config.project_name,
        "kb_source": config.kb_source,
        "elasticsearch_url": config.elasticsearch_url,
        "elasticsearch_index": config.elasticsearch_index,
        "elasticsearch_available": available,
        "error": error,
    })
    return 0


def cmd_crawl(args: argparse.Namespace) -> int:
    import subprocess
    import sys
    import time

    from rag.models import utc_now_iso

    config = load_config()
    root = Path(__file__).resolve().parent
    max_attempts = getattr(args, "crawl_attempts", None) or config.crawl_whole_run_max_attempts
    interval = config.crawl_whole_run_retry_interval_seconds
    error_log = Path(config.crawl_error_log_path)
    last_rc = 1
    for attempt in range(1, max_attempts + 1):
        cmd = [
            sys.executable, "-m", "scrapy", "crawl", "cdac",
            "-s", f"CRAWL_SEED_URL={getattr(args, 'url', None) or config.crawl_start_url}",
            "-s", f"CRAWL_ALLOWED_DOMAIN={getattr(args, 'domain', None) or config.crawl_allowed_domain}",
            "-s", f"CLOSESPIDER_PAGECOUNT={getattr(args, 'max_urls', None) or config.crawl_max_urls}",
            "-s", f"CRAWL_RUN_ATTEMPT={attempt}",
        ]
        last_rc = subprocess.call(cmd, cwd=str(root))
        if last_rc == 0:
            return 0
        if attempt < max_attempts:
            error_log.parent.mkdir(parents=True, exist_ok=True)
            error_log.open("a", encoding="utf-8").write(
                json.dumps({
                    "event": "whole_crawl_retry_scheduled",
                    "failure_kind": "whole_crawl",
                    "timestamp": utc_now_iso(),
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "exit_code": last_rc,
                    "retry_after_seconds": interval,
                }) + "\n"
            )
            time.sleep(interval)
    return last_rc


def cmd_process_pdfs(_args: argparse.Namespace) -> int:
    from rag.ingestion import process_crawled_pdfs
    artifacts = process_crawled_pdfs(load_config())
    _print({"processed": len(artifacts), "source_kind": "crawled_pdf"})
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from rag.ingestion import ingest_uploaded_pdf
    artifact = ingest_uploaded_pdf(args.pdf, load_config())
    _print({
        "document_id": artifact.page_id,
        "document_name": artifact.title,
        "source_kind": "uploaded_pdf",
        "chunks": len(artifact.chunks),
    })
    return 0


def cmd_ingest_all(_args: argparse.Namespace) -> int:
    from rag.ingestion import ingest_all_uploaded_pdfs
    artifacts = ingest_all_uploaded_pdfs(load_config())
    _print({"processed": len(artifacts), "source_kind": "uploaded_pdf"})
    return 0


def cmd_index(_args: argparse.Namespace) -> int:
    from rag.ingestion import index_corpus
    _print(index_corpus(load_config()))
    return 0


def cmd_build_kg(_args: argparse.Namespace) -> int:
    from rag.graph.store import KnowledgeGraphStore
    from rag.ingestion import load_all_chunks
    config = load_config()
    snapshot = KnowledgeGraphStore(config.kg_store_path).ingest_chunks(
        load_all_chunks(config)
    )
    _print({"nodes": len(snapshot.nodes), "edges": len(snapshot.edges)})
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from rag.answer.grounding import grounding_failure
    from rag.services import search_corpus
    config = load_config()
    results = search_corpus(
        args.query, config, top_k=args.top_k, mode=args.mode,
        sources=args.sources, domain=args.domain,
    )
    reason = grounding_failure(args.query, list(results), config)
    _print({
        "query": args.query,
        "grounded": reason is None,
        "abstain_reason": reason,
        "sources": [result.model_dump() for result in results],
    })
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from rag.services import answer_question
    answer = answer_question(
        args.query, load_config(), top_k=args.top_k, mode=args.mode,
        sources=args.sources, domain=args.domain, include_kg=args.include_kg,
    )
    _print(answer)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from rag.eval.runner import run_evaluation
    config = load_config()
    _print(run_evaluation(config, args.benchmark, top_k=args.top_k, domain=args.domain))
    return 0


def cmd_benchmark_hnsw(args: argparse.Namespace) -> int:
    from rag.eval.hnsw_benchmark import run_hnsw_benchmark
    _print(run_hnsw_benchmark(load_config(), top_k=args.top_k))
    return 0


def cmd_validate_page(args: argparse.Namespace) -> int:
    from rag.report.validation import validate_page_trace, validate_pdf_trace
    config = load_config()
    result = (
        validate_pdf_trace(config, args.url)
        if args.pdf else validate_page_trace(config, args.url)
    )
    _print(result)
    return 0


def cmd_validate_links(args: argparse.Namespace) -> int:
    from rag.report.validation import validate_representative_links
    _print(validate_representative_links(load_config(), args.links))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from rag.report.reporting import build_pipeline_report, write_pipeline_report
    config = load_config()
    result = (
        write_pipeline_report(config, args.output)
        if args.output else build_pipeline_report(config)
    )
    _print(result)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from rag.ui import launch_ui
    launch_ui(load_config(), host="127.0.0.1", port=args.port)
    return 0


def _query_flags(parser: argparse.ArgumentParser, *, include_kg: bool = False) -> None:
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--mode", choices=MODE_CHOICES)
    parser.add_argument("--sources", choices=SOURCE_CHOICES)
    parser.add_argument("--domain")
    if include_kg:
        parser.add_argument("--include-kg", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C-DAC Unified RAG")
    commands = parser.add_subparsers(dest="command", required=True)
    handlers = {
        "status": cmd_status,
        "process-pdfs": cmd_process_pdfs, "ingest-all": cmd_ingest_all,
        "index": cmd_index, "build-kg": cmd_build_kg,
        "benchmark-hnsw": cmd_benchmark_hnsw, "report": cmd_report,
        "ui": cmd_ui,
    }
    for name, handler in handlers.items():
        sub = commands.add_parser(name)
        sub.set_defaults(func=handler)
    crawl = commands.add_parser("crawl")
    crawl.add_argument("--url")
    crawl.add_argument("--domain")
    crawl.add_argument("--max-urls", type=int)
    crawl.add_argument("--crawl-attempts", type=int)
    crawl.set_defaults(func=cmd_crawl)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--pdf", type=Path, required=True)
    ingest.set_defaults(func=cmd_ingest)
    query = commands.add_parser("query")
    _query_flags(query)
    query.set_defaults(func=cmd_query)
    ask = commands.add_parser("ask")
    _query_flags(ask, include_kg=True)
    ask.set_defaults(func=cmd_ask)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument(
        "--benchmark", type=Path,
        default=Path("rag/eval/fixtures/benchmark.jsonl"),
    )
    evaluate.add_argument("--top-k", type=int)
    evaluate.add_argument("--domain")
    evaluate.set_defaults(func=cmd_evaluate)
    benchmark = commands.choices["benchmark-hnsw"]
    benchmark.add_argument("--top-k", type=int, default=5)
    validate_page = commands.add_parser("validate-page")
    validate_page.add_argument("url")
    validate_page.add_argument("--pdf", action="store_true")
    validate_page.set_defaults(func=cmd_validate_page)
    validate_links = commands.add_parser("validate-links")
    validate_links.add_argument("--links", type=Path)
    validate_links.set_defaults(func=cmd_validate_links)
    report = commands.choices["report"]
    report.add_argument("--output", type=Path)
    ui = commands.choices["ui"]
    ui.add_argument("--port", type=int, default=8765)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
