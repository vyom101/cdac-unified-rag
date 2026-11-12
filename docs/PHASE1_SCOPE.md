# Phase 1 Scope

## Included

- One corpus index, `rag_corpus_v1`, for crawled HTML, crawled PDFs, and uploaded PDFs.
- Scrapy crawling with bounded discovery, retries, accounting, and reports.
- Offline PDF extraction, OCR fallback, reference metadata, and chunking.
- Elasticsearch 8.12 BM25, vector, and hybrid retrieval.
- Source/domain filters, grounded answers, optional knowledge-graph augmentation.
- CLI workflows and a localhost-bound HTTP UI with `POST /ask`.
- Docker packaging for Elasticsearch, the application, and the UI.
- Stable tool-shaped functions: `search_corpus`, `search_kg`, `answer_question`.

## Excluded

- The legacy `pdf-rag` requests-based crawler.
- Agent/tool orchestration (LangGraph, AutoGen, CrewAI, routers, planners).
- Query rewriting, multi-turn chat memory, or a cross-encoder reranker.
- A new embedding model or a new search engine.
- A rewrite of Scrapy, the chunkers, the KG extractor, or the LLM client for cleanliness.
- Changes to `C:\CDAC\pdf-rag` or `C:\CDAC\cdac_crawler`.

## Acceptance

- Application port: `8765` (bound to `127.0.0.1`)
- Elasticsearch port: `9200`
- Unified index: `rag_corpus_v1`
- Unit-test check: `280 passed, 3 skipped`
- CLI-help check: passed
- Status-when-ES-down check: passed (exit 0)
- Container / live PDF `ask` check: not run (Docker engine not running)
- Tesseract check: not run (binary not on PATH)
