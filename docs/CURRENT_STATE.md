# Current State

## Ported from `cdac_crawler`

- Scrapy spider, URL canonicalization, depth/lineage, robots/sitemap discovery
- Interval retries, success.log / error.log, human-review flags, crawl ledger
- HTML section extraction and section-aware chunking
- Chunk summaries and composite summaries
- PDF download during crawl
- Domain filter on retrieval
- Knowledge graph build, store, and lookup (schema unchanged)
- Evaluation runner (`bm25_only`, `vector_only`, `hybrid`, `knowledge_graph`) and HNSW benchmark
- Page / representative-link validation and pipeline report
- Inspector UI on `127.0.0.1:8765` with `POST /ask`
- Docker Compose for Elasticsearch 8.12.2
- `GroundingValidator` abstain rules: no hits, score below threshold, role mismatch

## Ported from `pdf-rag`

- Local PDF ingest of `data/raw` (single file and directory)
- PDF text extraction with page and block structure
- OCR for scanned pages and embedded images (graceful skip if Tesseract is missing)
- Figure/table/equation reference metadata and query-time reference/image boosts
- Hybrid candidate pool before truncation to `top_k`
- `MockLLMClient` and `CustomAPILLMClient`
- RAG prompt rules, including the exact insufficient-context sentence
- Near-duplicate chunk filtering and PDF min-chunk merge (same-section only)

The pdf-rag requests crawler was not ported. Scrapy is the only crawler.

## Unified

- One package: `rag`
- One index: `rag_corpus_v1` (does not reuse `cdac_rag_index` or `cdac_website_chunks_v1`)
- Document model includes `source_kind` (`html` | `crawled_pdf` | `uploaded_pdf`) and website lineage fields
- HTML chunking: max 1200, overlap 150, min 80, composite group size 10
- PDF chunking (crawled and uploaded): max 1200, overlap 100, min 300, duplicate similarity 0.92
- Crawled PDFs go through the pdf-rag extraction/OCR path, then receive crawl lineage
- Retriever factory: `bm25` | `vector` | `hybrid` (default hybrid 0.3/0.7 min-max, pool 20, top_k 5)
- Source filter: `all` | `html` | `pdf` | `uploaded_pdf` | `crawled_pdf`
- Domain filter applies to `html` and `crawled_pdf`; uploaded PDFs stay visible
- Tool-shaped functions in `rag.services`: `search_corpus`, `search_kg`, `answer_question`
- `kb.source`: `html_only` | `pdf_only` | `html_and_pdf` | `uploaded_pdf_only` | `all` (default `all`)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`, dimension 384
- LLM: mock by default; `custom_api` when `LLM_ENDPOINT` is set

## Tests

Ported unit tests keep their original assertions except for import paths. The pdf-rag config default test was updated so the unified defaults match this repo (`rag_corpus_v1`, PDF chunk 1200/100/300, project name `C-DAC Unified RAG`). Validation messages from pdf-rag (`RETRIEVAL_MODE must be one of`, empty `ELASTICSEARCH_URL`, weight sum, secrets masking) are preserved.

Added unified contract tests for shared mapping/`source_kind` filters, abstain-without-LLM, prompt contents, HTML/PDF citations, `include_kg`, and role-mismatch abstention.

## Check status

- `python -m pytest tests -q`: **280 passed, 3 skipped**
- `python run.py --help`: **passed (exit 0)**
- `python run.py status`: **passed (exit 0)**; Elasticsearch was down and the command reported `elasticsearch_available: false`
- Docker live ingest + `ask`: **not run**. `docker` and `docker compose` are installed, but the Docker Desktop engine was not running (`dockerDesktopLinuxEngine` pipe missing).
- Tesseract OCR live check: **not run**. `tesseract` is not on PATH. OCR still degrades gracefully and flags `requires_ocr`.
- Source projects: **not modified**. This work only wrote under `C:\CDAC\RAG`. `C:\CDAC\pdf-rag` and `C:\CDAC\cdac_crawler` were read, never edited.

## Phase 2

Not implemented. Phase 2 will call `search_corpus`, `search_kg`, and `answer_question` as agent tools.
