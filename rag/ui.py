"""Minimal pipeline inspector UI using the standard-library HTTP server."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from rag.config import AppConfig
from rag.graph.store import KnowledgeGraphStore
from rag.report.reporting import build_pipeline_report
from rag.report.validation import validate_page_trace
from rag.retrieve.factory import get_retriever
from rag import services


def _html_page(body: str) -> bytes:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>C-DAC Pipeline Inspector</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
pre {{ background: #f6f8fa; padding: 12px; overflow: auto; }}
nav a {{ margin-right: 12px; }}
</style></head><body>
<nav><a href="/">Dashboard</a><a href="/graph">Graph</a></nav>
{body}
</body></html>""".encode("utf-8")


def launch_ui(config: AppConfig, host: str = "127.0.0.1", port: int = 8765) -> None:
    kg_store = KnowledgeGraphStore(config.kg_store_path)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(
                    "<h1>Pipeline Dashboard</h1>"
                    f"<pre>{json.dumps(build_pipeline_report(config), indent=2)}</pre>"
                )
            elif parsed.path == "/graph":
                graph = json.dumps(kg_store.snapshot.model_dump(), indent=2)[:12000]
                self._send_html(f"<h1>Knowledge Graph</h1><pre>{graph}</pre>")
            elif parsed.path == "/validate":
                url = parse_qs(parsed.query).get("url", [""])[0]
                if not url:
                    self._send_html("<p>Provide ?url=...</p>")
                    return
                self._send_html(
                    "<h1>Page Validation</h1>"
                    f"<pre>{json.dumps(validate_page_trace(config, url), indent=2)}</pre>"
                )
            elif parsed.path == "/search":
                query = parse_qs(parsed.query).get("q", [""])[0]
                if not query:
                    self._send_html("<p>Provide ?q=...</p>")
                    return
                retriever = get_retriever(config)
                payload = {
                    "elasticsearch": [result.model_dump() for result in retriever.search(query)],
                    "knowledge_graph": [result.model_dump() for result in kg_store.search(query)],
                }
                self._send_html(
                    f"<h1>Search Compare</h1><pre>{json.dumps(payload, indent=2)}</pre>"
                )
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/ask":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                answer = services.answer_question(
                    payload.get("query", ""),
                    config,
                    top_k=payload.get("top_k"),
                    mode=payload.get("mode"),
                    sources=payload.get("sources"),
                    domain=payload.get("domain"),
                    include_kg=bool(payload.get("include_kg", False)),
                )
                self._send_json(answer.model_dump())
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json({"error": str(error)}, status=400)
            except Exception as error:
                self._send_json({"error": str(error)}, status=500)

        def _send_html(self, body: str):
            payload = _html_page(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, body, status: int = 200):
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):  # noqa: A003
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"C-DAC UI running at http://{host}:{port}")
    server.serve_forever()
