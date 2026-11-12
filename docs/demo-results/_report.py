"""Build a readable HTML report and a markdown copy of the saved answers."""

from __future__ import annotations

import html
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def sources_html(sources: list[dict]) -> str:
    cards = []
    for index, source in enumerate(sources[:5], start=1):
        url = source.get("source_url") or ""
        cards.append(
            "<article class='source'>"
            f"<h3>Source {index}</h3>"
            f"<p><strong>{esc(source.get('document_name') or 'Untitled')}</strong></p>"
            f"<p>{esc(source.get('section') or '')}</p>"
            f"<p class='meta'>{esc(source.get('source_kind'))} · "
            f"{esc(source.get('retrieval_method'))} · score {esc(source.get('score'))}</p>"
            + (f"<p class='url'>{esc(url)}</p>" if url else "")
            + "</article>"
        )
    return "\n".join(cards)


def page(query: str, answer: str, meta: str, sources: str, nav: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{esc(query)}</title>
<style>
body {{ margin: 0; background: #f4f6f8; color: #1b2430; font-family: Segoe UI, Arial, sans-serif; }}
main {{ max-width: 980px; margin: 0 auto; padding: 32px 28px 48px; }}
.kicker {{ letter-spacing: .08em; text-transform: uppercase; color: #0b6e4f; font-size: 13px; font-weight: 700; }}
h1 {{ font-size: 32px; line-height: 1.25; margin: 8px 0 12px; }}
.meta {{ color: #526070; }}
.answer {{ background: white; border: 1px solid #d9e1e8; border-radius: 12px; padding: 22px 24px; white-space: pre-wrap; font-size: 18px; line-height: 1.45; }}
h2 {{ margin-top: 28px; font-size: 20px; }}
.sources {{ display: grid; gap: 12px; }}
.source {{ background: white; border: 1px solid #d9e1e8; border-radius: 12px; padding: 14px 16px; }}
.source h3 {{ margin: 0 0 6px; font-size: 15px; color: #0b6e4f; }}
.source p {{ margin: 4px 0; }}
.url {{ color: #1d4e89; word-break: break-all; font-size: 14px; }}
nav a {{ color: #1d4e89; margin-right: 14px; }}
</style></head><body><main>
<p class="kicker">C-DAC Unified RAG · Llama 3.2 · hybrid retrieval</p>
<h1>{esc(query)}</h1>
<p class="meta">{esc(meta)}</p>
<div class="answer">{esc(answer)}</div>
<h2>Retrieved sources</h2>
<div class="sources">{sources}</div>
<nav>{nav}</nav>
</main></body></html>
"""


def main() -> None:
    files = sorted(OUT.glob("q*.json"))
    markdown = ["# C-DAC RAG demo answers", "", "Model: llama3.2 through Ollama. Retrieval: hybrid, knowledge graph included.", ""]
    links = []
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    for index, payload in enumerate(payloads, start=1):
        links.append(f"<a href='q{index}.html'>Q{index}</a>")
    for index, payload in enumerate(payloads, start=1):
        query = payload["query"]
        answer = payload["answer"]
        meta = (
            f"Abstained: {payload['abstained']} · chunks retrieved: {payload['chunks_retrieved']} · "
            f"mode: {payload['retrieval_mode']} · sources: {payload['sources_filter']}"
        )
        html_path = OUT / f"q{index}.html"
        html_path.write_text(
            page(query, answer, meta, sources_html(payload.get("sources") or []), " ".join(links)),
            encoding="utf-8",
        )
        markdown.extend([f"## {index}. {query}", "", meta, "", answer, "", "### Sources", ""])
        for source_index, source in enumerate((payload.get("sources") or [])[:5], start=1):
            markdown.append(
                f"{source_index}. {source.get('document_name') or 'Untitled'} — {source.get('section') or ''} "
                f"({source.get('source_kind')}, score {source.get('score')})"
            )
            if source.get("source_url"):
                markdown.append(f"   {source['source_url']}")
        markdown.append("")
    (OUT / "responses.md").write_text("\n".join(markdown), encoding="utf-8")
    print("REPORT_DONE")


if __name__ == "__main__":
    main()
