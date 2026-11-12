"""Ask specific demo questions and save the live RAG responses."""

from __future__ import annotations

import json
from pathlib import Path

from rag.config import load_config
from rag.services import answer_question

QUESTIONS = [
    "What is C-DAC's Aadhaar Data Vault service?",
    "Where is the C-DAC Pune centre located?",
    "What has C-DAC done in High Performance Computing?",
    "What training does the C-DAC ACTS programme provide?",
    "What cyber forensics work does C-DAC describe?",
]

OUT = Path(__file__).resolve().parent


def main() -> None:
    config = load_config()
    summary = []
    for index, query in enumerate(QUESTIONS, start=1):
        print(f"ASK {index}: {query}", flush=True)
        answer = answer_question(query, config, include_kg=True)
        payload = answer.model_dump()
        path = OUT / f"q{index}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary.append(
            {
                "file": path.name,
                "query": payload["query"],
                "abstained": payload["abstained"],
                "chunks_retrieved": payload["chunks_retrieved"],
                "retrieval_mode": payload["retrieval_mode"],
            }
        )
        print(
            f"DONE {index} abstained={payload['abstained']} chunks={payload['chunks_retrieved']}",
            flush=True,
        )
    (OUT / "index.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
