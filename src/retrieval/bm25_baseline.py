"""BM25-only retrieval baseline over `data/chunks.jsonl`."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
from typing import Any

from rank_bm25 import BM25Okapi

LOGGER = logging.getLogger(__name__)
DEFAULT_CHUNKS_PATH = Path("data/chunks.jsonl")


def simple_tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, and split on whitespace."""
    normalized = text.lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.split() if normalized else []


def load_chunks_jsonl(chunks_path: Path) -> list[dict[str, Any]]:
    """Load chunk records from a JSONL file."""
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {chunks_path}. "
            "Run `python3 -m src.ingest.build_chunks` first."
        )
    if chunks_path.stat().st_size == 0:
        raise ValueError(
            f"Chunk file is empty: {chunks_path}. "
            "Run ingestion first to populate chunks."
        )

    chunks: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                item = json.loads(payload)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSON at line %d: %s", line_number, exc)
                continue
            if not isinstance(item, dict):
                LOGGER.warning("Skipping non-object JSON at line %d", line_number)
                continue
            chunks.append(item)

    if not chunks:
        raise ValueError(
            f"No valid chunk records found in: {chunks_path}. "
            "Check the file or re-run ingestion."
        )
    return chunks


class BM25BaselineRetriever:
    """A CPU-only BM25 retriever over chunked policy text."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        """Build a BM25 index from loaded chunk records."""
        if not chunks:
            raise ValueError("Cannot build BM25 index: no chunks provided.")
        self.chunks = chunks

        tokenized_corpus: list[list[str]] = []
        for chunk in chunks:
            tokens = simple_tokenize(str(chunk.get("text", "")))
            tokenized_corpus.append(tokens if tokens else ["__empty__"])

        self.bm25 = BM25Okapi(tokenized_corpus)
        LOGGER.info("Built BM25 index over %d chunks", len(self.chunks))

    @classmethod
    def from_jsonl(cls, chunks_path: Path = DEFAULT_CHUNKS_PATH) -> "BM25BaselineRetriever":
        """Construct retriever from a chunks JSONL file."""
        chunks = load_chunks_jsonl(chunks_path)
        LOGGER.info("Loaded %d chunks from %s", len(chunks), chunks_path)
        return cls(chunks)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieve top-k chunks for a query."""
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")
        query_tokens = simple_tokenize(query)
        if not query_tokens:
            raise ValueError("Query has no searchable tokens after tokenization.")

        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]

        results: list[dict[str, Any]] = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "score": float(score),
                    "doc_name": chunk.get("doc_name"),
                    "section_id": chunk.get("section_id"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "text": chunk.get("text", ""),
                }
            )
        return results


def retrieve(query: str, top_k: int) -> list[dict[str, Any]]:
    """Module-level convenience retrieval API."""
    retriever = BM25BaselineRetriever.from_jsonl(DEFAULT_CHUNKS_PATH)
    return retriever.retrieve(query=query, top_k=top_k)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="BM25 retrieval baseline over chunks.jsonl")
    parser.add_argument("--query", type=str, required=True, help="Search query text.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to return.")
    parser.add_argument(
        "--chunks_path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to chunks JSONL file.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def _snippet(text: str, max_chars: int = 280) -> str:
    """Create a compact one-line snippet for CLI display."""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def main() -> int:
    """CLI entrypoint for BM25 retrieval."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        retriever = BM25BaselineRetriever.from_jsonl(args.chunks_path)
        results = retriever.retrieve(query=args.query, top_k=args.top_k)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Query: {args.query}")
    print(f"Top K: {args.top_k}")
    print(f"Chunks Path: {args.chunks_path}")
    print("-" * 100)

    for rank, item in enumerate(results, start=1):
        doc_name = item.get("doc_name") or "NA"
        section_id = item.get("section_id") or "NA"
        page_start = item.get("page_start")
        page_end = item.get("page_end")
        print(
            f"{rank:>2}. score={item['score']:.4f} | chunk_id={item.get('chunk_id')} "
            f"| source={doc_name}/{section_id} | pages={page_start}-{page_end}"
        )
        print(f"    snippet={_snippet(str(item.get('text', '')))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
