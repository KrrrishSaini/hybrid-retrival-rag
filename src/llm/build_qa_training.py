"""Build QA-style training text from QA set and chunk evidence."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_QA_PATH = Path("data/qa_dataset.json")
DEFAULT_CHUNKS_PATH = Path("data/chunks.jsonl")
DEFAULT_OUT_PATH = Path("data/qa_training.txt")
EXAMPLE_DELIMITER = "\n\n<END_EXAMPLE>\n\n"

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
}


def load_qa_items(qa_path: Path) -> list[dict[str, Any]]:
    """Load QA items from JSON file supporting list or wrapped object formats."""
    if not qa_path.exists():
        raise FileNotFoundError(f"QA dataset not found: {qa_path}")
    if qa_path.stat().st_size == 0:
        raise ValueError(f"QA dataset is empty: {qa_path}")

    with qa_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    items: Any = payload
    if isinstance(payload, dict):
        for key in ("items", "data", "questions"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break

    if not isinstance(items, list):
        raise ValueError("QA dataset must be a list, or contain list key: items/data/questions.")

    valid_items: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            LOGGER.warning("Skipping malformed QA entry #%d (not an object).", idx)
            continue
        valid_items.append(item)

    if not valid_items:
        raise ValueError(f"No valid QA entries found in: {qa_path}")
    return valid_items


def load_chunks_by_id(chunks_path: Path) -> dict[str, dict[str, Any]]:
    """Load chunk JSONL and index by chunk_id."""
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    if chunks_path.stat().st_size == 0:
        raise ValueError(f"Chunks file is empty: {chunks_path}")

    chunk_map: dict[str, dict[str, Any]] = {}
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid chunks.jsonl line %d: %s", line_number, exc)
                continue
            if not isinstance(row, dict):
                continue
            chunk_id = str(row.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            chunk_map[chunk_id] = row

    if not chunk_map:
        raise ValueError(f"No usable chunks found in: {chunks_path}")
    return chunk_map


def _split_sentences(text: str) -> list[str]:
    """Split text into coarse sentence-like units."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences


def synthesize_answer(question: str, context: str, max_sentences: int = 2) -> str:
    """Generate fallback answer from top context sentences by token overlap."""
    sentences = _split_sentences(context)
    if not sentences:
        return "Not found in context."

    query_tokens = {
        tok
        for tok in re.findall(r"[a-z0-9]+", question.lower())
        if tok not in _STOPWORDS and len(tok) > 1
    }
    if not query_tokens:
        return " ".join(sentences[:max_sentences]).strip()

    scored: list[tuple[int, int, str]] = []
    for idx, sentence in enumerate(sentences):
        sent_tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        score = len(query_tokens & sent_tokens)
        scored.append((score, idx, sentence))

    scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    top = [row for row in scored if row[0] > 0][:max_sentences]
    if not top:
        return " ".join(sentences[:max_sentences]).strip()

    top.sort(key=lambda row: row[1])
    return " ".join(row[2] for row in top).strip()


def build_example(question: str, context: str, answer: str) -> str:
    """Render a single QA-style training example in target format.

    The ``<END>`` token marks the boundary of the answer so the model
    learns to stop generating instead of drifting into continuation or
    echoing the context.
    """
    return (
        "You are a helpful assistant. Answer ONLY using the context. "
        'If the answer is not in the context, say "Not found in context."\n\n'
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "ANSWER:\n"
        f"{answer}\n<END>"
    )


def build_training_corpus(
    *,
    qa_path: Path = DEFAULT_QA_PATH,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    out_path: Path = DEFAULT_OUT_PATH,
    max_context_chars: int = 2000,
) -> int:
    """Build QA training text file from QA entries and evidence chunks."""
    qa_items = load_qa_items(qa_path)
    chunks_by_id = load_chunks_by_id(chunks_path)

    examples: list[str] = []
    missing_chunk_refs = 0

    for entry_idx, item in enumerate(qa_items, start=1):
        question = str(item.get("question", "")).strip()
        if not question:
            LOGGER.warning("Skipping QA entry #%d (missing question).", entry_idx)
            continue

        raw_ids = item.get("gold_chunk_ids", [])
        if not isinstance(raw_ids, list):
            LOGGER.warning("QA entry #%d has non-list gold_chunk_ids. Treating as empty.", entry_idx)
            raw_ids = []

        context_parts: list[str] = []
        for chunk_id_obj in raw_ids:
            chunk_id = str(chunk_id_obj).strip()
            if not chunk_id:
                continue
            chunk_row = chunks_by_id.get(chunk_id)
            if chunk_row is None:
                missing_chunk_refs += 1
                LOGGER.warning("Missing chunk_id in chunks store: %s (QA entry #%d)", chunk_id, entry_idx)
                continue
            chunk_text = str(chunk_row.get("text", "")).strip()
            if chunk_text:
                context_parts.append(chunk_text)

        context = "\n\n".join(context_parts).strip()
        if max_context_chars > 0 and len(context) > max_context_chars:
            context = context[:max_context_chars].rstrip()

        gold_answer = str(item.get("gold_answer", "")).strip()
        answer = gold_answer if gold_answer else synthesize_answer(question=question, context=context)
        if not answer:
            answer = "Not found in context."
        if not context:
            context = "Not found in context."

        examples.append(build_example(question=question, context=context, answer=answer))

    if not examples:
        raise ValueError("No training examples were built from the QA dataset.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(EXAMPLE_DELIMITER.join(examples))
        handle.write("\n")

    LOGGER.info("Wrote QA training corpus to %s", out_path)
    if missing_chunk_refs:
        LOGGER.warning("Total missing chunk references: %d", missing_chunk_refs)
    print(f"Built {len(examples)} training examples")
    print(f"Output written to: {out_path}")
    return len(examples)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Build QA-style training text for mini LLM.")
    parser.add_argument(
        "--qa_path",
        type=Path,
        default=DEFAULT_QA_PATH,
        help="Path to QA dataset JSON.",
    )
    parser.add_argument(
        "--chunks_path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to chunk store JSONL.",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Output .txt training corpus path.",
    )
    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=2000,
        help="Maximum context characters per QA example.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        build_training_corpus(
            qa_path=args.qa_path,
            chunks_path=args.chunks_path,
            out_path=args.out_path,
            max_context_chars=args.max_context_chars,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
