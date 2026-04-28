"""Build QA-style supervised training data from chunk corpus."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_CHUNKS_PATH = Path("data/chunks.json")
DEFAULT_OUT_PATH = Path("data/qa_training.txt")
FALLBACK_CHUNKS_JSONL = Path("data/chunks.jsonl")

ANSWER_KEYWORDS = (
    "eligible",
    "provides",
    "provide",
    "includes",
    "include",
    "covered",
    "coverage",
    "benefit",
    "service",
    "treatment",
    "enrol",
    "enroll",
)


def _normalize_whitespace(text: str) -> str:
    """Normalize repeated whitespace to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units."""
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately `max_tokens` words."""
    if max_tokens <= 0:
        return _normalize_whitespace(text)
    tokens = _normalize_whitespace(text).split()
    if len(tokens) <= max_tokens:
        return " ".join(tokens)
    return " ".join(tokens[:max_tokens])


def _resolve_chunks_path(chunks_path: Path) -> Path:
    """Resolve chunks path, with fallback from chunks.json to chunks.jsonl."""
    if chunks_path.exists():
        return chunks_path
    if chunks_path == DEFAULT_CHUNKS_PATH and FALLBACK_CHUNKS_JSONL.exists():
        LOGGER.warning(
            "Input %s not found. Falling back to %s.",
            chunks_path,
            FALLBACK_CHUNKS_JSONL,
        )
        return FALLBACK_CHUNKS_JSONL
    raise FileNotFoundError(f"Chunks file not found: {chunks_path}")


def _extract_rows_from_json(payload: Any) -> list[dict[str, Any]]:
    """Extract list rows from JSON payload."""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("chunks", "items", "data"):
            maybe = payload.get(key)
            if isinstance(maybe, list):
                return [row for row in maybe if isinstance(row, dict)]
    raise ValueError("Unsupported JSON structure. Expected list or object with chunks/items/data.")


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Load chunk rows from .json or .jsonl and map to canonical fields."""
    path = _resolve_chunks_path(chunks_path)
    if path.stat().st_size == 0:
        raise ValueError(f"Chunks file is empty: {path}")

    rows: list[dict[str, Any]] = []
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    row = json.loads(payload)
                except json.JSONDecodeError as exc:
                    LOGGER.warning("Skipping invalid JSONL line %d: %s", line_number, exc)
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = _extract_rows_from_json(payload)
    else:
        raise ValueError(
            f"Unsupported chunks file extension: {suffix}. Use .json or .jsonl."
        )

    if not rows:
        raise ValueError(f"No chunk rows found in: {path}")

    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        text = _normalize_whitespace(str(row.get("text", "")))
        if not text:
            LOGGER.warning("Skipping chunk row #%d with empty text.", idx)
            continue
        chunk_id = str(row.get("chunk_id", f"chunk_{idx:06d}")).strip() or f"chunk_{idx:06d}"
        source = str(
            row.get("source")
            or row.get("doc_name")
            or row.get("source_path")
            or "unknown-source"
        ).strip()
        section = str(
            row.get("section")
            or row.get("section_title")
            or row.get("section_id")
            or ""
        ).strip()
        page = row.get("page")
        if page is None:
            page = row.get("page_start")

        normalized.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "source": source,
                "section": section,
                "page": page,
            }
        )

    if not normalized:
        raise ValueError("All chunks were empty after normalization.")
    return normalized


def _infer_program_name(chunk: dict[str, Any]) -> str:
    """Infer a program/scheme name for question templates."""
    haystack = " ".join(
        [
            str(chunk.get("source", "")),
            str(chunk.get("section", "")),
            str(chunk.get("text", ""))[:300],
        ]
    ).lower()

    if "pmjay" in haystack or "ayushman" in haystack:
        return "PMJAY"
    if "ab-nhpm" in haystack or "nhpm" in haystack:
        return "AB-NHPM"
    if "pmkvy" in haystack:
        return "PMKVY"
    if "mid-day meal" in haystack or "mid day meal" in haystack:
        return "Mid-Day Meal Scheme"
    return "the scheme"


def generate_questions(chunk: dict[str, Any], max_questions: int = 3) -> list[str]:
    """Generate 1-3 rule-based questions for one chunk."""
    section = _normalize_whitespace(str(chunk.get("section", "")))
    program = _infer_program_name(chunk)

    candidates: list[str] = []
    if section:
        candidates.append(f"What is {section}?")
    candidates.append(f"Who is eligible for {program}?")
    candidates.append("What does the scheme provide?")
    candidates.append(f"How does {program} work?")
    candidates.append("What services are covered?")

    # De-duplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for question in candidates:
        q = _normalize_whitespace(question)
        if q and q not in seen:
            deduped.append(q)
            seen.add(q)

    limit = max(1, min(max_questions, 3))
    return deduped[:limit]


def extract_answer_from_chunk(text: str) -> str:
    """Extract answer sentences from keyword hits, with first-sent fallback."""
    sentences = _split_sentences(text)
    if not sentences:
        return "Not found in context."

    keyword_hits = [
        sentence
        for sentence in sentences
        if any(keyword in sentence.lower() for keyword in ANSWER_KEYWORDS)
    ]

    selected: list[str] = []
    for sentence in keyword_hits[:2]:
        if sentence not in selected:
            selected.append(sentence)
    if not selected:
        selected.extend(sentences[:2])

    answer = _normalize_whitespace(" ".join(selected))
    return answer if answer else "Not found in context."


def format_example(question: str, context: str, answer: str) -> str:
    """Format one QA training example."""
    return (
        "<question>\n"
        f"{question}\n"
        "</question>\n\n"
        "<context>\n"
        f"{context}\n"
        "</context>\n\n"
        "<answer>\n"
        f"{answer}\n"
        "</answer>"
    )


def build_qa_dataset(
    *,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    out_path: Path = DEFAULT_OUT_PATH,
    max_context_tokens: int = 512,
    max_questions_per_chunk: int = 3,
    min_target_examples: int = 500,
) -> tuple[int, int, float]:
    """Build QA-style supervised text dataset from chunks."""
    chunks = load_chunks(chunks_path)
    examples: list[str] = []
    token_counts: list[int] = []

    for chunk in chunks:
        context = _truncate_to_tokens(str(chunk.get("text", "")), max_context_tokens)
        context = _normalize_whitespace(context)
        if not context:
            continue
        answer = extract_answer_from_chunk(context)
        questions = generate_questions(chunk, max_questions=max_questions_per_chunk)

        for question in questions:
            qa = format_example(question=question, context=context, answer=answer)
            examples.append(qa)
            token_counts.append(len(_normalize_whitespace(qa).split()))

    if not examples:
        raise ValueError("No QA examples generated. Check chunk inputs.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("\n\n".join(examples))
        handle.write("\n")

    chunks_processed = len(chunks)
    pairs_generated = len(examples)
    avg_tokens = sum(token_counts) / pairs_generated

    print(f"Chunks processed: {chunks_processed}")
    print(f"QA pairs generated: {pairs_generated}")
    print(f"Average tokens per example: {avg_tokens:.2f}")
    if pairs_generated >= min_target_examples:
        print(f"Target check: met (>= {min_target_examples})")
    else:
        print(f"Target check: not met (< {min_target_examples}); generated as many as possible")
    print(f"Output written to: {out_path}")
    return chunks_processed, pairs_generated, avg_tokens


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Create QA-style supervised corpus from chunks.")
    parser.add_argument(
        "--chunks_path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to chunks file (.json or .jsonl). Default: data/chunks.json",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Output training file path.",
    )
    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=512,
        help="Maximum context token/word count per example.",
    )
    parser.add_argument(
        "--max_questions_per_chunk",
        type=int,
        default=3,
        help="Number of generated questions per chunk (1-3).",
    )
    parser.add_argument(
        "--min_target_examples",
        type=int,
        default=500,
        help="Target minimum QA examples for info reporting.",
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
        build_qa_dataset(
            chunks_path=args.chunks_path,
            out_path=args.out_path,
            max_context_tokens=args.max_context_tokens,
            max_questions_per_chunk=args.max_questions_per_chunk,
            min_target_examples=args.min_target_examples,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
