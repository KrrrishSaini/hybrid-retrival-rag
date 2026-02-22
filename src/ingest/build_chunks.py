"""CLI entrypoint to build policy-document chunks from PDF files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

try:
    from .chunker import chunk_document
    from .pdf_to_text import dedupe_boilerplate_lines, extract_pdf_pages, write_extracted_text
except ImportError:  # Allows `python src/ingest/build_chunks.py`
    from chunker import chunk_document
    from pdf_to_text import dedupe_boilerplate_lines, extract_pdf_pages, write_extracted_text

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build section-aware chunks from policy PDFs.")
    parser.add_argument("--input_dir", type=Path, default=Path("data/raw_pdfs"), help="Folder containing input PDFs.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/chunks.jsonl"),
        help="Path to output JSONL chunk file.",
    )
    parser.add_argument(
        "--extracted_text_dir",
        type=Path,
        default=Path("data/extracted_text"),
        help="Folder for optional extracted text dumps.",
    )
    parser.add_argument("--write_extracted_text", action="store_true", help="Write page-wise extracted text files.")
    parser.add_argument("--limit_pdfs", type=int, default=None, help="Process only first N PDFs.")
    parser.add_argument("--min_words", type=int, default=300, help="Approximate minimum words per chunk.")
    parser.add_argument("--max_words", type=int, default=800, help="Approximate maximum words per chunk.")
    return parser.parse_args()


def list_pdf_files(input_dir: Path) -> list[Path]:
    """Find PDF files recursively in the input folder."""
    pdfs = sorted(input_dir.rglob("*.pdf"))
    pdfs.extend(sorted(input_dir.rglob("*.PDF")))
    unique = sorted({path.resolve() for path in pdfs})
    return unique


def write_jsonl(chunks: list[dict[str, object]], output_path: Path) -> None:
    """Write chunk records to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the chunk-building pipeline."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args.input_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list_pdf_files(args.input_dir)

    if args.limit_pdfs is not None:
        pdf_files = pdf_files[: args.limit_pdfs]

    all_chunks: list[dict[str, object]] = []

    for pdf_path in pdf_files:
        LOGGER.info("Processing %s", pdf_path.name)
        pages = extract_pdf_pages(pdf_path)
        pages = dedupe_boilerplate_lines(pages)

        if args.write_extracted_text:
            dump_path = args.extracted_text_dir / f"{pdf_path.stem}.txt"
            write_extracted_text(pages, dump_path)

        doc_chunks = chunk_document(
            pages,
            doc_name=pdf_path.stem,
            source_path=str(pdf_path),
            min_words=args.min_words,
            max_words=args.max_words,
        )
        all_chunks.extend(doc_chunks)

    write_jsonl(all_chunks, args.output_path)

    print(f"PDFs processed: {len(pdf_files)}")
    print(f"Chunks produced: {len(all_chunks)}")

    if all_chunks:
        sample = all_chunks[0]
        snippet = str(sample["text"]).replace("\n", " ")[:180]
        print(f"Sample chunk: {sample['chunk_id']} | {snippet}...")
    else:
        print("Sample chunk: N/A (no chunks generated; add PDFs to data/raw_pdfs)")


if __name__ == "__main__":
    main()
