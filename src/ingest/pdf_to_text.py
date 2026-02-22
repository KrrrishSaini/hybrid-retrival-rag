"""PDF extraction and cleanup helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import logging
import re

from pypdf import PdfReader

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageText:
    """Page-level extracted text."""

    page_number: int
    text: str


def normalize_line(line: str) -> str:
    """Normalize whitespace for a single line."""
    compact = line.replace("\x00", " ")
    compact = re.sub(r"\s+", " ", compact)
    return compact.strip()


def extract_pdf_pages(pdf_path: Path) -> list[PageText]:
    """Extract text from all pages in a PDF."""
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive extraction fallback
            LOGGER.warning(
                "Failed text extraction on %s page %s: %s",
                pdf_path.name,
                page_idx,
                exc,
            )
            raw_text = ""

        cleaned_lines: list[str] = []
        for raw_line in raw_text.splitlines():
            normalized = normalize_line(raw_line)
            if normalized:
                cleaned_lines.append(normalized)
            elif cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")

        pages.append(PageText(page_number=page_idx, text="\n".join(cleaned_lines).strip()))

    return pages


def dedupe_boilerplate_lines(
    pages: list[PageText],
    *,
    min_occurrences: int = 3,
    page_ratio_threshold: float = 0.45,
) -> list[PageText]:
    """Remove repeated short lines that usually represent headers/footers."""
    if not pages:
        return pages

    line_pages: dict[str, set[int]] = defaultdict(set)

    for page in pages:
        unique_lines = {normalize_line(line) for line in page.text.splitlines() if normalize_line(line)}
        for line in unique_lines:
            if len(line) < 3 or len(line) > 120:
                continue
            if re.fullmatch(r"[-\d\s]+", line):
                continue
            line_pages[line].add(page.page_number)

    threshold = max(min_occurrences, int(len(pages) * page_ratio_threshold))
    boilerplate = {line for line, page_set in line_pages.items() if len(page_set) >= threshold}

    if not boilerplate:
        return pages

    cleaned_pages: list[PageText] = []
    for page in pages:
        kept: list[str] = []
        for line in page.text.splitlines():
            if normalize_line(line) in boilerplate:
                continue
            kept.append(line)

        compact: list[str] = []
        for line in kept:
            if line == "" and compact and compact[-1] == "":
                continue
            compact.append(line)

        cleaned_pages.append(PageText(page_number=page.page_number, text="\n".join(compact).strip()))

    LOGGER.info("Removed %d repeated boilerplate lines from %d pages", len(boilerplate), len(pages))
    return cleaned_pages


def write_extracted_text(pages: list[PageText], output_path: Path) -> None:
    """Write page-wise extracted text for debugging/inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for page in pages:
            handle.write(f"=== Page {page.page_number} ===\n")
            handle.write(page.text)
            handle.write("\n\n")
