"""Section-aware chunking utilities for policy documents."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

try:
    from .pdf_to_text import PageText
except ImportError:  # Allows direct script-style imports.
    from pdf_to_text import PageText


@dataclass
class LineRecord:
    """A single line of text with page provenance."""

    text: str
    page_number: int


@dataclass
class SectionRecord:
    """Grouped lines belonging to one detected section."""

    section_title: Optional[str]
    section_id: Optional[str]
    lines: list[LineRecord]


@dataclass
class ParagraphRecord:
    """Paragraph-level text and source page span."""

    text: str
    page_start: Optional[int]
    page_end: Optional[int]

    @property
    def word_count(self) -> int:
        """Approximate token count with whitespace-separated words."""
        return len(self.text.split())


SECTION_PATTERNS = [
    re.compile(r"^(?:SECTION|Section|Sec\.)\s+([A-Za-z0-9][A-Za-z0-9.\-]*)\s*[:\-]?\s*(.*)$"),
    re.compile(r"^(?:CHAPTER|Chapter|PART|Part)\s+([A-Za-z0-9IVXLC]+)\s*[:\-]?\s*(.*)$"),
    re.compile(r"^((?:\d+(?:\.\d+){0,4}|[IVXLC]+))[\.)]\s+(.+)$"),
]


def slugify(value: str) -> str:
    """Convert a value into a filesystem-safe slug."""
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered)
    slug = slug.strip("-")
    return slug or "unknown"


def detect_heading(line: str) -> tuple[Optional[str], Optional[str]]:
    """Detect common section heading styles."""
    candidate = re.sub(r"\s+", " ", line).strip()
    if not candidate:
        return None, None

    for pattern in SECTION_PATTERNS:
        match = pattern.match(candidate)
        if match:
            section_id = match.group(1).strip() if match.group(1) else None
            if len(match.groups()) >= 2:
                maybe_title = match.group(2).strip() if match.group(2) else ""
                title = maybe_title if maybe_title else candidate
            else:
                title = candidate
            return section_id, title

    letters = [char for char in candidate if char.isalpha()]
    if 4 <= len(letters) <= 120 and len(candidate.split()) <= 14:
        upper_ratio = sum(char.isupper() for char in letters) / len(letters)
        if upper_ratio >= 0.8:
            return None, candidate

    return None, None


def build_sections(pages: list[PageText]) -> list[SectionRecord]:
    """Group lines into sections using heading detection."""
    sections: list[SectionRecord] = []
    current = SectionRecord(section_title=None, section_id=None, lines=[])

    for page in pages:
        for raw_line in page.text.splitlines():
            line = raw_line.strip()
            if not line:
                if current.lines and current.lines[-1].text != "":
                    current.lines.append(LineRecord(text="", page_number=page.page_number))
                continue

            section_id, section_title = detect_heading(line)
            if section_id is not None or section_title is not None:
                if current.lines:
                    sections.append(current)
                current = SectionRecord(section_title=section_title, section_id=section_id, lines=[])
                current.lines.append(LineRecord(text=line, page_number=page.page_number))
                continue

            current.lines.append(LineRecord(text=line, page_number=page.page_number))

    if current.lines:
        sections.append(current)

    return sections


def _lines_to_paragraphs(lines: list[LineRecord]) -> list[ParagraphRecord]:
    """Convert line records into paragraph records while preserving page spans."""
    paragraphs: list[ParagraphRecord] = []
    buffer: list[LineRecord] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(item.text for item in buffer).strip()
        if text:
            paragraphs.append(
                ParagraphRecord(
                    text=text,
                    page_start=min(item.page_number for item in buffer),
                    page_end=max(item.page_number for item in buffer),
                )
            )
        buffer.clear()

    for line in lines:
        if line.text == "":
            flush()
            continue
        buffer.append(line)
    flush()

    return paragraphs


def _split_long_paragraph(paragraph: ParagraphRecord, max_words: int) -> list[ParagraphRecord]:
    """Split very long paragraphs into fixed-size word windows."""
    if paragraph.word_count <= max_words:
        return [paragraph]

    words = paragraph.text.split()
    out: list[ParagraphRecord] = []
    for start in range(0, len(words), max_words):
        segment_words = words[start : start + max_words]
        out.append(
            ParagraphRecord(
                text=" ".join(segment_words),
                page_start=paragraph.page_start,
                page_end=paragraph.page_end,
            )
        )
    return out


def chunk_document(
    pages: list[PageText],
    *,
    doc_name: str,
    source_path: str,
    min_words: int = 300,
    max_words: int = 800,
) -> list[dict[str, object]]:
    """Create section-aware chunks for one document."""
    sections = build_sections(pages)
    chunks: list[dict[str, object]] = []

    doc_slug = slugify(doc_name)
    chunk_counter = 1

    for section_index, section in enumerate(sections, start=1):
        paragraphs = _lines_to_paragraphs(section.lines)
        expanded: list[ParagraphRecord] = []
        for paragraph in paragraphs:
            expanded.extend(_split_long_paragraph(paragraph, max_words=max_words))

        if not expanded:
            continue

        section_key = section.section_id or section.section_title or f"sec-{section_index}"
        section_slug = slugify(section_key)

        current_text: list[str] = []
        current_words = 0
        page_start: Optional[int] = None
        page_end: Optional[int] = None

        for paragraph in expanded:
            would_overflow = current_words + paragraph.word_count > max_words
            if current_text and would_overflow and current_words >= min_words:
                chunk_id = f"{doc_slug}__{section_slug}__p{page_start or 0}-{page_end or 0}__c{chunk_counter:04d}"
                chunks.append(
                    {
                        "doc_name": doc_name,
                        "source_path": source_path,
                        "page_start": page_start,
                        "page_end": page_end,
                        "section_title": section.section_title,
                        "section_id": section.section_id,
                        "chunk_id": chunk_id,
                        "text": "\n\n".join(current_text).strip(),
                    }
                )
                chunk_counter += 1
                current_text = []
                current_words = 0
                page_start = None
                page_end = None

            current_text.append(paragraph.text)
            current_words += paragraph.word_count
            if paragraph.page_start is not None:
                page_start = paragraph.page_start if page_start is None else min(page_start, paragraph.page_start)
            if paragraph.page_end is not None:
                page_end = paragraph.page_end if page_end is None else max(page_end, paragraph.page_end)

        if current_text:
            chunk_id = f"{doc_slug}__{section_slug}__p{page_start or 0}-{page_end or 0}__c{chunk_counter:04d}"
            chunks.append(
                {
                    "doc_name": doc_name,
                    "source_path": source_path,
                    "page_start": page_start,
                    "page_end": page_end,
                    "section_title": section.section_title,
                    "section_id": section.section_id,
                    "chunk_id": chunk_id,
                    "text": "\n\n".join(current_text).strip(),
                }
            )
            chunk_counter += 1

    return chunks
