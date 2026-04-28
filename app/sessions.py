"""Ephemeral upload-session storage for ad-hoc document Q&A.

When a user uploads a PDF, we extract text, chunk it, build an in-memory BM25
index scoped to that document, and keep it alive under a session_id for the
lifetime of the process (with simple LRU-ish eviction).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from src.ingest.chunker import chunk_document
from src.ingest.pdf_to_text import dedupe_boilerplate_lines, extract_pdf_pages
from src.retrieval.bm25_baseline import BM25BaselineRetriever

LOGGER = logging.getLogger(__name__)

_MAX_SESSIONS = 16
_UPLOAD_DIR = Path("data/uploads")


class UploadSession:
    """A single uploaded document with its own BM25 retriever."""

    def __init__(
        self,
        session_id: str,
        doc_name: str,
        chunks: list[dict[str, Any]],
        retriever: BM25BaselineRetriever,
        source_path: Path,
    ) -> None:
        self.session_id = session_id
        self.doc_name = doc_name
        self.chunks = chunks
        self.retriever = retriever
        self.source_path = source_path
        self.created_at = time.time()
        self.last_used_at = self.created_at


class SessionStore:
    """Thread-safe in-memory session registry."""

    def __init__(self, max_sessions: int = _MAX_SESSIONS) -> None:
        self._sessions: dict[str, UploadSession] = {}
        self._lock = threading.Lock()
        self._max_sessions = max_sessions

    def create(self, pdf_bytes: bytes, filename: str) -> UploadSession:
        """Save a PDF upload, extract + chunk it, and register a session."""
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        session_id = uuid.uuid4().hex[:12]
        safe_stem = Path(filename).stem.replace("/", "_").replace(" ", "_") or "upload"
        target = _UPLOAD_DIR / f"{session_id}__{safe_stem}.pdf"
        target.write_bytes(pdf_bytes)

        LOGGER.info("Extracting text from uploaded PDF: %s (%d bytes)", target.name, len(pdf_bytes))
        pages = extract_pdf_pages(target)
        pages = dedupe_boilerplate_lines(pages)

        LOGGER.info("Chunking document...")
        chunks = chunk_document(
            pages,
            doc_name=safe_stem,
            source_path=str(target),
            min_words=300,
            max_words=800,
        )
        if not chunks:
            raise ValueError(
                "No text extracted from the uploaded PDF. It may be an image-only "
                "scan (requires OCR) or empty."
            )

        LOGGER.info("Building BM25 index over %d chunks", len(chunks))
        retriever = BM25BaselineRetriever(chunks)

        session = UploadSession(
            session_id=session_id,
            doc_name=safe_stem,
            chunks=chunks,
            retriever=retriever,
            source_path=target,
        )

        with self._lock:
            self._sessions[session_id] = session
            self._evict_if_needed()

        LOGGER.info("Session %s registered (%d chunks)", session_id, len(chunks))
        return session

    def get(self, session_id: str) -> UploadSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_used_at = time.time()
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        try:
            session.source_path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover — best-effort cleanup
            LOGGER.warning("Could not remove upload file %s: %s", session.source_path, exc)
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "doc_name": s.doc_name,
                    "num_chunks": len(s.chunks),
                    "created_at": s.created_at,
                }
                for s in sorted(
                    self._sessions.values(), key=lambda x: x.created_at, reverse=True
                )
            ]

    def _evict_if_needed(self) -> None:
        """Drop the least-recently-used session if we're over capacity."""
        if len(self._sessions) <= self._max_sessions:
            return
        victim_id = min(
            self._sessions, key=lambda sid: self._sessions[sid].last_used_at
        )
        victim = self._sessions.pop(victim_id)
        LOGGER.info("Evicting session %s to stay under cap", victim_id)
        try:
            victim.source_path.unlink(missing_ok=True)
        except OSError:
            pass


# Module-level singleton used by the API layer.
STORE = SessionStore()
