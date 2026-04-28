"""Minimal FastAPI app for the Chat-with-Policy-Docs RAG system.

Start the server:
    uvicorn app.api:app --reload --port 8000

Endpoints:
    GET  /health          — liveness check
    POST /query           — RAG query → grounded answer with citations
    GET  /chunks          — list available documents
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Force offline mode if models are already cached — prevents HuggingFace
# network check from hanging on slow/blocked connections.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Configure logging BEFORE imports so we see progress during heavy imports.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
    force=True,
)
LOGGER = logging.getLogger("app.api")
LOGGER.info("Starting app.api module — importing heavy dependencies...")
_import_t0 = time.time()

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

LOGGER.info("  fastapi/pydantic imported (%.1fs)", time.time() - _import_t0)

from src.llm.pretrained_generate import PretrainedGenerator  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.retrieval.reranker import CrossEncoderReranker  # noqa: E402

from app.sessions import STORE as SESSION_STORE  # noqa: E402

LOGGER.info("  project modules imported (total %.1fs)", time.time() - _import_t0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload all models on startup so first request is fast."""
    LOGGER.info("Preloading retriever...")
    _get_retriever()
    LOGGER.info("Preloading reranker...")
    _get_reranker()
    LOGGER.info("Preloading generator (Qwen2.5-0.5B-Instruct)...")
    _get_generator()
    LOGGER.info("✓ All models loaded. Server ready at http://127.0.0.1:8000")
    yield


app = FastAPI(
    title="Policy-Docs RAG API",
    description="Ask questions about government policy documents using hybrid retrieval + custom LLM.",
    version="0.1.0",
    lifespan=lifespan,
)

# Serve the web UI from app/static/
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for /query endpoint."""

    question: str = Field(..., min_length=1, description="User question.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")
    alpha: float = Field(default=0.6, ge=0.0, le=1.0, description="Hybrid fusion weight.")
    use_reranker: bool = Field(default=True, description="Apply cross-encoder reranking.")
    doc_filter: str | None = Field(default=None, description="Restrict to chunks from this document.")
    session_id: str | None = Field(
        default=None,
        description="If set, query against an uploaded document (session) "
        "instead of the policy corpus.",
    )
    max_new_tokens: int = Field(default=200, ge=1, le=512)
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)


class UploadResponse(BaseModel):
    """Response body for /upload endpoint."""

    session_id: str
    doc_name: str
    num_chunks: int
    filename: str


class ChunkInfo(BaseModel):
    """Chunk metadata in query response."""

    chunk_id: str
    doc_name: str | None
    section_id: str | None
    score: float
    snippet: str


class QueryResponse(BaseModel):
    """Response body for /query endpoint."""

    question: str
    answer: str
    citations: list[str]
    faithfulness: float
    ungrounded_sentences: list[str]
    chunks: list[ChunkInfo]


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_retriever: HybridRetriever | None = None
_reranker: CrossEncoderReranker | None = None
_generator: PretrainedGenerator | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        LOGGER.info("Loading hybrid retriever...")
        _retriever = HybridRetriever(alpha=0.6, device="cpu")
    return _retriever


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        LOGGER.info("Loading cross-encoder reranker...")
        _reranker = CrossEncoderReranker(device="cpu")
    return _reranker


def _get_generator() -> PretrainedGenerator:
    global _generator
    if _generator is None:
        LOGGER.info("Loading pretrained LLM (Qwen2.5-0.5B-Instruct)...")
        _generator = PretrainedGenerator(device="auto")
    return _generator


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root():
    """Serve the web UI (falls back to API docs if static files missing)."""
    if _STATIC_DIR.is_dir() and (_STATIC_DIR / "index.html").exists():
        return RedirectResponse(url="/static/index.html")
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/chunks")
def list_documents():
    """List unique document names available in the chunk store."""
    retriever = _get_retriever()
    doc_names = sorted({str(c.get("doc_name", "")) for c in retriever.bm25_retriever.chunks})
    return {"documents": doc_names, "total_chunks": len(retriever.bm25_retriever.chunks)}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """Accept a PDF upload, extract + chunk it, and open a query session."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(pdf_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50 MB).")

    try:
        session = SESSION_STORE.create(pdf_bytes=pdf_bytes, filename=file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        LOGGER.exception("Error processing upload")
        raise HTTPException(status_code=500, detail=f"Failed to ingest PDF: {exc}")

    return UploadResponse(
        session_id=session.session_id,
        doc_name=session.doc_name,
        num_chunks=len(session.chunks),
        filename=file.filename,
    )


@app.get("/sessions")
def list_sessions():
    """List active upload sessions."""
    return {"sessions": SESSION_STORE.list_sessions()}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Drop an upload session and its underlying file."""
    if not SESSION_STORE.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"deleted": session_id}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Run RAG pipeline: retrieve → generate (pretrained LLM) → respond."""
    try:
        # Route: session-scoped (uploaded doc) vs. global policy corpus
        if req.session_id:
            session = SESSION_STORE.get(req.session_id)
            if session is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {req.session_id} not found or evicted. Please re-upload.",
                )
            # BM25 over the uploaded doc, optionally reranked
            candidate_k = max(req.top_k * 4, 20) if req.use_reranker else req.top_k
            candidates = session.retriever.retrieve(
                query=req.question, top_k=candidate_k
            )
            if req.use_reranker and candidates:
                reranker = _get_reranker()
                chunks = reranker.rerank(
                    query=req.question, candidates=candidates, top_k=req.top_k
                )
            else:
                chunks = candidates[: req.top_k]
        else:
            retriever = _get_retriever()
            if req.use_reranker:
                reranker = _get_reranker()
                candidates = retriever.retrieve(
                    query=req.question, top_k=30, doc_filter=req.doc_filter
                )
                chunks = reranker.rerank(
                    query=req.question, candidates=candidates, top_k=req.top_k
                )
            else:
                chunks = retriever.retrieve(
                    query=req.question, top_k=req.top_k, doc_filter=req.doc_filter
                )

        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant chunks found.")

        generator = _get_generator()
        answer = generator.generate_answer(
            question=req.question,
            chunks=chunks,
            max_new_tokens=req.max_new_tokens,
        )

        # Build citation list from chunk IDs used as context
        citations = [str(c.get("chunk_id", "")) for c in chunks if c.get("chunk_id")]

        chunk_infos = [
            ChunkInfo(
                chunk_id=str(c.get("chunk_id", "")),
                doc_name=c.get("doc_name"),
                section_id=c.get("section_id"),
                score=float(c.get("score", c.get("rerank_score", 0.0))),
                snippet=str(c.get("text", ""))[:200],
            )
            for c in chunks
        ]

        return QueryResponse(
            question=req.question,
            answer=answer,
            citations=citations,
            faithfulness=1.0,
            ungrounded_sentences=[],
            chunks=chunk_infos,
        )

    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(exc))
