"""Dense retrieval baseline with FAISS-first indexing and sklearn fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
from time import time
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)

DEFAULT_CHUNKS_PATH = Path("data/chunks.jsonl")
DEFAULT_INDEX_DIR = Path("data/indexes/dense")
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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


def sha256_file(path: Path) -> str:
    """Compute a stable SHA256 hash for file content."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors row-wise."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


class DenseRetriever:
    """Dense vector retriever with FAISS and sklearn backends."""

    def __init__(
        self,
        *,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        index_dir: Path = DEFAULT_INDEX_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 64,
        device: str = "cpu",
        local_files_only: bool = False,
        prefer_faiss: bool = True,
        force_rebuild: bool = False,
    ) -> None:
        """Initialize retriever, cache, and vector index."""
        self.chunks_path = chunks_path
        self.index_dir = index_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.local_files_only = local_files_only
        self.prefer_faiss = prefer_faiss

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.metadata_path = self.index_dir / "chunks_metadata.jsonl"
        self.manifest_path = self.index_dir / "manifest.json"

        self.chunks = load_chunks_jsonl(self.chunks_path)
        self.chunks_hash = sha256_file(self.chunks_path)
        LOGGER.info(
            "Loaded %d chunks from %s (sha256=%s...)",
            len(self.chunks),
            self.chunks_path,
            self.chunks_hash[:10],
        )

        LOGGER.info(
            "Loading embedding model: %s (device=%s, local_files_only=%s)",
            self.model_name,
            self.device,
            self.local_files_only,
        )
        self.model = self._load_model()

        self.embeddings = self._load_or_build_embeddings(force_rebuild=force_rebuild)
        self.index_backend = ""
        self.index: Any = None
        self._build_index()

    def _encode_texts(self, texts: list[str], *, show_progress: bool) -> np.ndarray:
        """Encode text to normalized float32 embeddings."""
        encode_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "convert_to_numpy": True,
            "show_progress_bar": show_progress,
        }
        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True, **encode_kwargs)
        except TypeError:
            embeddings = self.model.encode(texts, **encode_kwargs)
            embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return l2_normalize(matrix)

    def _load_model(self) -> SentenceTransformer:
        """Load sentence-transformers model with local-cache retry."""
        if self.local_files_only:
            return SentenceTransformer(self.model_name, device=self.device, local_files_only=True)

        try:
            return SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            LOGGER.warning(
                "Online model load failed for %s; retrying with local cache only. Reason: %s",
                self.model_name,
                exc,
            )
            try:
                return SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except Exception as local_exc:
                raise RuntimeError(
                    f"Failed to load model '{self.model_name}'. "
                    "Download it once with network access or pass a local model path."
                ) from local_exc

    def _cache_valid(self) -> bool:
        """Check whether cached embeddings can be reused."""
        if not self.manifest_path.exists():
            return False
        if not self.embeddings_path.exists():
            return False
        if not self.metadata_path.exists():
            return False

        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return False

        return (
            manifest.get("chunks_sha256") == self.chunks_hash
            and manifest.get("model_name") == self.model_name
            and manifest.get("chunk_count") == len(self.chunks)
        )

    def _write_cache(self, embeddings: np.ndarray) -> None:
        """Persist embeddings, metadata, and manifest."""
        np.save(self.embeddings_path, embeddings)

        with self.metadata_path.open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        manifest = {
            "chunks_path": str(self.chunks_path.resolve()),
            "chunks_sha256": self.chunks_hash,
            "chunks_mtime": self.chunks_path.stat().st_mtime,
            "chunk_count": len(self.chunks),
            "model_name": self.model_name,
            "embedding_dim": int(embeddings.shape[1]),
            "created_at_unix": int(time()),
            "backend_preference": "faiss_then_sklearn" if self.prefer_faiss else "sklearn_only",
        }
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    def _load_or_build_embeddings(self, *, force_rebuild: bool) -> np.ndarray:
        """Load cached embeddings or rebuild from chunk text."""
        if not force_rebuild and self._cache_valid():
            LOGGER.info("Using cached embeddings from %s", self.embeddings_path)
            cached = np.load(self.embeddings_path).astype(np.float32)
            if cached.ndim != 2 or cached.shape[0] != len(self.chunks):
                LOGGER.warning("Cached embeddings shape mismatch. Rebuilding cache.")
            else:
                return l2_normalize(cached)

        LOGGER.info("Building dense embeddings for %d chunks", len(self.chunks))
        texts = [str(chunk.get("text", "")) for chunk in self.chunks]
        embeddings = self._encode_texts(texts, show_progress=True)
        self._write_cache(embeddings)
        LOGGER.info("Saved embeddings cache to %s", self.index_dir)
        return embeddings

    def _build_index(self) -> None:
        """Build vector index with FAISS first, then sklearn fallback."""
        if self.prefer_faiss:
            try:
                import faiss  # type: ignore

                index = faiss.IndexFlatIP(self.embeddings.shape[1])
                index.add(self.embeddings.astype(np.float32))
                self.index = index
                self.index_backend = "faiss"
                LOGGER.info("Using FAISS backend (IndexFlatIP)")
                return
            except Exception as exc:
                LOGGER.warning("FAISS unavailable; falling back to sklearn. Reason: %s", exc)

        try:
            from sklearn.neighbors import NearestNeighbors
        except Exception as exc:
            raise ImportError(
                "scikit-learn is required for dense fallback indexing. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        nn_index = NearestNeighbors(metric="cosine", algorithm="brute")
        nn_index.fit(self.embeddings)
        self.index = nn_index
        self.index_backend = "sklearn"
        LOGGER.info("Using sklearn NearestNeighbors backend (cosine)")

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieve top-k chunks for a query."""
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self._encode_texts([query], show_progress=False).astype(np.float32)
        k = min(top_k, len(self.chunks))

        results: list[dict[str, Any]] = []
        if self.index_backend == "faiss":
            scores, indices = self.index.search(query_embedding, k)
            ranked_indices = indices[0]
            ranked_scores = scores[0]
        else:
            distances, indices = self.index.kneighbors(query_embedding, n_neighbors=k)
            ranked_indices = indices[0]
            ranked_scores = 1.0 - distances[0]

        for idx, score in zip(ranked_indices, ranked_scores):
            if int(idx) < 0:
                continue
            chunk = self.chunks[int(idx)]
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


def retrieve(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Module-level convenience retrieval API."""
    retriever = DenseRetriever()
    return retriever.retrieve(query=query, top_k=top_k)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Dense retrieval baseline over chunks.jsonl")
    parser.add_argument("--query", type=str, required=True, help="Search query text.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results to return.")
    parser.add_argument(
        "--chunks_path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to chunks JSONL file.",
    )
    parser.add_argument(
        "--index_dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Path to dense index cache directory.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Sentence-transformers model name.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Embedding model device (default: cpu).",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load embedding model strictly from local cache (no network).",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force rebuilding embeddings cache.",
    )
    parser.add_argument(
        "--no_faiss",
        action="store_true",
        help="Skip FAISS and use sklearn directly.",
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
    """CLI entrypoint for dense retrieval."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        retriever = DenseRetriever(
            chunks_path=args.chunks_path,
            index_dir=args.index_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            local_files_only=args.local_files_only,
            prefer_faiss=not args.no_faiss,
            force_rebuild=args.force_rebuild,
        )
        results = retriever.retrieve(query=args.query, top_k=args.top_k)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Query: {args.query}")
    print(f"Top K: {args.top_k}")
    print(f"Chunks Path: {args.chunks_path}")
    print(f"Index Dir: {args.index_dir}")
    print(f"Model: {args.model_name}")
    print(f"Backend: {retriever.index_backend}")
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
