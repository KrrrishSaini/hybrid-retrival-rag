"""Hybrid retriever combining BM25 and dense retrieval with score fusion."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
from typing import Any

try:
    from .bm25_baseline import BM25BaselineRetriever as BM25Retriever
    from .bm25_baseline import DEFAULT_CHUNKS_PATH
    from .dense_baseline import DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME, DenseRetriever
    from .reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
except ImportError:  # Allows direct script-style execution.
    from bm25_baseline import BM25BaselineRetriever as BM25Retriever
    from bm25_baseline import DEFAULT_CHUNKS_PATH
    from dense_baseline import DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME, DenseRetriever
    from reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker

LOGGER = logging.getLogger(__name__)


def _min_max_normalize(scores_by_chunk: dict[str, float]) -> dict[str, float]:
    """Normalize scores into [0, 1] using min-max scaling."""
    if not scores_by_chunk:
        return {}
    values = list(scores_by_chunk.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        default_value = 1.0 if max_score > 0 else 0.0
        return {chunk_id: default_value for chunk_id in scores_by_chunk}
    scale = max_score - min_score
    return {chunk_id: (score - min_score) / scale for chunk_id, score in scores_by_chunk.items()}


class HybridRetriever:
    """Hybrid retriever: BM25 + Dense with weighted score fusion."""

    def __init__(
        self,
        *,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        index_dir: Path = DEFAULT_INDEX_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        alpha: float = 0.5,
        top_k_bm25: int = 20,
        top_k_dense: int = 20,
        batch_size: int = 64,
        device: str = "cpu",
        local_files_only: bool = False,
        prefer_faiss: bool = True,
        force_rebuild_dense: bool = False,
    ) -> None:
        """Initialize BM25 and dense retrievers."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be within [0, 1].")
        if top_k_bm25 <= 0 or top_k_dense <= 0:
            raise ValueError("top_k_bm25 and top_k_dense must be > 0.")

        self.alpha = alpha
        self.top_k_bm25 = top_k_bm25
        self.top_k_dense = top_k_dense

        LOGGER.info(
            "Initializing hybrid retriever (alpha=%.2f, top_k_bm25=%d, top_k_dense=%d)",
            alpha,
            top_k_bm25,
            top_k_dense,
        )

        self.bm25_retriever = BM25Retriever.from_jsonl(chunks_path)
        self.dense_retriever = DenseRetriever(
            chunks_path=chunks_path,
            index_dir=index_dir,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            local_files_only=local_files_only,
            prefer_faiss=prefer_faiss,
            force_rebuild=force_rebuild_dense,
        )

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieve top-k results after merging BM25 and dense candidates."""
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        bm25_results = self.bm25_retriever.retrieve(query=query, top_k=self.top_k_bm25)
        dense_results = self.dense_retriever.retrieve(query=query, top_k=self.top_k_dense)

        bm25_scores_raw = {
            str(item.get("chunk_id")): float(item.get("score", 0.0))
            for item in bm25_results
            if item.get("chunk_id")
        }
        dense_scores_raw = {
            str(item.get("chunk_id")): float(item.get("score", 0.0))
            for item in dense_results
            if item.get("chunk_id")
        }

        bm25_scores_norm = _min_max_normalize(bm25_scores_raw)
        dense_scores_norm = _min_max_normalize(dense_scores_raw)

        merged: dict[str, dict[str, Any]] = {}
        for item in bm25_results + dense_results:
            chunk_id = item.get("chunk_id")
            if not chunk_id:
                continue
            chunk_id = str(chunk_id)
            if chunk_id not in merged:
                merged[chunk_id] = {
                    "chunk_id": chunk_id,
                    "doc_name": item.get("doc_name"),
                    "section_id": item.get("section_id"),
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "text": item.get("text", ""),
                }

        fused_results: list[dict[str, Any]] = []
        for chunk_id, base in merged.items():
            bm25_norm = bm25_scores_norm.get(chunk_id, 0.0)
            dense_norm = dense_scores_norm.get(chunk_id, 0.0)
            final_score = self.alpha * bm25_norm + (1.0 - self.alpha) * dense_norm

            fused_results.append(
                {
                    **base,
                    "score": final_score,
                    "final_score": final_score,
                    "bm25_score": bm25_scores_raw.get(chunk_id, 0.0),
                    "dense_score": dense_scores_raw.get(chunk_id, 0.0),
                    "bm25_score_norm": bm25_norm,
                    "dense_score_norm": dense_norm,
                }
            )

        fused_results.sort(key=lambda item: float(item["final_score"]), reverse=True)
        return fused_results[:top_k]


def retrieve(
    query: str,
    top_k: int = 10,
    *,
    alpha: float = 0.5,
    top_k_bm25: int = 20,
    top_k_dense: int = 20,
) -> list[dict[str, Any]]:
    """Module-level convenience retrieval API."""
    retriever = HybridRetriever(alpha=alpha, top_k_bm25=top_k_bm25, top_k_dense=top_k_dense)
    return retriever.retrieve(query=query, top_k=top_k)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Hybrid retrieval (BM25 + Dense)")
    parser.add_argument("--query", type=str, required=True, help="Search query text.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of final merged results.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Fusion weight: final = alpha*bm25 + (1-alpha)*dense.",
    )
    parser.add_argument(
        "--top_k_bm25",
        type=int,
        default=20,
        help="BM25 candidate pool size before fusion.",
    )
    parser.add_argument(
        "--top_k_dense",
        type=int,
        default=20,
        help="Dense candidate pool size before fusion.",
    )
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
    parser.add_argument("--batch_size", type=int, default=64, help="Dense embedding batch size.")
    parser.add_argument("--device", type=str, default="cpu", help="Dense model device (default: cpu).")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load dense model strictly from local cache (no network).",
    )
    parser.add_argument(
        "--use_reranker",
        action="store_true",
        help="Enable cross-encoder reranking on top of hybrid candidates.",
    )
    parser.add_argument(
        "--top_k_hybrid_candidates",
        type=int,
        default=20,
        help="Candidate pool size to pass from hybrid retrieval to reranker.",
    )
    parser.add_argument(
        "--reranker_model_name",
        type=str,
        default=DEFAULT_RERANKER_MODEL,
        help="Cross-encoder reranker model name.",
    )
    parser.add_argument(
        "--reranker_batch_size",
        type=int,
        default=32,
        help="Cross-encoder reranking batch size.",
    )
    parser.add_argument("--no_faiss", action="store_true", help="Skip FAISS and use sklearn directly.")
    parser.add_argument(
        "--force_rebuild_dense",
        action="store_true",
        help="Force dense embedding cache rebuild.",
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
    """CLI entrypoint for hybrid retrieval."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        retriever = HybridRetriever(
            chunks_path=args.chunks_path,
            index_dir=args.index_dir,
            model_name=args.model_name,
            alpha=args.alpha,
            top_k_bm25=args.top_k_bm25,
            top_k_dense=args.top_k_dense,
            batch_size=args.batch_size,
            device=args.device,
            local_files_only=args.local_files_only,
            prefer_faiss=not args.no_faiss,
            force_rebuild_dense=args.force_rebuild_dense,
        )
        if args.use_reranker:
            candidate_k = max(args.top_k, args.top_k_hybrid_candidates)
            hybrid_candidates = retriever.retrieve(query=args.query, top_k=candidate_k)
            reranker = CrossEncoderReranker(
                model_name=args.reranker_model_name,
                batch_size=args.reranker_batch_size,
                device=args.device,
                local_files_only=args.local_files_only,
            )
            results = reranker.rerank(query=args.query, candidates=hybrid_candidates, top_k=args.top_k)
        else:
            results = retriever.retrieve(query=args.query, top_k=args.top_k)
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Query: {args.query}")
    print(f"Top K: {args.top_k}")
    print(f"Alpha: {args.alpha}")
    print(f"Candidates: bm25={args.top_k_bm25}, dense={args.top_k_dense}")
    print(f"Use Reranker: {args.use_reranker}")
    if args.use_reranker:
        print(f"Reranker: {args.reranker_model_name}")
        print(f"Hybrid Candidates for Rerank: {max(args.top_k, args.top_k_hybrid_candidates)}")
    print("-" * 130)

    for rank, item in enumerate(results, start=1):
        doc_name = item.get("doc_name") or "NA"
        section_id = item.get("section_id") or "NA"
        if args.use_reranker:
            print(
                f"{rank:>2}. rerank={item['rerank_score']:.4f} | hybrid={item['final_score']:.4f} "
                f"| bm25={item['bm25_score']:.4f} | dense={item['dense_score']:.4f} "
                f"| chunk_id={item['chunk_id']} | source={doc_name}/{section_id}"
            )
        else:
            print(
                f"{rank:>2}. final={item['final_score']:.4f} | bm25={item['bm25_score']:.4f} "
                f"| dense={item['dense_score']:.4f} | chunk_id={item['chunk_id']} "
                f"| source={doc_name}/{section_id}"
            )
        print(f"    snippet={_snippet(str(item.get('text', '')))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
