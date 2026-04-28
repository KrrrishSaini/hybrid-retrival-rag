"""Retrieval evaluation: Recall@K, MRR@K, and nDCG@K for all retrieval methods."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from ..retrieval.bm25_baseline import DEFAULT_CHUNKS_PATH
    from ..retrieval.dense_baseline import DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME
    from ..retrieval.hybrid_retriever import HybridRetriever
    from ..retrieval.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
except ImportError:  # Allows direct script-style execution.
    from src.retrieval.bm25_baseline import DEFAULT_CHUNKS_PATH
    from src.retrieval.dense_baseline import DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker

LOGGER = logging.getLogger(__name__)
DEFAULT_QA_PATH = Path("data/qa_dataset.json")


def recall_at_k(gold_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    """Compute Recall@k as fraction of gold ids retrieved in top-k."""
    if k <= 0:
        raise ValueError("k must be > 0.")
    gold_set = {item for item in gold_ids if item}
    if not gold_set:
        return 0.0
    retrieved_top_k = set(retrieved_ids[:k])
    return len(gold_set & retrieved_top_k) / len(gold_set)


def mrr_at_k(gold_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    """Compute MRR@k based on first relevant rank in top-k."""
    if k <= 0:
        raise ValueError("k must be > 0.")
    gold_set = {item for item in gold_ids if item}
    if not gold_set:
        return 0.0
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(gold_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    """Compute nDCG@k (binary relevance: 1 if gold, 0 otherwise)."""
    if k <= 0:
        raise ValueError("k must be > 0.")
    gold_set = {item for item in gold_ids if item}
    if not gold_set:
        return 0.0

    # DCG: sum of 1/log2(rank+1) for each relevant result in top-k
    dcg = 0.0
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in gold_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: all relevant results at the top
    ideal_k = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_k + 1))

    return dcg / idcg if idcg > 0 else 0.0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval methods on QA gold chunk ids.")
    parser.add_argument("--qa_path", type=Path, default=DEFAULT_QA_PATH, help="Path to QA dataset JSON.")
    parser.add_argument(
        "--chunks_path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to chunk store JSONL.",
    )
    parser.add_argument(
        "--index_dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Path to dense index cache directory.",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Retrieval depth per method.")
    parser.add_argument("--alpha", type=float, default=0.6, help="Hybrid fusion alpha.")
    parser.add_argument("--top_k_bm25", type=int, default=20, help="Hybrid BM25 candidate pool.")
    parser.add_argument("--top_k_dense", type=int, default=20, help="Hybrid Dense candidate pool.")
    parser.add_argument(
        "--top_k_hybrid_candidates",
        type=int,
        default=20,
        help="Hybrid candidate pool size for reranking.",
    )
    parser.add_argument(
        "--use_reranker",
        action="store_true",
        help="Evaluate Hybrid+Reranker method.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Dense retriever sentence-transformers model.",
    )
    parser.add_argument(
        "--reranker_model_name",
        type=str,
        default=DEFAULT_RERANKER_MODEL,
        help="Cross-encoder reranker model.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Dense embedding batch size.")
    parser.add_argument("--reranker_batch_size", type=int, default=32, help="Reranker batch size.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for dense/reranker models.")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load model files strictly from local cache.",
    )
    parser.add_argument("--no_faiss", action="store_true", help="Skip FAISS and force sklearn backend.")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def load_qa_dataset(qa_path: Path) -> list[dict[str, Any]]:
    """Load and validate QA dataset entries."""
    if not qa_path.exists():
        LOGGER.warning("QA dataset file not found: %s", qa_path)
        return []

    try:
        with qa_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in QA file {qa_path}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError(f"QA dataset must be a JSON list: {qa_path}")

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            LOGGER.warning("Skipping QA entry %d: not an object", idx)
            continue
        sample_id = item.get("id")
        question = item.get("question")
        gold_ids = item.get("gold_chunk_ids")
        if not isinstance(sample_id, str) or not sample_id.strip():
            LOGGER.warning("Skipping QA entry %d: invalid id", idx)
            continue
        if not isinstance(question, str) or not question.strip():
            LOGGER.warning("Skipping QA entry %d: invalid question", idx)
            continue
        if not isinstance(gold_ids, list) or not all(isinstance(x, str) for x in gold_ids):
            LOGGER.warning("Skipping QA entry %d: invalid gold_chunk_ids", idx)
            continue
        records.append(
            {
                "id": sample_id.strip(),
                "question": question.strip(),
                "gold_chunk_ids": [x for x in gold_ids if x.strip()],
            }
        )
    return records


def _extract_chunk_ids(results: list[dict[str, Any]]) -> list[str]:
    """Extract valid chunk IDs from retrieval outputs."""
    ids: list[str] = []
    for item in results:
        chunk_id = item.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            ids.append(chunk_id)
    return ids


def _format_metric(value: float | None) -> str:
    """Format metric values for summary table."""
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _print_summary(metrics: dict[str, dict[str, float | None]]) -> None:
    """Print a clean summary table."""
    print("Method            | Recall@5 | MRR@10   | nDCG@10")
    print("---------------------------------------------------")
    for method in ["BM25", "Dense", "Hybrid", "Hybrid+Reranker"]:
        recall_value = metrics.get(method, {}).get("recall@5")
        mrr_value = metrics.get(method, {}).get("mrr@10")
        ndcg_value = metrics.get(method, {}).get("ndcg@10")
        print(
            f"{method:<17} | {_format_metric(recall_value):<8} "
            f"| {_format_metric(mrr_value):<8} | {_format_metric(ndcg_value):<8}"
        )


def main() -> int:
    """CLI entrypoint for retrieval evaluation."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    qa_samples = load_qa_dataset(args.qa_path)
    if not qa_samples:
        print(f"No QA samples found in {args.qa_path}. Add entries and rerun evaluation.")
        return 0

    if args.top_k <= 0:
        print("Error: --top_k must be > 0.")
        return 1

    metric_k_recall = 5
    metric_k_mrr = 10
    retrieval_k = max(args.top_k, metric_k_mrr)
    rerank_candidate_k = max(args.top_k_hybrid_candidates, retrieval_k)

    try:
        hybrid_retriever = HybridRetriever(
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
        )

        bm25_retriever = hybrid_retriever.bm25_retriever
        dense_retriever = hybrid_retriever.dense_retriever

        reranker: CrossEncoderReranker | None = None
        if args.use_reranker:
            reranker = CrossEncoderReranker(
                model_name=args.reranker_model_name,
                batch_size=args.reranker_batch_size,
                device=args.device,
                local_files_only=args.local_files_only,
            )
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1

    scores: dict[str, dict[str, list[float]]] = {
        "BM25": {"recall@5": [], "mrr@10": [], "ndcg@10": []},
        "Dense": {"recall@5": [], "mrr@10": [], "ndcg@10": []},
        "Hybrid": {"recall@5": [], "mrr@10": [], "ndcg@10": []},
        "Hybrid+Reranker": {"recall@5": [], "mrr@10": [], "ndcg@10": []},
    }

    for sample in qa_samples:
        question = sample["question"]
        gold_ids = sample["gold_chunk_ids"]

        bm25_ids = _extract_chunk_ids(bm25_retriever.retrieve(question, top_k=retrieval_k))
        dense_ids = _extract_chunk_ids(dense_retriever.retrieve(question, top_k=retrieval_k))
        hybrid_ids = _extract_chunk_ids(hybrid_retriever.retrieve(question, top_k=retrieval_k))

        scores["BM25"]["recall@5"].append(recall_at_k(gold_ids, bm25_ids, metric_k_recall))
        scores["BM25"]["mrr@10"].append(mrr_at_k(gold_ids, bm25_ids, metric_k_mrr))
        scores["BM25"]["ndcg@10"].append(ndcg_at_k(gold_ids, bm25_ids, metric_k_mrr))

        scores["Dense"]["recall@5"].append(recall_at_k(gold_ids, dense_ids, metric_k_recall))
        scores["Dense"]["mrr@10"].append(mrr_at_k(gold_ids, dense_ids, metric_k_mrr))
        scores["Dense"]["ndcg@10"].append(ndcg_at_k(gold_ids, dense_ids, metric_k_mrr))

        scores["Hybrid"]["recall@5"].append(recall_at_k(gold_ids, hybrid_ids, metric_k_recall))
        scores["Hybrid"]["mrr@10"].append(mrr_at_k(gold_ids, hybrid_ids, metric_k_mrr))
        scores["Hybrid"]["ndcg@10"].append(ndcg_at_k(gold_ids, hybrid_ids, metric_k_mrr))

        if reranker is not None:
            hybrid_candidates = hybrid_retriever.retrieve(question, top_k=rerank_candidate_k)
            reranked = reranker.rerank(query=question, candidates=hybrid_candidates, top_k=retrieval_k)
            rerank_ids = _extract_chunk_ids(reranked)
            scores["Hybrid+Reranker"]["recall@5"].append(
                recall_at_k(gold_ids, rerank_ids, metric_k_recall)
            )
            scores["Hybrid+Reranker"]["mrr@10"].append(
                mrr_at_k(gold_ids, rerank_ids, metric_k_mrr)
            )
            scores["Hybrid+Reranker"]["ndcg@10"].append(
                ndcg_at_k(gold_ids, rerank_ids, metric_k_mrr)
            )

    summary: dict[str, dict[str, float | None]] = {}
    for method, method_scores in scores.items():
        recall_values = method_scores["recall@5"]
        mrr_values = method_scores["mrr@10"]
        ndcg_values = method_scores["ndcg@10"]
        summary[method] = {
            "recall@5": mean(recall_values) if recall_values else None,
            "mrr@10": mean(mrr_values) if mrr_values else None,
            "ndcg@10": mean(ndcg_values) if ndcg_values else None,
        }

    print(f"QA samples evaluated: {len(qa_samples)}")
    print(f"Retrieval depth used: {retrieval_k}")
    if args.use_reranker:
        print(f"Reranker candidate depth: {rerank_candidate_k}")
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
